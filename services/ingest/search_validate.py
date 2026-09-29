"""Search validation for time-sensitive knowledge nodes (Phase 4).

Reuses the SSRF-safe ``WebSearchTool`` and the embedding client for a bounded,
evidence-recording validation pass. Writer/fetch paths stay on the safe fetcher
in ``core.net``; results are stored as ``search_evidence`` rows — they never
overwrite or delete the authoritative source node. A web outage or network
error never rolls back the ingestion write (best-effort).
"""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert as pg_insert

from core import database as db
from core.config import settings
from core.logging import get_logger
from core.vllm import VLLMClient
from services.tools.web import WebSearchTool

logger = get_logger(__name__)

VERIFIED = "VERIFIED"
UPDATED = "UPDATED"
UNCHANGED = "UNCHANGED"
CONFLICTED = "CONFLICTED"
UNVERIFIED = "UNVERIFIED"

# Source types that may be web-validated; local/authoritative sources are never.
# ``web_learn`` is what the deep-learn crawler writes, so it belongs here too.
_VALIDATED_SOURCE_TYPES = {"web", "rss", "web_learn"}

# Freshness classes eligible for validation.
_VALIDATED_FRESHNESS = {"TIME_SENSITIVE", "REALTIME"}

# Subtle-but-changed / still-valid cues that separate UPDATED (web has more
# recent info) from UNCHANGED (still accurate).
_CHANGE_CUES = (
    "no longer", "has been", "was replaced", "discontinued", "now supports",
    "as of 202", "released ", "updated ", "new version", "has changed",
    "không còn", "đã thay thế", "hiện đã",
)
_UNCHANGED_CUES = (
    "remains", "unchanged", "still valid", "same", "no change",
    "vẫn còn", "không thay đổi",
)


def cosine_sim(a: list[float] | None, b: list[float] | None) -> float:
    """Cosine similarity between two embedding vectors, in ``[-1, 1]``.

    Returns ``0.0`` on any length mismatch so a failed embedding never looks
    like a strong CONFLICT (0 sim is well below the conflict threshold).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _pick_verdict(sim: float, web_text: str) -> str:
    """Deterministic verdict from cosine + change cues (no network/LLM)."""
    if not web_text or not web_text.strip():
        return UNVERIFIED
    if sim >= settings.search_validation_verify_threshold:
        return VERIFIED
    if sim >= settings.search_validation_conflict_threshold:
        low = web_text.lower()
        if any(cue in low for cue in _CHANGE_CUES) and not any(
            cue in low for cue in _UNCHANGED_CUES
        ):
            return UPDATED
        return UNCHANGED
    return CONFLICTED


class SearchValidator:
    """Validate time-sensitive nodes against bounded, SSRF-safe web lookups."""

    def __init__(self, vllm: VLLMClient | None = None, web_tool: WebSearchTool | None = None):
        self.vllm = vllm
        self.web_tool = web_tool

    # ------------------------------------------------------------------
    # Gating
    # ------------------------------------------------------------------

    @staticmethod
    def should_search(freshness: str | None, source_type: str | None) -> bool:
        """Whether this node qualifies for web validation (cheap, no IO)."""
        if not settings.search_validation_enabled:
            return False
        return (
            freshness in _VALIDATED_FRESHNESS
            and (source_type or "").lower() in _VALIDATED_SOURCE_TYPES
        )

    @staticmethod
    def query_for_chunk(text: str, *, section_path: str | None = None,
                        keywords: list[str] | None = None) -> str:
        """A compact search query — never the raw chunk.

        Prefers explicit keywords, then the section path, then the first few
        short content tokens. Bounded so the query stays focused.
        """
        if keywords:
            top = [k for k in keywords if isinstance(k, str) and len(k) >= 3][:4]
            if top:
                return " ".join(top)
        if section_path and section_path.strip():
            segs = [s for s in section_path.strip().split("/") if s.strip()]
            return " ".join(segs[:3])
        words = [w for w in text.split()][:8]
        return " ".join(words) if words else text[:120]

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def validate_node(
        self,
        *,
        node_content: str,
        search_query: str,
        document_id=None,
        chunk_id=None,
        unit_id=None,
    ) -> dict:
        """Validate one node; returns the verdict record (may be deselected).

        Preserves the input node — the web text is stored only as an
        ``external_update_candidate`` / ``web_fetched_text`` column. Bounded to
        ``search_validation_max_outbound`` results and one page fetch.
        Returns ``None``-like ``{verdict: UNVERIFIED, ...}`` on any failure.
        """
        if self.web_tool is None:
            return self._noop()
        try:
            results = await self.web_tool.search(
                search_query, max_results=settings.search_validation_max_results
            )
            if not results:
                return self._noop()
            top = results[0]
            url = str(top.get("url", ""))
            snippet = str(top.get("snippet", ""))[:1000]

            web_text = ""
            if settings.search_validation_fetch_page and url:
                try:
                    page = await self.web_tool.fetch_page_text(
                        url,
                        max_chars=4000,
                        max_bytes=1_500_000,
                        timeout_seconds=8.0,
                    )
                    web_text = str(page.get("text", ""))[:4000]
                except Exception as exc:  # unsafe URL / http / size — degrade
                    logger.warning("searchval.fetch_page_failed", error=str(exc))
            compare_text = web_text or snippet

            # Embed input + web text once for the cosine verdict.
            sim = 0.0
            if self.vllm is not None and compare_text:
                try:
                    input_emb = await self.vllm.embed(node_content[:3000])
                    web_emb = await self.vllm.embed(compare_text[:3000])
                    sim = cosine_sim(input_emb, web_emb)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("searchval.embed_failed", error=str(exc))
                    sim = 0.0

            verdict = _pick_verdict(sim, compare_text)
            if not compare_text:
                verdict = UNVERIFIED

            record = {
                "verdict": verdict,
                "similarity": round(sim, 4),
                "query": search_query,
                "url": url,
                "snippet": snippet or None,
                "web_text": web_text or None,
                "conflict_sentence": self._conflict_sentence(verdict, node_content, web_text),
                "candidate": web_text[:2000] or None,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "unit_id": unit_id,
            }
            return record
        except Exception as exc:  # pragma: no cover - defensive, never fatal
            logger.warning("searchval.failed", error=str(exc))
            return self._noop()

    async def persist(self, session, record: dict) -> None:
        """Persist one ``search_evidence`` row (idempotent per chunk+query)."""
        if not record or record.get("verdict") not in (VERIFIED, UPDATED, CONFLICTED,
                                                       UNCHANGED):
            return
        stmt = (
            pg_insert(db.SearchEvidence)
            .values(
                document_id=record.get("document_id"),
                chunk_id=record.get("chunk_id"),
                unit_id=record.get("unit_id"),
                search_query=record.get("query") or "",
                web_source_url=record.get("url"),
                web_snippet=record.get("snippet"),
                web_fetched_text=record.get("web_text"),
                verdict=record.get("verdict") or UNVERIFIED,
                conflict_sentence=record.get("conflict_sentence"),
                external_update_candidate=record.get("candidate"),
                similarity_input_vs_web=record.get("similarity"),
                model_version=settings.embed_model,
            )
            .on_conflict_do_update(
                index_elements=["chunk_id", "search_query"],
                set_={"verdict": record.get("verdict") or UNVERIFIED},
            )
        )
        await session.execute(stmt)

    @staticmethod
    def _conflict_sentence(verdict: str, node_content: str, web_text: str) -> str | None:
        if verdict != CONFLICTED:
            return None
        # Keep a short, data-preserving pointer — never overwrite the source.
        node_lead = " ".join(node_content.split())[:160]
        web_lead = " ".join(web_text.split())[:160]
        return f"input: {node_lead} || web: {web_lead}"

    @staticmethod
    def _noop() -> dict:
        return {
            "verdict": UNVERIFIED,
            "similarity": 0.0,
            "query": "",
            "url": None,
            "snippet": None,
            "web_text": None,
            "conflict_sentence": None,
            "candidate": None,
            "document_id": None,
            "chunk_id": None,
            "unit_id": None,
        }