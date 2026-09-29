"""Context assembly for the unified RAG pipeline (Brain Second Phase 10).

Takes the re-ranked chunk list from ``FeatureReranker`` and turns it into a
single, token-bounded context string suitable for the LLM prompt.

Assembly rules:

1. **Deduplication** — drop chunks whose content is near-identical to a chunk
   already kept (token-level Jaccard).
2. **Temporal ordering** — for temporal queries, sort by ``valid_from`` (or
   ``created_at`` fallback) ascending; otherwise keep the reranker score order.
3. **Grouping / separation** — chunks are prefixed with source headers and a
   separator is placed between chunks for clarity.
4. **Citations** — each excerpt is suffixed with ``[citation:<chunk_id>]`` so
   the LLM can refer to evidence without inventing IDs.
5. **Dynamic slots** — the available token budget is split across chunks using a
   rank-decayed allocation; no single source can dominate the budget.

The assembler is deterministic, fully mocked, and does not call any LLM.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.rag.analyzer import QueryProfile


# Approximate characters per token for the simple local models we run.
_CHAR_PER_TOKEN = 4

# Default limits.
_DEFAULT_MAX_TOKENS = 2048
_DEFAULT_MAX_CHUNK_TOKENS = 512


@dataclass
class AssembledContext:
    """Result of context assembly."""

    text: str
    included_chunk_ids: list[str] = field(default_factory=list)
    token_estimate: int = 0


class ContextAssembler:
    """Build a compact, cited, ordered context window from ranked chunks."""

    def __init__(
        self,
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_chunk_tokens: int = _DEFAULT_MAX_CHUNK_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_chunk_tokens = max_chunk_tokens

    def assemble(
        self,
        chunks: list[dict],
        profile: QueryProfile | None = None,
    ) -> AssembledContext:
        """Return an ``AssembledContext`` from ``chunks``.

        Args:
            chunks: List of chunk dicts (e.g. from ``FeatureReranker.rank``).
            profile: Optional query profile used to pick ordering.

        Returns:
            Assembled context, list of included chunk IDs, and token estimate.
        """
        # 1. Deduplicate near-identical content.
        unique = self._deduplicate(chunks)

        # 2. Order: temporal or score.
        ordered = self._order(unique, profile)

        # 3. Allocate token budget per chunk with rank decay.
        budget = self._allocate(len(ordered))

        # 4. Build excerpt blocks.
        included: list[str] = []
        blocks: list[str] = []
        for idx, chunk in enumerate(ordered):
            chunk_id = str(chunk.get("chunk_id", idx))
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue

            excerpt = self._trim(content, budget[idx])
            source = str(chunk.get("source_id", "unknown"))
            header = f"[source:{source}]"
            blocks.append(f"{header}\n{excerpt}\n[citation:{chunk_id}]")
            included.append(chunk_id)

        text = "\n\n---\n\n".join(blocks)
        token_estimate = len(text) // _CHAR_PER_TOKEN
        return AssembledContext(
            text=text,
            included_chunk_ids=included,
            token_estimate=token_estimate,
        )

    def _deduplicate(self, chunks: list[dict]) -> list[dict]:
        """Remove near-duplicate chunks, keeping the earlier (higher-ranked) one."""
        kept: list[dict] = []
        seen_tokens: list[set[str]] = []
        for chunk in chunks:
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue
            tokens = self._token_set(content)
            if any(tokens == s for s in seen_tokens):
                continue
            if any(self._jaccard(tokens, s) > 0.9 for s in seen_tokens):
                continue
            kept.append(chunk)
            seen_tokens.append(tokens)
        return kept

    def _order(
        self,
        chunks: list[dict],
        profile: QueryProfile | None,
    ) -> list[dict]:
        """Order chunks: temporal ascending if the query is temporal."""
        if profile is not None and getattr(profile, "temporal", None):
            return sorted(
                chunks,
                key=lambda c: self._temporal_key(c),
            )
        return chunks

    def _allocate(self, n: int) -> list[int]:
        """Return per-chunk token budgets using rank-decayed allocation."""
        if n <= 0:
            return []
        # Budget_i = base * decay^i, but never exceed max_chunk_tokens.
        total_budget = self.max_tokens
        per_source_min = total_budget // max(n, 1)
        budgets: list[int] = []
        for i in range(n):
            decay = 0.8 ** i
            budget = int(min(self.max_chunk_tokens, per_source_min * decay))
            budgets.append(max(budget, 50))  # minimum useful excerpt
        return budgets

    def _trim(self, content: str, max_tokens: int) -> str:
        """Trim content to ``max_tokens`` by keeping the leading excerpt."""
        max_chars = max_tokens * _CHAR_PER_TOKEN
        if len(content) <= max_chars:
            return content
        # Try to break at a sentence boundary.
        truncated = content[:max_chars]
        for delim in (". ", "\n", " "):
            pos = truncated.rfind(delim)
            if pos > max_chars * 0.8:
                return truncated[:pos]
        return truncated

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _temporal_key(chunk: dict) -> int:
        meta = chunk.get("meta") or {}
        valid_from = meta.get("valid_from") or meta.get("created_at")
        if valid_from is None:
            return 0
        if isinstance(valid_from, datetime.datetime):
            return int(valid_from.timestamp())
        if isinstance(valid_from, datetime.date):
            return int(datetime.datetime.combine(valid_from, datetime.time.min).timestamp())
        try:
            return int(valid_from)
        except (TypeError, ValueError):
            return 0
