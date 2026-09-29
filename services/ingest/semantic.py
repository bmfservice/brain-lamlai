"""Braind Semantic Ingest (Brain Second Phase 4 wiring).

Orchestrates the Phase 3/5/6/7 pipeline for one ingested document, writing
into the semantic-layer tables:

    HierarchicalChunker → chunks
    EntityExtractor + EntityResolver → entities
    RelationValidator → typed knowledge_links (evidence-gated)
    ClaimExtractor + temporal → semantic_units (atomic facts per chunk)
    RelationExtractor   → relation candidates from co-occurring mentions

Each step is independent and retryable (§22) — a failure in one step never
rolls back the source document or the chunks already persisted.  Nothing
here re-embeds by cosine similarity, merges entities without evidence, or
creates an edge without evidence (§9/§13/§49/§63).
"""

from __future__ import annotations

import contextlib
import datetime
import uuid

from sqlalchemy import delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import load_only

from core import database as db
from core import taxonomy as tax
from core.chunking import HierarchicalChunker
from core.config import settings
from core.logging import get_logger
from services.consolidation.dataset import build_features, decision_label
from services.consolidation.features import REJECT_LABEL
from services.entity.blocking import block_candidates
from services.entity.extractor import EntityExtractor
from services.entity.normalize import (
    EntityIndex,
    clean_name,
    coerce_entity_type,
    describe,
    is_valid_name,
    normalize_name,
)
from services.entity.resolver import (
    AUTO_MERGE,
    DO_NOT_MERGE,
    REVIEW_REQUIRED,
    CandidateEntity,
    EntityResolver,
)
from services.entity.resolver import normalize_name as resolver_norm
from services.extract.claims import ClaimExtractor
from services.extract.relation_llm import LLMRelationVerifier
from services.extract.relations import RelationExtractor
from services.ingest.classify import classify_chunk_role
from services.ingest.freshness import FreshnessClassifier, LanguageClassifier
from services.relation.validator import RelationCandidate, RelationValidator

logger = get_logger(__name__)

# Entity descriptions are short; batching keeps the request under the model
# window while avoiding one round-trip per entity.
_ENTITY_EMBED_BATCH = 16


def _year_dt(year: int | None, *, end_of_year: bool = False):
    """Map a temporal extractor year onto the ``semantic_units`` timestamp.

    ``end_of_year=True`` renders Dec 31 23:59:59 UTC so a single-year window
    (CASE C: "in 2025" → valid 2025-2025) genuinely covers that year.
    """
    if year is None:
        return None
    if end_of_year:
        return datetime.datetime(
            year, 12, 31, 23, 59, 59, tzinfo=datetime.UTC
        )
    return datetime.datetime(year, 1, 1, tzinfo=datetime.UTC)


class SemanticIngestService:
    """Populate chunks / entities / typed edges for one ingested document."""

    def __init__(
        self,
        vllm=None,
        chunk_size: int = 800,
        embedding_version: str = "bge-m3-v1",
        *,
        language_classifier=None,
        freshness_classifier=None,
        search_validator=None,
        relation_classifier=None,
        metrics=None,
    ):
        self.vllm = vllm
        self.chunker = HierarchicalChunker(chunk_size=chunk_size)
        self.extractor = EntityExtractor(vllm=vllm)
        self.resolver = EntityResolver()
        self.validator = RelationValidator()
        self.claim_extractor = ClaimExtractor(vllm=vllm)
        self.relation_extractor = RelationExtractor()
        self.relation_verifier = LLMRelationVerifier(vllm)
        # Giai đoạn 5 (Layer 4): the distilled fast tier. Optional — when None or
        # cold, every candidate defers to the LLM (pre-Phase-5 behavior).
        self.relation_classifier = relation_classifier
        self.metrics = metrics
        self.embedding_version = embedding_version
        # Phase 2/3 collaborators — deterministic heuristics by default; the
        # caller may inject an LLM-backed classifier or configured override.
        self.language_classifier = language_classifier or LanguageClassifier(vllm=vllm)
        self.freshness_classifier = freshness_classifier or FreshnessClassifier(vllm=vllm)
        # Phase 4 collaborator (default None → search validation is a no-op).
        self.search_validator = search_validator
        # Aggregates reported back to the caller in ``semantic_pass``.
        self._last_freshness_counts: dict[str, int] = {}
        # Blocking + review counters from the last ``resolve_entities`` call.
        self._last_resolution_stats: dict[str, int] = {}
        # Chunk ids holding at least one TIME_SENSITIVE/REALTIME unit (Phase 4).
        self._fresh_sensitive_chunks: set[uuid.UUID] = set()
        # Chunk rows skipped in the most recent ``write_chunks`` by fingerprint.
        self._last_fingerprint_skips: int = 0
        self._last_llm_verified: int = 0
        self._last_llm_refined: int = 0

    # ------------------------------------------------------------------
    # Orchestration (called from the ingest pipeline)
    # ------------------------------------------------------------------

    async def semantic_pass(
        self,
        session,
        *,
        source_id=None,
        content: str,
        title: str = "",
        domain: str | None = None,
        relation_texts: list[str] | None = None,
        source_type: str = "manual",
        source_tags: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        """One self-contained semantic pass for an already-stored source.

        Creates an owning ``documents`` row (content = the whole text) and runs
        chunks → entities → typed edges → semantic units under the caller's
        session.  It is *independent and retryable* (§22): every step is
        best-effort, so a failure in entity extraction never destroys the
        persisted chunks and a vLLM outage leaves embeddings for maintenance to
        retry.
        """
        stats: dict = {"chunks": 0, "entities": 0, "relations": 0}
        try:
            if source_id is not None:
                # Replace the previous derived layer for this source inside this
                # pass's own transaction, so a failure here rolls back to the
                # old layer rather than leaving the source with none.
                await session.execute(
                    delete(db.Document).where(
                        db.Document.source_id == source_id,
                        db.Document.meta.has_key("semantic_pass"),  # noqa: W601 - JSONB
                    )
                )
            try:
                _sid = uuid.UUID(session_id) if session_id else None
            except (ValueError, AttributeError):
                _sid = None
            semantic_doc = db.Document(
                source_id=source_id,
                title=title,
                content=content,
                content_type="markdown",
                chunk_index=0,
                chunk_count=1,
                # Giai đoạn 4: the semantic container inherits the session so
                # it can never leak across visitors via retrieval.
                session_id=_sid,
                meta={"semantic_pass": True, "title": title[:120]},
            )
            session.add(semantic_doc)
            await session.flush()
            stats["document_id"] = str(semantic_doc.id)

            # compact: merge fragment blocks and fold undersized content into
            # its section. Storage wants readable units, not one row per table
            # cell. root_path gives web pages (which carry no markdown
            # headings) a locator they otherwise lack entirely.
            nodes = self.chunker.chunk(
                content,
                document_id=semantic_doc.id,
                title=title,
                compact=True,
                # Giai đoạn 3: prune=False everywhere. The retrieval path
                # already passed it; the semantic pass was the one remaining
                # lossy call — folding short paragraphs into their parent
                # section merged distinct facts and dropped heading-only
                # sections. Fragment merging happens once, upstream, with the
                # provenance preserved.
                prune=False,
                root_path=title,
            )
            ids = await self.write_chunks(
                session,
                document_id=semantic_doc.id,
                content=content,
                title=title,
                nodes=nodes,
            )
            stats["chunks"] = len(ids)
            stats["skipped"] = self._last_fingerprint_skips
        except Exception:  # pragma: no cover - defensive
            logger.exception("semantic.pass_chunks_failed")
            stats["error"] = "chunks"

        if stats.get("error"):
            return stats

        # The caller-supplied texts bound entity+relation extraction scope
        # (never a whole-dataset send, §123).
        texts = relation_texts or [content]
        try:
            entities = await self.resolve_entities(
                session,
                document_id=semantic_doc.id,
                texts=texts,
                domain=domain,
                extra_concepts=source_tags,
            )
            stats["entities"] = len(entities)
            stats.update(self._last_resolution_stats)
        except Exception:  # pragma: no cover - defensive
            logger.exception("semantic.pass_entities_failed")
            stats["error"] = "entities"
            return stats

        try:
            # Only the entities resolved for THIS document. Loading every
            # entity in the database would make the index match names the
            # document never mentions, and grows linearly with the corpus.
            name_to_id = {
                e.canonical_name: str(e.id) for e in entities if e.id is not None
            }
            for entity in entities:
                for alias in entity.aliases or []:
                    name_to_id.setdefault(str(alias), str(entity.id))
            chunk_texts = [
                (ids[i], nodes[i].content) for i in range(min(len(ids), len(nodes)))
            ]
            units = await self.write_semantic_units(
                session, chunk_texts=chunk_texts, name_to_id=name_to_id
            )
            stats["units"] = units
        except Exception:  # pragma: no cover - defensive
            logger.exception("semantic.pass_units_failed")
            stats["error"] = "units"

        # Phase 4: web search validation over TIME_SENSITIVE/REALTIME chunks
        # (bounded + SSRF-safe). Best-effort — a web outage never rolls back
        # the semantic write, and evidence rows never overwrite the source.
        if self.search_validator is not None and not stats.get("error"):
            try:
                stats["search_validated"] = await self._run_search_validation(
                    session,
                    source_type=source_type,
                    document_id=semantic_doc.id,
                    ids=ids,
                    nodes=nodes,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("semantic.pass_search_validation_failed")
                stats["search_validation_error"] = True
        return stats

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    async def write_chunks(
        self,
        session,
        *,
        document_id,
        content: str,
        title: str = "",
        nodes=None,
    ) -> list[uuid.UUID]:
        """Split ``content`` hierarchically and persist ``chunks`` rows.

        Returns the chunk ids in document order.  Chunk embedding is best-effort
        (a vLLM outage leaves the rows for later re-embedding by maintenance).
        ``nodes`` may be passed pre-computed (from ``semantic_pass``) so callers
        can pair chunk ids with their texts without re-chunking.
        """
        if nodes is None:
            nodes = self.chunker.chunk(
                content,
                document_id=document_id,
                title=title,
                compact=True,
                root_path=title,
            )
        ids: list[uuid.UUID] = []

        # Phase 8: idempotent reprocess — skip a chunk whose content_fingerprint
        # already exists for this document (mirrors the units (span_start,text)
        # skip). Gathers the DB's existing fingerprints once before the loop.
        existing_fps: set[str] = set()
        try:
            fps = await session.execute(
                select(db.Chunk.content_fingerprint).where(
                    db.Chunk.document_id == document_id,
                    db.Chunk.content_fingerprint.is_not(None),
                )
            )
            existing_fps = {f for f in fps.scalars().all() if f}
        except Exception:  # pragma: no cover - defensive
            logger.warning("semantic.fingerprint_scan_failed", exc_info=True)

        self._last_fingerprint_skips = 0
        try:
            detected_language, language_confidence = (
                await self.language_classifier.classify(content)
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("semantic.language_detect_failed", exc_info=True)

        # Build every row first, then insert all chunks in ONE multi-row
        # statement. The chunks table has self-referential FKs (parent_id /
        # prev_chunk_id / next_chunk_id -> chunks.id); PostgreSQL resolves
        # those within a single statement, but one-row-at-a-time inserts in
        # document order violate next_chunk_id (the following row is not yet
        # inserted). The mock-based tests never enforce FKs, so this only
        # fails against the live DB.
        rows: list[dict] = []
        for node in nodes:
            row = node.to_row()
            # Phase 8: skip an already-ingested chunk (identical fingerprint).
            if row.get("content_fingerprint") and row["content_fingerprint"] in existing_fps:
                self._last_fingerprint_skips += 1
                continue
            embedding = None
            if self.vllm is not None and node.chunk_kind in {"section", "paragraph"}:
                try:
                    embedding = await self.vllm.embed(node.content)
                except Exception:  # pragma: no cover - defensive
                    logger.warning("semantic.chunk_embed_failed", exc_info=True)

            role = classify_chunk_role(node.content, chunk_kind=node.chunk_kind)
            node_meta = {
                "chunk_kind": node.chunk_kind,
                "chunk_role": role,
                "detected_language": detected_language,
                "language_confidence": language_confidence,
            }
            if node.section_path:
                node_meta["section_path"] = node.section_path
            rows.append(
                {
                    "id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "parent_id": row["parent_id"],
                    "prev_chunk_id": row["prev_chunk_id"],
                    "next_chunk_id": row["next_chunk_id"],
                    "section_path": row["section_path"],
                    "hierarchy_level": row["hierarchy_level"],
                    "content": row["content"],
                    "chunk_role": role,
                    "start_offset": row["start_offset"],
                    "end_offset": row["end_offset"],
                    "context_summary": row["context_summary"],
                    "root_id": row.get("root_id"),
                    "order": row.get("order"),
                    "page_number": row.get("page_number"),
                    "section_number": row.get("section_number"),
                    "source_location": row.get("source_location"),
                    "content_fingerprint": row.get("content_fingerprint"),
                    "embedding_vllm_1024": embedding,
                    "embedding_version": self.embedding_version,
                    "meta": node_meta,
                }
            )

        if rows:
            stmt = (
                pg_insert(db.Chunk)
                .values(rows)
                .returning(db.Chunk.id)
            )
            result = await session.execute(stmt)
            # RETURNING matches the values order, so ids stay in document order.
            ids.extend(result.scalars().all())

        await session.flush()
        return ids

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    async def resolve_entities(
        self,
        session,
        *,
        document_id=None,
        texts: list[str],
        domain: str | None = None,
        extra_concepts: list[str] | None = None,
    ) -> list[db.Entity]:
        """Extract mentions, resolve against existing entities, persist.

        Merge-policy decisions are honoured: AUTO_MERGE attaches the mention to
        the existing canonical id (adds the alias if new), reviews and new
        entities are inserted with a review/ambiguous state — never silently
        collapsed (CASE E/F).  Subsumption hints (CASE E/F, "related to but
        distinct from") from the resolver AND co-occurrence relation candidates
        (P4-09) both become evidence-gated typed edges via ``write_relations``
        — structural evidence, never cosine-only (§9/§63).
        """
        known = await self._load_entities(session)
        known_by_id = {e.id: e for e in known}
        known_by_name = {e.canonical_name: e for e in known}
        known_dtos = [
            CandidateEntity(
                entity_id=str(e.id),
                canonical_name=e.canonical_name,
                aliases=list(e.aliases or []),
                entity_type=e.entity_type,
                domain=(e.meta or {}).get("domain"),
            )
            for e in known
        ]

        # Extraction runs for the whole document *before* resolution, because
        # blocking needs every mention to build the comparison set once. Doing it
        # per mention would rebuild the resolver's lookup tables N times over the
        # whole corpus — 519ms vs 32ms for one 60-mention document at 2470
        # entities, and the gap widens with every ingest.
        # (text, [(name, start, end), ...]) for P4-09 co-occurrence candidates.
        text_mentions: list[tuple[str, list[tuple[str, int, int]]]] = []
        extracted: list[tuple[str, object]] = []  # (text, EntityMention)
        for text in texts:
            mentions = await self.extractor.extract(text)
            text_mentions.append(
                (text, [(m.text, m.start, m.end) for m in mentions])
            )
            extracted.extend((text, m) for m in mentions)

        # Layer 1 of §5 — blocking. Only candidates sharing a token or acronym
        # key with something this document mentions are comparable at all. The
        # domain is deliberately *not* a blocker: an exact name from another
        # domain must still reach the resolver, which then records a review
        # instead of silently creating a duplicate.
        blocked = block_candidates([m.text for _, m in extracted], known_dtos)
        # Entities this document demonstrably talks about: a mention matched one
        # of their names exactly. Computed up front so structural support does
        # not depend on mention order — and so a document that anchors on nothing
        # known skips the graph query entirely (support would be 0 regardless).
        context_ids = self._document_context_ids(
            [m.text for _, m in extracted], blocked.candidates
        )
        graph_neighbors = (
            await self._entity_neighbors(
                session, [c.entity_id for c in blocked.candidates]
            )
            if context_ids
            else {}
        )
        logger.info(
            "semantic.entity_blocking",
            mentions=len(extracted),
            considered=blocked.considered,
            candidates=blocked.kept,
            keys=len(blocked.keys),
            truncated=blocked.truncated,
            context=len(context_ids),
            graph_nodes=len(graph_neighbors),
        )

        resolved: dict[str, db.Entity] = {}
        alias_additions: list[tuple[uuid.UUID, str]] = []
        # (source_mention, relation_type, target_entity) for CASE E/F edges.
        hint_edges: list[tuple[str, str, db.Entity]] = []
        review_count = 0
        for text, mention in extracted:
            res = self.resolver.resolve(
                mention.text,
                entity_type=mention.entity_type,
                known=blocked.candidates,
                domain=domain,
                graph_neighbors=graph_neighbors,
                context_ids=context_ids,
            )
            if res.decision == REVIEW_REQUIRED:
                # Never silently dropped and never merged: the mention
                # becomes its own row flagged for a human, carrying which
                # entity it collided with and why (§5 NEEDS_REVIEW).
                review_count += 1
                name = clean_name(mention.text)
                key = normalize_name(name)
                # `canonical_name` is globally unique. A same-name entity in
                # another domain therefore cannot be represented as a second
                # entity row — inserting it rolls the entire semantic pass back.
                # The pre-existing candidate remains the review anchor; record a
                # review edge with both domain assertions instead. It is visible,
                # auditable and non-destructive, while the operator decides
                # whether the word is one polysemous entity or two concepts that
                # need an eventual schema-level identity namespace.
                if is_valid_name(name) and key not in resolved:
                    review_target = next(
                        (
                            c
                            for c in blocked.candidates
                            if c.entity_id and c.canonical_name
                            == res.evidence.canonical_of_candidate
                        ),
                        None,
                    )
                    if review_target is None:
                        logger.warning(
                            "semantic.review_target_missing",
                            mention=name[:80],
                            target=res.evidence.canonical_of_candidate,
                        )
                        continue
                    target_id = uuid.UUID(review_target.entity_id)
                    review_evidence = [
                        {
                            "kind": "entity_resolution_review",
                            "mention": name,
                            "domain": domain,
                            "candidate_domain": review_target.domain,
                            "matched_by": res.evidence.matched_by,
                            "structural_support": res.evidence.structural_support,
                            "reason": res.reason,
                        }
                    ]
                    await session.execute(
                        pg_insert(db.KnowledgeLink)
                        .values(
                            source_id=target_id,
                            target_id=target_id,
                            source_kind="entity",
                            target_kind="entity",
                            relation_type="POSSIBLE_RELATION",
                            confidence=res.confidence,
                            evidence=review_evidence,
                            evidence_count=1,
                            reason=res.reason,
                            model_version="entity-resolver-v1",
                            lifecycle="review",
                            meta={
                                "entity_resolution_review": True,
                                "mention": name,
                                "domain": domain,
                                "review_of": review_target.canonical_name,
                                "candidate_domain": review_target.domain,
                                "matched_by": res.evidence.matched_by,
                                "structural_support": res.evidence.structural_support,
                            },
                        )
                        .on_conflict_do_nothing(
                            index_elements=[
                                db.KnowledgeLink.source_id,
                                db.KnowledgeLink.target_id,
                                db.KnowledgeLink.relation_type,
                            ]
                        )
                    )
                continue
            if res.decision == AUTO_MERGE:
                alias_additions.append((uuid.UUID(res.target_id), mention.text))
                continue
            if res.decision == DO_NOT_MERGE and res.target_id is None:
                name = clean_name(mention.text)
                if not is_valid_name(name):
                    # Extraction noise: a stopword, a two-letter fragment,
                    # or a whole sentence caught by a bad span.
                    continue
                key = normalize_name(name)
                if key in resolved:
                    continue
                entity_type = coerce_entity_type(res.entity_type, name=name)
                entity = db.Entity(
                    canonical_name=name,
                    normalized_name=key,
                    aliases=[name],
                    entity_type=entity_type,
                    description=describe(name, entity_type, text[: 240]),
                    resolution_state="ambiguous",
                    merge_policy="review",
                    confidence=res.confidence,
                    meta={"domain": domain, "reason": res.reason},
                )
                resolved[key] = entity
                # The subsumption guard named a distinct-but-related
                # pre-existing entity → record the edge evidence (CASE E/F).
                if res.relation_hint is not None:
                    relation_type, target_name = res.relation_hint
                    target = known_by_name.get(target_name)
                    if target is not None:
                        hint_edges.append((key, relation_type, target))

        # Persist new entities (dedupe repeats within this document).
        for alias_target, alias in alias_additions:
            if alias_target in resolved:
                continue
            target = known_by_id.get(alias_target)
            if target is not None and alias not in (target.aliases or []):
                # Add the alias to the *existing* entity. An UPDATE (not an
                # INSERT ... ON CONFLICT) is used because the target may have
                # been created in a rolled-back savepoint and no longer exist
                # in the DB — an INSERT would then violate canonical_name NOT
                # NULL. An UPDATE affects 0 rows in that case, which is the
                # correct no-op (we only ever add an alias to a real entity).
                await session.execute(
                    update(db.Entity)
                    .where(db.Entity.id == alias_target)
                    .values(aliases=(target.aliases or []) + [alias])
                )

        # Phase 6: keywords/tags are also first-class ``concept`` entities (no
        # parallel taxonomy). They dedupe on canonical_name against everything
        # already resolved/known here, so re-ingesting never duplicates.
        known_names = {normalize_name(e.canonical_name) for e in known}
        for concept in extra_concepts or []:
            name = clean_name(concept or "")
            key = normalize_name(name)
            if not is_valid_name(name) or key in known_names or key in resolved:
                continue
            resolved[key] = db.Entity(
                canonical_name=name,
                normalized_name=key,
                aliases=[name],
                entity_type=tax.ENTITY_CONCEPT,
                description=describe(name, tax.ENTITY_CONCEPT),
                resolution_state="resolved",
                merge_policy="auto",
                confidence=0.7,
                meta={"domain": domain, "from_tag": True, "reason": "source_keyword"},
            )

        for entity in resolved.values():
            session.add(entity)
        await session.flush()

        # Embed the descriptions. Every entity previously carried a NULL
        # vector, so the entity HNSW index was empty and entity search
        # returned nothing at all. Best-effort: a vLLM outage leaves the rows
        # for the maintenance re-index rather than failing the pass.
        if self.vllm is not None and resolved:
            pending = [e for e in resolved.values() if e.description]
            for start in range(0, len(pending), _ENTITY_EMBED_BATCH):
                batch = pending[start : start + _ENTITY_EMBED_BATCH]
                try:
                    vectors = await self.vllm.embed_batch([e.description for e in batch])
                except Exception:  # pragma: no cover - defensive
                    logger.warning("semantic.entity_embed_failed", exc_info=True)
                    break
                for entity, vector in zip(batch, vectors, strict=False):
                    entity.embedding_vllm_1024 = vector
            await session.flush()

        # Resolve hint edges now that new entities carry ids, and add P4-09
        # co-occurrence candidates.  Every edge exists either because the
        # subsumption guard decided the two names are *distinct but related*, or
        # because the two resolved entities co-occur with lexical structure —
        # never cosine similarity alone — and everything flows through the
        # validator so typing + confidence rules still apply (§10/§49).
        all_candidates: list[RelationCandidate] = []
        if hint_edges:
            for mention, relation_type, target in hint_edges:
                source = resolved.get(mention)
                if source is None or source.id is None:
                    continue
                all_candidates.append(
                    RelationCandidate(
                        source_id=str(source.id),
                        target_id=str(target.id),
                        proposed_type=relation_type.upper(),
                        signals={"entity": 1.0, "structural": 0.9, "lexical": 0.6},
                        evidence_sources=[
                            f"resolution:{mention}",
                            f"subsumption-guard:{target.canonical_name}",
                        ],
                    )
                )
        if text_mentions:
            name_to_id = {
                e.canonical_name: str(e.id)
                for e in resolved.values()
                if e.id is not None
            }
            name_to_id.update(
                {e.canonical_name: str(e.id) for e in known if e.id is not None}
            )
            for text, spans in text_mentions:
                drafts = self.relation_extractor.candidates(
                    text=text,
                    mentions=spans,
                    source_ref=str(document_id) if document_id else "",
                )
                all_candidates.extend(
                    await self._verify_candidates(
                        drafts, name_to_id=name_to_id, session=session
                    )
                )
        persisted = await self.write_relations(session, candidates=all_candidates)
        if persisted:
            logger.info(
                "semantic.relations",
                count=len(persisted),
                llm_verified=self._last_llm_verified,
                llm_refined=self._last_llm_refined,
            )
        # §5 requires the report to say how much a human still has to look at.
        # A resolution run that produced 40 review rows and one that produced 0
        # are very different outcomes and used to be indistinguishable.
        self._last_resolution_stats = {
            "mentions": len(extracted),
            "candidates_considered": blocked.considered,
            "candidates_blocked": blocked.kept,
            "needs_review": review_count,
        }
        if review_count:
            logger.info(
                "semantic.entities_need_review",
                needs_review=review_count,
                mentions=len(extracted),
            )
        return list(resolved.values())

    async def _load_entities(self, session) -> list[db.Entity]:
        """Load the entities blocking + resolution compare against.

        Only the columns resolution reads are fetched: pulling the 1024-dim
        ``embedding_vllm_1024`` for every row cost 400ms at 2470 entities where
        the projection below costs 28ms, and blocking discards ~98% of the rows
        anyway. The identity work never touches the vector, so loading it is pure
        waste (and a reminder that resolution is orthographic, not cosine, §13).
        """
        rows = (
            await session.execute(
                select(db.Entity).options(
                    load_only(
                        db.Entity.id,
                        db.Entity.canonical_name,
                        db.Entity.aliases,
                        db.Entity.entity_type,
                        db.Entity.meta,
                    )
                )
            )
        ).scalars().all()
        return list(rows)

    @staticmethod
    def _document_context_ids(
        mention_texts: list[str], candidates: list[CandidateEntity]
    ) -> set[str]:
        """Ids of blocked candidates a mention names *exactly* (canonical/alias).

        These anchor the structural signal: an entity this document unambiguously
        talks about is the fixed point a shared-neighbour count is measured
        against. An approximate or subsumed mention is deliberately excluded — it
        is the very thing we are still trying to resolve.
        """
        wanted = {resolver_norm(t) for t in mention_texts}
        wanted.discard("")
        ids: set[str] = set()
        for c in candidates:
            names = [c.canonical_name, *c.aliases]
            if any(resolver_norm(n) in wanted for n in names):
                ids.add(c.entity_id)
        return ids

    async def _entity_neighbors(
        self, session, entity_ids: list[str]
    ) -> dict[str, set[str]]:
        """Map each candidate id to its knowledge-graph neighbour ids.

        Reads only entity→entity edges that ``RelationValidator`` already
        accepted (evidence-gated, never cosine-only), so the structural signal
        inherits that discipline. One indexed query over the blocked set, not the
        whole graph.
        """
        if not entity_ids:
            return {}
        uuids: list[uuid.UUID] = []
        for raw in entity_ids:
            try:
                uuids.append(uuid.UUID(str(raw)))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        if not uuids:
            return {}
        rows = (
            await session.execute(
                select(
                    db.KnowledgeLink.source_id, db.KnowledgeLink.target_id
                ).where(
                    db.KnowledgeLink.source_kind == "entity",
                    db.KnowledgeLink.target_kind == "entity",
                    db.KnowledgeLink.lifecycle == "active",
                    or_(
                        db.KnowledgeLink.source_id.in_(uuids),
                        db.KnowledgeLink.target_id.in_(uuids),
                    ),
                )
            )
        ).all()
        neighbours: dict[str, set[str]] = {}
        for source_id, target_id in rows:
            s, t = str(source_id), str(target_id)
            neighbours.setdefault(s, set()).add(t)
            neighbours.setdefault(t, set()).add(s)
        return neighbours

    # ------------------------------------------------------------------
    # Semantic units (P4-08 claim extraction + P4-10 temporal windows)
    # ------------------------------------------------------------------

    async def write_semantic_units(
        self,
        session,
        *,
        chunk_texts: list[tuple[uuid.UUID, str]],
        name_to_id: dict[str, str] | None = None,
    ) -> int:
        """Extract and persist atomic ``semantic_units`` rows per chunk.

        One deterministic claim+temporal pass per chunk (LLM-assisted only when
        a model is available AND the chunk is long enough, §master "no LLM for
        every cleaning").  Retry-safe (§22): units already present for a chunk
        with the same (span_start, content) are skipped, never duplicated, and
        year windows map onto the table's timestamp columns.
        """
        name_to_id = name_to_id or {}
        entity_index = EntityIndex(name_to_id)
        written = 0
        freshness_counts: dict[str, int] = {}
        sensitive_chunks: set[uuid.UUID] = set()
        for chunk_id, content in chunk_texts:
            if not content or not content.strip():
                continue
            drafts = await self.claim_extractor.extract(
                content, chunk_id=str(chunk_id)
            )
            existing = await self._existing_unit_keys(
                session, chunk_id=chunk_id
            )
            for draft in drafts:
                key = (draft.span_start, draft.content)
                if key in existing:
                    continue
                # Phase 3: freshness classification (heuristic by default; the
                # classifier reuses the draft's temporal window).
                freshness_cls, freshness_conf, _reasons = (
                    await self.freshness_classifier.classify(
                        draft.content,
                        has_temporal=self._draft_window(draft),
                    )
                )
                freshness_counts[freshness_cls] = (
                    freshness_counts.get(freshness_cls, 0) + 1
                )
                if freshness_cls in ("TIME_SENSITIVE", "REALTIME"):
                    sensitive_chunks.add(chunk_id)
                # The claim extractor leaves entity_mentions empty, so match
                # the claim text against the entities resolved for this
                # document. Without this every semantic unit stored an empty
                # entity_ids and the graph had no way back from evidence.
                entity_ids = [
                    name_to_id[m] for m in draft.entity_mentions if m in name_to_id
                ] or entity_index.find(draft.content)
                embedding = None
                if self.vllm is not None:
                    try:
                        embedding = await self.vllm.embed(draft.content)
                    except Exception:  # pragma: no cover - defensive
                        logger.warning("semantic.unit_embed_failed", exc_info=True)
                session.add(
                    db.SemanticUnit(
                        chunk_id=chunk_id,
                        unit_type=draft.unit_type,
                        content=draft.content,
                        polarity=draft.polarity,
                        valid_from=_year_dt(draft.valid_from),
                        valid_to=_year_dt(draft.valid_to, end_of_year=True),
                        entity_ids=entity_ids,
                        confidence=draft.confidence,
                        span_start=draft.span_start,
                        span_end=draft.span_end,
                        fact_class=draft.fact_class,
                        embedding_vllm_1024=embedding,
                        meta={
                            "extractor": "braind-claims-v1",
                            "freshness": freshness_cls,
                            "freshness_confidence": freshness_conf,
                        },
                    )
                )
                written += 1
        self._last_freshness_counts = freshness_counts
        self._fresh_sensitive_chunks = sensitive_chunks
        await session.flush()
        return written

    @staticmethod
    def _draft_window(draft) -> tuple | bool:
        """Pass the draft's temporal window to the freshness classifier.

        A tuple ``(valid_from_year_or_None, valid_to_year_or_None)`` lets the
        heuristic treat an open window as REALTIME and a dated window as
        TIME_SENSITIVE without requiring a live clock.
        """
        vf = draft.valid_from
        vt = draft.valid_to
        if vf is None and vt is None:
            return False
        return (vf, vt)

    async def _run_search_validation(
        self,
        session,
        *,
        source_type: str,
        document_id,
        ids: list[uuid.UUID],
        nodes,
    ) -> int:
        """Validate fresh-sensitive chunks against a bounded web lookup.

        Only the first ``search_validation_max_outbound`` qualifying chunks are
        checked per ingest; results are persisted as ``search_evidence`` rows
        (additive). A web/network error in one chunk never aborts later ones.
        """
        if not self._fresh_sensitive_chunks:
            return 0
        # The validator's ``should_search`` gate already bounds by source type;
        # the loop hard-caps outbound calls regardless.
        if self.search_validator is None:
            return 0
        id_to_content = {
            str(ids[i]): nodes[i].content
            for i in range(min(len(ids), len(nodes)))
        }
        outbound = 0
        count = 0

        for chunk_id in self._fresh_sensitive_chunks:
            if outbound >= settings.search_validation_max_outbound:
                break
            content = id_to_content.get(str(chunk_id))
            if not content or not content.strip():
                continue
            if not self.search_validator.should_search("TIME_SENSITIVE", source_type):
                break
            query = self.search_validator.query_for_chunk(content)
            record = await self.search_validator.validate_node(
                node_content=content,
                search_query=query,
                document_id=document_id,
                chunk_id=chunk_id,
            )
            await self.search_validator.persist(session, record)
            outbound += 1
            count += 1
        return count

    async def _existing_unit_keys(
        self, session, *, chunk_id: uuid.UUID
    ) -> set[tuple[int, str]]:
        """(span_start, content) keys already persisted for a chunk."""
        rows = (
            await session.execute(
                select(db.SemanticUnit.span_start, db.SemanticUnit.content).where(
                    db.SemanticUnit.chunk_id == chunk_id
                )
            )
        ).all()
        return {(int(span or 0), str(content)) for span, content in rows}

    # ------------------------------------------------------------------
    # Typed edges
    # ------------------------------------------------------------------

    async def _verify_candidates(
        self,
        drafts,
        *,
        name_to_id: dict[str, str],
        session=None,
    ) -> list[RelationCandidate]:
        """Bind drafts to ids, then rule on each pair through the two-tier brain.

        Giai đoạn 5 — Complementary Learning Systems. Each in-budget candidate
        takes one of three branches:

        1. **Consolidated (Layer 4):** the distilled classifier scores the
           pair's feature vector. When its confidence clears the threshold it
           rules immediately (microseconds), and ``decided_by`` is its version.
        2. **LLM (Layer 5):** everything the classifier is cold on or unsure of
           goes to the chat model — one *batched* call (B3). The model quotes
           the evidence and justifies its ruling.
        3. **Pass-through:** structurally weak / sentence-less candidates keep
           their lexical verdict unchanged (the pre-Phase-5 path).

        Every branch-1/2 decision — accept *and* reject — is written to the
        ``relation_decisions`` ledger so tonight's consolidation has fresh
        training data. Each failure degrades to the lexical candidate; an
        outage of either tier must never fail an ingest.
        """
        candidates = self.relation_extractor.to_candidates(
            drafts, name_to_id=name_to_id
        )
        self._last_llm_verified = 0
        self._last_llm_refined = 0
        if not candidates or not settings.relation_llm_enabled:
            return candidates

        # Bound the reasoning workload: at most ``relation_llm_max_pairs``
        # candidates get judged, prioritizing the strongest structural signal so
        # one huge document cannot turn into an unbounded batch.
        budget = max(0, int(settings.relation_llm_max_pairs))
        floor = settings.relation_llm_min_structural
        pairs: list[tuple[RelationCandidate, object, str]] = []
        passed_through: set[int] = set()
        for index, candidate in enumerate(candidates):
            draft = drafts[index] if index < len(drafts) else None
            sentence = getattr(draft, "sentence", "")
            strong = candidate.signals.get("structural", 0.0) >= floor
            if sentence and budget > 0 and strong and self.relation_verifier is not None:
                pairs.append((candidate, draft, sentence))
                budget -= 1
            else:
                passed_through.add(index)

        name_by_id = {v: k for k, v in name_to_id.items()}
        refined: list[RelationCandidate] = []
        if not pairs:
            return [c for i, c in enumerate(candidates) if i in passed_through]

        # Featurize every in-budget pair once; the same vector feeds both the
        # classifier and the ledger row (train/infer parity — features.py).
        pair_features = [
            build_features(
                proposed_type=candidate.proposed_type,
                signals=candidate.signals,
                sentence=sentence,
            )
            for candidate, _draft, sentence in pairs
        ]

        # --- Branch 1: consolidated classifier decides the confident ones ---
        llm_slots: list[int] = []  # indices into ``pairs`` still needing the LLM
        overridden = 0
        consolidated_decided = 0
        if self.relation_classifier is not None:
            predictions = self.relation_classifier.predict_batch(pair_features)
        else:
            predictions = [None] * len(pairs)

        for index, prediction in enumerate(predictions):
            candidate, _draft, sentence = pairs[index]

            # --- Branch 0: human override on this exact pair wins outright ---
            # Ground truth, effective immediately (no retrain, no restart). A
            # REJECT override drops the candidate just like a model reject.
            override = (
                self.relation_classifier.lookup_override(
                    candidate.source_id, candidate.target_id
                )
                if self.relation_classifier is not None
                else None
            )
            if override is not None:
                o_label, o_conf = override
                overridden += 1
                self._count_metric("relation.decided_by.override")
                if o_label == REJECT_LABEL:
                    continue
                self._last_llm_verified += 1
                refined.append(
                    RelationCandidate(
                        source_id=candidate.source_id,
                        target_id=candidate.target_id,
                        proposed_type=o_label,
                        signals={**candidate.signals, "semantic": max(0.0, o_conf)},
                        evidence_sources=list(candidate.evidence_sources),
                        model_version="human-override",
                        evidence_text=sentence,
                        rationale="human override (ground truth)",
                    )
                )
                continue

            if prediction is None or not prediction.decided:
                llm_slots.append(index)
                continue
            consolidated_decided += 1
            self._count_metric("relation.decided_by.consolidated")
            decided_by = self.relation_classifier.version or "consolidated"
            await self._record_decision(
                session,
                candidate=candidate,
                features=pair_features[index],
                label=prediction.relation_type,
                decided_by=decided_by,
                confidence=prediction.confidence,
                rationale="consolidated classifier ruling",
            )
            if prediction.relation_type == REJECT_LABEL:
                continue  # fast tier learned the sentence asserts no relation
            self._last_llm_verified += 1
            refined.append(
                RelationCandidate(
                    source_id=candidate.source_id,
                    target_id=candidate.target_id,
                    proposed_type=prediction.relation_type,
                    signals={
                        **candidate.signals,
                        "semantic": max(0.0, prediction.confidence),
                    },
                    evidence_sources=list(candidate.evidence_sources),
                    model_version=decided_by,
                    evidence_text=sentence,
                    rationale="consolidated classifier ruling",
                )
            )

        # --- Branch 2: one batched LLM call for the remainder ---
        if llm_slots:
            judgments = await self.relation_verifier.judge_batch(
                [
                    (
                        pairs[i][2],
                        name_by_id.get(pairs[i][0].source_id, pairs[i][0].source_id),
                        name_by_id.get(pairs[i][0].target_id, pairs[i][0].target_id),
                        pairs[i][0].proposed_type or "POSSIBLE_RELATION",
                    )
                    for i in llm_slots
                ]
            )
            for slot, judgment in zip(llm_slots, judgments, strict=True):
                candidate, _draft, sentence = pairs[slot]
                if judgment is None:
                    # LLM outage → keep the lexical verdict, record nothing
                    # (no reliable label to learn from).
                    refined.append(candidate)
                    continue
                label = decision_label(
                    supported=judgment.supported, relation_type=judgment.relation_type
                )
                self._count_metric("relation.decided_by.llm")
                await self._record_decision(
                    session,
                    candidate=candidate,
                    features=pair_features[slot],
                    label=label,
                    decided_by=self.relation_verifier.model_version,
                    confidence=max(0.0, judgment.confidence),
                    rationale=judgment.rationale,
                )
                if not judgment.supported:
                    self._count_metric("relation.decided_by.rejected")
                    logger.info(
                        "relation_llm.rejected",
                        source=name_by_id.get(candidate.source_id, "?")[:40],
                        target=name_by_id.get(candidate.target_id, "?")[:40],
                    )
                    continue
                self._last_llm_verified += 1
                if judgment.relation_type != candidate.proposed_type:
                    self._last_llm_refined += 1
                refined.append(
                    RelationCandidate(
                        source_id=candidate.source_id,
                        target_id=candidate.target_id,
                        proposed_type=judgment.relation_type,
                        signals={
                            **candidate.signals,
                            "semantic": max(0.0, judgment.confidence),
                        },
                        evidence_sources=list(candidate.evidence_sources),
                        model_version=self.relation_verifier.model_version,
                        evidence_text=judgment.quote or sentence,
                        rationale=judgment.rationale,
                    )
                )
        logger.info(
            "relation_llm.judged",
            pairs=len(pairs),
            override=overridden,
            consolidated=consolidated_decided,
            llm=len(llm_slots),
            kept=len(refined),
            batched=True,
        )
        return [c for i, c in enumerate(candidates) if i in passed_through] + refined

    def _count_metric(self, name: str) -> None:
        """Bump a counter on the metrics registry when one is wired (best-effort)."""
        if self.metrics is None:
            return
        with contextlib.suppress(Exception):  # metrics never break an ingest
            self.metrics.counter(name)

    async def _record_decision(
        self,
        session,
        *,
        candidate: RelationCandidate,
        features: dict[str, float],
        label: str,
        decided_by: str,
        confidence: float,
        rationale: str,
    ) -> None:
        """Append one ruling to the ``relation_decisions`` training ledger.

        Best-effort: a missing session or a bad id must not fail the ingest, and
        rows land in the caller's transaction so they roll back together with the
        edges they justified.
        """
        if session is None:
            return
        try:
            source_uuid = uuid.UUID(str(candidate.source_id))
            target_uuid = uuid.UUID(str(candidate.target_id))
        except (ValueError, AttributeError, TypeError):
            return
        try:
            session.add(
                db.RelationDecision(
                    source_id=source_uuid,
                    target_id=target_uuid,
                    features=features,
                    label=label,
                    decided_by=decided_by[:60],
                    confidence=float(confidence),
                    rationale=rationale or None,
                    human_override=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("relation_decision.record_failed", error=str(exc)[:120])

    async def write_relations(
        self,
        session,
        *,
        candidates: list[RelationCandidate],
    ) -> list[dict]:
        """Validate + persist evidence-gated typed edges into knowledge_links.

        Giai đoạn 3: each candidate's quoted ``evidence_text`` goes into the
        edge's ``evidence`` JSONB (alongside the provenance labels) so every
        stored edge carries the sentence that justifies it — and the optional
        LLM verifier can add a semantic ruling (family typing + rationale)
        before the validator's lexical verdict. Cosine-only edges and
        evidence-less candidates never make it here (§9/§63).
        """
        persisted: list[dict] = []
        for candidate in candidates:
            verdict = self.validator.validate(candidate)
            if not verdict.accepted:
                continue
            evidence: list[str] = list(verdict.evidence_sources)
            if verdict.evidence_text:
                evidence.append(f"quote: {verdict.evidence_text}")
            await session.execute(
                pg_insert(db.KnowledgeLink)
                .values(
                    source_id=verdict.source_id,
                    target_id=verdict.target_id,
                    relation_type=verdict.relation_type,
                    confidence=verdict.confidence,
                    evidence=evidence,
                    evidence_count=len(evidence),
                    model_version=verdict.model_version,
                    reason=verdict.reason,
                    source_kind="entity",
                    target_kind="entity",
                    weight=verdict.confidence,
                )
                .on_conflict_do_nothing(
                    index_elements=["source_id", "target_id", "relation_type"]
                )
            )
            persisted.append(
                {
                    "source_id": verdict.source_id,
                    "target_id": verdict.target_id,
                    "relation_type": verdict.relation_type,
                    "confidence": verdict.confidence,
                }
            )
        await session.flush()
        return persisted