"""Braind Core - Database models for PostgreSQL + pgvector.

All data is stored in PostgreSQL with:
- pgvector for vector similarity search
- pg_trgm for fuzzy text matching
- JSONB for flexible metadata
"""

from __future__ import annotations

import datetime
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from core.errors import DatabaseInitializationError
from core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


# --- Documents / Knowledge Base ---


class Source(Base):
    """A source of knowledge (file, RSS, web, manual)."""

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="manual"
    )  # file, rss, web, obsidian, manual
    path: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    # Deduplication: sha256 of first 4k chars of the first chunk.
    content_hash: Mapped[str | None] = mapped_column(Text, unique=True)
    # Normalized tags (deduped keywords/topic labels) for link building.
    tags: Mapped[list | None] = mapped_column(JSONB, default=list)
    # --- Classification (deep-ingest upgrade) -----------------------------
    # Closed vocabularies from core/taxonomy.py. Columns, not JSONB keys, so
    # retrieval can filter and rank on them without scanning every row's meta.
    doc_type: Mapped[str | None] = mapped_column(String(32))
    domain: Mapped[str | None] = mapped_column(String(64))
    audience: Mapped[str | None] = mapped_column(String(32))
    quality_tier: Mapped[str | None] = mapped_column(String(16))
    quality_score: Mapped[float | None] = mapped_column(Float)
    language: Mapped[str | None] = mapped_column(String(8))
    # True parent-child hierarchy: a source can be a child of another source.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="SET NULL"),
        nullable=True,
    )
    # Giai đoạn 4 — anonymous session scoping. NULL = system/global (background
    # jobs, RSS, vault) and visible to every session; a value scopes the row to
    # one visitor session. fingerprint_hash ties the session to a device.
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    fingerprint_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")
    parent = relationship("Source", remote_side="Source.id", back_populates="children")
    children = relationship("Source", back_populates="parent", foreign_keys="Source.parent_id")

    __table_args__ = (
        Index("idx_sources_content_hash", "content_hash", unique=True),
        Index("idx_sources_parent_id", "parent_id"),
        Index("idx_sources_type", "source_type"),
        Index("idx_sources_doc_type", "doc_type"),
        Index("idx_sources_domain", "domain"),
        Index("idx_sources_quality_tier", "quality_tier"),
        Index("idx_sources_session_id", "session_id"),
    )


class KnowledgeLink(Base):
    """Normalized, bidirectional knowledge graph edge between two nodes.

    Every relationship is a first-class row (not JSONB) so the graph is
    queryable, indexed, weighted and never silently dropped. Since the Brain
    Second upgrade the edge is **typed, weighted, confident and evidence-aware**:
    ``relation_type`` is one of the taxonomy in ``docs/brain-second/
    KNOWLEDGE_GRAPH.md`` (IS_A, PART_OF, DEPENDS_ON, …, POSSIBLE_RELATION), and
    each edge records confidence, evidence, reason and model version.  No edge
    is created without evidence (never on cosine similarity alone).
    """

    __tablename__ = "knowledge_links"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Polymorphic edge endpoints: source_id/target_id reference the node id of
    # whatever ``source_kind``/``target_kind`` name (source|chunk|unit|entity|tag).
    # There is deliberately NO FK to ``sources.id`` — a single FK cannot cover
    # all five node kinds, and the semantic layer writes entity/chunk/unit ids
    # here. The FK was dropped in the ingestion upgrade (see alembic
    # ``x1f3a2b9cdd1p``); referential integrity is enforced by the writer.
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(50), nullable=False, default="related")
    # Legacy default kept for backward compatibility; new edges use the full taxonomy.
    weight: Mapped[float] = mapped_column(default=1.0)
    # --- Semantic layer (Phase 3) ---
    # Node kinds on each end (source|chunk|unit|entity|tag). Legacy edges
    # carry 'source' so existing queries keep working.
    source_kind: Mapped[str] = mapped_column(String(20), nullable=False, default="source")
    target_kind: Mapped[str] = mapped_column(String(20), nullable=False, default="source")
    # Confidence in [0,1]; evidence is a JSONB list of provenance refs.
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    evidence: Mapped[list | None] = mapped_column(JSONB, default=list)
    evidence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reason: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(String(100))
    # Giai đoạn 5 — reuse tracking feeds the consolidation sweep: edges
    # retrieval keeps hitting gain a log-weighted reuse bonus; edges nobody
    # reads decay toward zero (exp half-life).
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    # lifecycle: active | review | superseded  (graph versioning, §76)
    lifecycle: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    valid_from: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # One directed edge per pair + relation type.
        Index("uq_knowledge_links_pair", "source_id", "target_id", "relation_type", unique=True),
        Index("idx_knowledge_links_target", "target_id"),
        Index("idx_knowledge_links_type", "relation_type"),
        Index("idx_knowledge_links_confidence", "confidence"),
    )


class Document(Base):
    """A chunked document in the knowledge base."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="text"
    )  # text, markdown, code, etc.
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # pgvector embedding — real `vector(768)` type (not JSONB)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)
    # Active vLLM embedding vector.  Legacy 768-dim data remains in the
    # original ``embedding`` column until the explicit re-embedding cutover.
    embedding_vllm_1024: Mapped[list[float] | None] = mapped_column(
        Vector(1024),
        nullable=True,
    )
    # Full-text search vector (auto-populated by trigger)
    fts_vector: Mapped[str | None] = mapped_column(TSVECTOR, nullable=True)
    # --- Classification (deep-ingest upgrade) -----------------------------
    # Denormalized from the owning source so retrieval can filter without a
    # join, plus the per-passage role that source-level labels cannot express.
    doc_type: Mapped[str | None] = mapped_column(String(32))
    domain: Mapped[str | None] = mapped_column(String(64))
    chunk_role: Mapped[str | None] = mapped_column(String(32))
    quality_tier: Mapped[str | None] = mapped_column(String(16))
    quality_score: Mapped[float | None] = mapped_column(Float)
    # Metadata
    tags: Mapped[list | None] = mapped_column(JSONB, default=list)
    # Giai đoạn 4 — denormalized from the owning source so the retrieval SQL
    # (which already filters on documents) can scope per session without a join.
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    source = relationship("Source", back_populates="documents")

    __table_args__ = (
        Index("idx_documents_content_fts", "fts_vector", postgresql_using="gin"),
        Index("idx_documents_tags", "tags", postgresql_using="gin"),
        Index("idx_documents_source_id", "source_id"),
        Index("idx_documents_doc_type", "doc_type"),
        Index("idx_documents_chunk_role", "chunk_role"),
        Index("idx_documents_quality_tier", "quality_tier"),
        Index("idx_documents_session_id", "session_id"),
    )


# --- Conversations / Chat History ---


class Conversation(Base):
    """A chat conversation."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False, default="New Chat")
    # Giai đoạn 4 — the visitor session that owns this conversation.
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    fingerprint_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """A single message in a conversation."""

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # user, assistant, system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Which documents were used as context (RAG citation)
    citations: Mapped[list | None] = mapped_column(JSONB, default=list)
    # Giai đoạn 4 — owning visitor session (denormalized for direct scoping).
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversation = relationship("Conversation", back_populates="messages")

    __table_args__ = (
        Index("idx_messages_conversation_id", "conversation_id"),
        Index("idx_messages_session_id", "session_id"),
    )


# --- Content Generation (Blog / Marketing) ---


class GeneratedContent(Base):
    """Generated marketing/blog content with metadata."""

    __tablename__ = "generated_content"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="blog"
    )  # blog, social, email, ad, seo
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="draft"
    )  # draft, published, archived
    source_documents: Mapped[list | None] = mapped_column(JSONB, default=list)
    tags: Mapped[list | None] = mapped_column(JSONB, default=list)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# --- Press / News / RSS Items ---


class NewsItem(Base):
    """News/RSS items fetched and stored for learning."""

    __tablename__ = "news_items"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text, unique=True)
    source_name: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    is_processed: Mapped[bool] = mapped_column(default=False)
    # After processing, link to document
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (Index("idx_news_url", "url", unique=True),)

    @property
    def word_count(self) -> int:
        return len(self.content.split()) if self.content else 0


# --- Scheduling / Tasks (replaces n8n) ---


class ScheduledTask(Base):
    """Scheduled tasks (replaces n8n workflows)."""

    __tablename__ = "scheduled_tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # rss_fetch, web_scrape, digest, cleanup
    cron_expression: Mapped[str | None] = mapped_column(String(100))
    interval_minutes: Mapped[int | None] = mapped_column(Integer)
    config: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    enabled: Mapped[bool] = mapped_column(default=True)
    last_run: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RegistryEntry(Base):
    """Dynamic UI/system registry entry (nav, command, widget, model, status).

    Business lists rendered by the frontend must come from this table (or a
    service that seeds/reads it), never from hardcoded frontend arrays.
    """

    __tablename__ = "registry_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)  # nav|command|widget|model|status
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    icon: Mapped[str | None] = mapped_column(String(100))
    href: Mapped[str | None] = mapped_column(String(255))
    action: Mapped[str | None] = mapped_column(String(100))
    group_name: Mapped[str | None] = mapped_column(String(100))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    enabled: Mapped[bool] = mapped_column(default=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_registry_entries_kind_key", "kind", "key", unique=True),
        Index("idx_registry_entries_kind_enabled", "kind", "enabled"),
        Index("idx_registry_entries_sort", "kind", "sort_order"),
    )


class AuditLog(Base):
    """Structured application audit trail for observability and security."""

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str | None] = mapped_column(String(64))
    actor: Mapped[str | None] = mapped_column(String(255))
    service: Mapped[str] = mapped_column(String(100), nullable=False, default="braind")
    module: Mapped[str | None] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="ok")
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_audit_logs_created", "created_at"),
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_request_id", "request_id"),
    )


class ScheduledTaskRun(Base):
    """Execution history for scheduler jobs."""

    __tablename__ = "scheduled_task_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_name: Mapped[str] = mapped_column(String(255), nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="running")
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    message: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_scheduled_task_runs_name", "task_name"),
        Index("idx_scheduled_task_runs_started", "started_at"),
    )


# --- Semantic Knowledge Layer (Phase 3, Brain Second) ---
# Additive tables for semantic chunking, entities, and knowledge provenance.
# All are uppercase-safe to be created idempotently and reference existing
# tables via FKs so nothing existing is altered structurally.


class Chunk(Base):
    """A semantic chunk with full context metadata.

    Chunks form a hierarchy (document → section → paragraph) and a linear
    chain (prev/next). ``section_path`` and ``context_summary`` preserve the
    ambient context so retrieval never loses meaning (golden CASE A).
    """

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    # Parent chunk (section → paragraph hierarchy); null for a root chunk.
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    section_path: Mapped[str | None] = mapped_column(Text)
    hierarchy_level: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # The chunk's own text. Offsets alone cannot recover it once the owning
    # document row is replaced, so the text is stored on the chunk itself.
    content: Mapped[str | None] = mapped_column(Text)
    # Byte offsets within the raw document content.
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Linear order inside an owning document.
    prev_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    next_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    context_summary: Mapped[str | None] = mapped_column(Text)
    # What this passage does inside its document (core/taxonomy.py roles).
    # Source-level labels cannot express it: one reference page holds both a
    # definition and a table, and a question wants only one of them.
    chunk_role: Mapped[str | None] = mapped_column(String(32))
    # Provenance + structural fidelity (ingestion upgrade, additive). These let a
    # chunk be located in the source document and idempotently re-reprocessed
    # without disturbing existing rows (all nullable).
    root_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )  # top-most section/document owning this chunk's subtree
    order: Mapped[int | None] = mapped_column(Integer, nullable=True)  # pre-order index
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)  # null when unparsed
    section_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_location: Mapped[str | None] = mapped_column(String(255))  # e.g. "§3 Heading / para 2"
    content_fingerprint: Mapped[str | None] = mapped_column(String(64))  # sha256 of node content
    # Versioned embedding (see embedding_versions registry).
    embedding_vllm_1024: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(100))
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_parent_id", "parent_id"),
        Index("idx_chunks_section_path", "section_path"),
        Index("idx_chunks_content_fingerprint", "document_id", "content_fingerprint"),
        Index("idx_chunks_role", "chunk_role"),
    )


class SemanticUnit(Base):
    """An atomic unit of meaning: fact, claim, event, definition, …

    ``polarity`` records negation (+1 / -1 / 0) so ``"does not use CUDA 13"``
    is never retrieved as a positive fact (golden CASE B). ``entity_ids``
    references resolved entities. Spans map back to the owning chunk.
    """

    __tablename__ = "semantic_units"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    unit_type: Mapped[str] = mapped_column(String(50), nullable=False, default="fact")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # +1 = affirmative, -1 = negated, 0 = neutral/unknown.
    polarity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Temporal validity window (§21, golden CASE C).
    valid_from: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    valid_to: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    entity_ids: Mapped[list | None] = mapped_column(JSONB, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding_vllm_1024: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    # fact_class: SOURCE_FACT | INFERRED_FACT | MODEL_GENERATED | USER_PROVIDED
    #            | SYSTEM_DERIVED | UNCERTAIN  (§41)
    fact_class: Mapped[str] = mapped_column(String(40), nullable=False, default="SOURCE_FACT")
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_units_chunk_id", "chunk_id"),
        Index("idx_units_valid_to", "valid_to"),
        Index("idx_units_fact_class", "fact_class"),
        CheckConstraint("polarity IN (-1, 0, 1)", name="ck_units_polarity"),
    )


class RetrievalFeedback(Base):
    """User feedback on retrieval results (Phase 11).

    Captures implicit signals (clicks) and explicit signals (thumbs up/down)
    so the reranker can be fine-tuned and hard negatives can be mined.
    """

    __tablename__ = "retrieval_feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_id: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)  # click | upvote | downvote
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_feedback_query", "query_text"),
        Index("idx_feedback_chunk", "chunk_id"),
        Index("idx_feedback_label", "label"),
    )


class Entity(Base):
    """A resolved entity with canonical name, aliases, and resolution state.

    Canonical identity (``canonical_name``) is distinct from aliases. Entities
    are never merged on embedding similarity alone (golden CASE E/F) — the
    resolution pipeline requires identity signals and an auditable policy.
    """

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Case- and whitespace-folded form used to detect duplicates that differ
    # only in presentation. 227 such collision groups existed before this.
    normalized_name: Mapped[str | None] = mapped_column(String(255))
    aliases: Mapped[list | None] = mapped_column(JSONB, default=list)
    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default="concept"
    )  # person|organization|place|concept|software|product|…
    description: Mapped[str | None] = mapped_column(Text)
    embedding_vllm_1024: Mapped[list[float] | None] = mapped_column(Vector(1024), nullable=True)
    # resolved | manual | ambiguous | review
    resolution_state: Mapped[str] = mapped_column(String(20), nullable=False, default="ambiguous")
    # auto | review | do_not_merge
    merge_policy: Mapped[str] = mapped_column(String(20), nullable=False, default="review")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_entities_canonical_name", "canonical_name", unique=True),
        Index("idx_entities_type", "entity_type"),
        Index("idx_entities_resolution_state", "resolution_state"),
        Index("idx_entities_normalized_name", "normalized_name"),
    )


class Tag(Base):
    """A taxonomy tag with hierarchy and synonyms (§14/§15).

    Tags are a graph of canonical nodes, not a comma-separated string.
    ``parent_id`` gives ``Domain → AI → Machine Learning`` structure and
    ``synonyms`` maps equivalent labels onto the canonical node.
    """

    __tablename__ = "tags"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tags.id", ondelete="SET NULL"), nullable=True
    )
    synonyms: Mapped[list | None] = mapped_column(JSONB, default=list)
    description: Mapped[str | None] = mapped_column(Text)
    domain: Mapped[str | None] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_tags_name", "name", "parent_id", unique=True),
        Index("idx_tags_parent_id", "parent_id"),
    )


class DocumentVersion(Base):
    """Immutable version record per source (§18)."""

    __tablename__ = "document_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    diff_summary: Mapped[str | None] = mapped_column(Text)
    reason: Mapped[str | None] = mapped_column(String(50))  # ingest|update|rollback|maintenance
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("uq_document_versions_source_version", "source_id", "version", unique=True),
        Index("idx_document_versions_source_id", "source_id"),
    )


class EmbeddingVersion(Base):
    """Registry of embedding-model generations (§24, §57).

    The active embedding model is the row with status='active'.  All
    embeddings reference this registry so a model migration is additive and
    versionable — never hard-coded.
    """

    __tablename__ = "embedding_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    # active | legacy | retired
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    config: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_embedding_versions_status", "status"),
        Index("uq_embedding_versions_model_dim", "model_name", "dimensions", unique=True),
    )


class MaintenanceRun(Base):
    """One nightly/scheduled maintenance execution (§12.1)."""

    __tablename__ = "maintenance_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scope: Mapped[str] = mapped_column(
        String(50), nullable=False, default="all"
    )  # chunk|unit|entity|graph|index|all
    # running | completed | failed | skipped
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    summary: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_maintenance_runs_status", "status"),
        Index("idx_maintenance_runs_started", "started_at"),
    )


class DirtyFlag(Base):
    """Incremental-maintenance dirty-marker (§47).

    When a document changes, affected chunks/entities/relations become dirty in
    a cascade; night-time maintenance processes only the flagged rows.
    """

    __tablename__ = "dirty_flags"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_kind: Mapped[str] = mapped_column(String(50), nullable=False)
    # document|chunk|unit|entity|tag|relation|summary
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    dirty_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # ingested|updated|deleted|relation_changed|embedding_stale
    reason: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_dirty_flags_kind_id", "entity_kind", "entity_id"),
        Index(
            "idx_dirty_flags_unprocessed",
            "processed_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )


class ModelRecord(Base):
    """Model usage registry (§58, §91)."""

    __tablename__ = "model_records"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # chat|embedding|reranker|extraction|reasoning
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    # active | legacy | retired
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    version: Mapped[str | None] = mapped_column(String(100))
    params: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_model_records_role_status", "role", "status"),
        Index("uq_model_records_role_model", "role", "model", unique=True),
    )


class FeedbackEvent(Base):
    """User feedback event for learning-to-rank (§38/§37)."""

    __tablename__ = "feedback_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # click|cite|skip|thumbs|reformulation
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_ids: Mapped[list | None] = mapped_column(JSONB, default=list)
    chosen_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    # 3=highly relevant, 2=relevant, 1=partial, 0=irrelevant, -1=misleading.
    label: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str | None] = mapped_column(String(50))  # click|cite|golden|human_review
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    embedding_version: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_feedback_events_created", "created_at"),
        Index("idx_feedback_events_query", "query"),
    )


class SearchEvidence(Base):
    """One web-search validation event recorded against a knowledge node.

    Ships with the search-validation pass (ingestion upgrade). A node may have
    many evidence rows, one per (node, search_query). Web results are stored as
    evidence / external-update candidates — they never overwrite the
    authoritative source node (the verdict row is the only mutation).
    """

    __tablename__ = "search_evidence"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Node scope — exactly one of these is set per row.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE")
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE")
    )
    unit_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("semantic_units.id", ondelete="CASCADE")
    )
    search_query: Mapped[str] = mapped_column(Text, nullable=False)
    web_source_url: Mapped[str | None] = mapped_column(Text)
    web_snippet: Mapped[str | None] = mapped_column(Text)
    web_fetched_text: Mapped[str | None] = mapped_column(Text)
    # VERIFIED | UPDATED | UNCHANGED | CONFLICTED | UNVERIFIED (see SearchValidator).
    verdict: Mapped[str] = mapped_column(String(30), nullable=False, default="UNVERIFIED")
    conflict_sentence: Mapped[str | None] = mapped_column(Text)
    external_update_candidate: Mapped[str | None] = mapped_column(Text)
    similarity_input_vs_web: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(100))
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_search_evidence_node", "chunk_id"),
        Index("idx_search_evidence_chunk_query", "chunk_id", "search_query", unique=True),
        Index("idx_search_evidence_verdict", "verdict"),
    )


class IngestionJob(Base):
    """Stage-level trace of one ingestion run (observability, Phase 7).

    ``ingestion_id`` groups the stages of a single ingest call; one row per
    stage (e.g. "legacy_chunk", "semantic_chunk", "entities", "freshness",
    "search_validate"). A failed stage records ``error`` and the row may be
    re-marked and replayed by the retry endpoint — it never deletes prior work.
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ingestion_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # The ingest stage belongs to a source, not to one of its chunk rows.
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sources.id", ondelete="SET NULL")
    )
    stage: Mapped[str] = mapped_column(String(60), nullable=False)
    # stage status: running | ok | failed
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="running")
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    tool: Mapped[str | None] = mapped_column(String(100))  # vllm.chat_json | web_search | ...
    model: Mapped[str | None] = mapped_column(String(255))
    tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_ingestion_jobs_ingestion_id", "ingestion_id"),
        Index("idx_ingestion_jobs_source_id", "source_id"),
        Index("idx_ingestion_jobs_status", "status"),
    )


class FreshnessState(Base):
    """Seeded taxonomy of freshness classes that classify knowledge nodes.

    Seed rows (STATIC, LONG_LIVED, TIME_SENSITIVE, REALTIME) are created
    idempotently by ``init_db``. ``search_validate`` gates which classes get
    web re-validation. Additive via ``name`` PK — never dropped.
    """

    __tablename__ = "freshness_states"

    name: Mapped[str] = mapped_column(String(40), primary_key=True)
    table_name: Mapped[str] = mapped_column(String(60), nullable=False, default="semantic_units")
    description: Mapped[str | None] = mapped_column(Text)
    llm_probe: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class QuarantinedKnowledge(Base):
    """Knowledge moved out of the active graph by the nightly maintenance.

    Giai đoạn 4: data that is corrupted is deleted; data that is merely
    contradicted, duplicated or ambiguous is quarantined here — hidden from
    retrieval, reviewable by a human, never silently destroyed. ``payload``
    holds the original row so quarantine is fully reversible.
    """

    __tablename__ = "quarantined_knowledge"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    table_name: Mapped[str] = mapped_column(String(60), nullable=False)
    row_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # duplicate | contradiction | ambiguous
    reason: Mapped[str] = mapped_column(String(40), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_quarantined_session", "session_id"),
        Index("idx_quarantined_reason", "reason"),
    )


class PolicyRule(Base):
    """One declarative governance rule (Giai đoạn 5, Layer 6).

    Stored in the database so rules are editable via REST without a restart.
    The engine treats ``deny`` as beating ``allow`` at equal priority; a
    human-set priority above everything sits above the engine itself.
    """

    __tablename__ = "policy_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # file_import | response | tool | category
    scope: Mapped[str] = mapped_column(String(60), nullable=False)
    rule_type: Mapped[str] = mapped_column(String(12), nullable=False)  # allow|deny|ask_first
    pattern: Mapped[str] = mapped_column(String(255), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_policy_rules_scope", "scope"),
        Index("idx_policy_rules_enabled", "enabled"),
    )


class RelationDecision(Base):
    """The training ledger for the consolidated classifier (Giai đoạn 5).

    Every relation decision — LLM verdict, consolidated-model verdict, or a
    human override — is recorded with its feature vector. The nightly
    consolidation retrains the small model from this ledger; human overrides
    carry ``human_override=true`` and receive priority weighting.
    """

    __tablename__ = "relation_decisions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    features: Mapped[dict] = mapped_column(JSONB, nullable=False)
    label: Mapped[str] = mapped_column(String(50), nullable=False)
    decided_by: Mapped[str] = mapped_column(String(60), nullable=False)  # llm|consolidated|human
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rationale: Mapped[str | None] = mapped_column(Text)
    human_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trained_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_relation_decisions_trained", "trained_at"),
        Index("idx_relation_decisions_override", "human_override"),
    )


# --- Dynamic taxonomy and runtime foundation ---


class TaxonomyFacet(Base):
    __tablename__ = "taxonomy_facets"

    facet: Mapped[str] = mapped_column(String(120), primary_key=True)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    growable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    authority_editable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class TaxonomyTerm(Base):
    __tablename__ = "taxonomy_terms"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    facet: Mapped[str] = mapped_column(
        String(120), ForeignKey("taxonomy_facets.facet", ondelete="CASCADE"), nullable=False
    )
    term: Mapped[str] = mapped_column(String(255), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomy_terms.id", ondelete="SET NULL")
    )
    origin: Mapped[str] = mapped_column(String(40), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    created_from_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL")
    )
    merged_into_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomy_terms.id", ondelete="SET NULL")
    )
    authority: Mapped[float | None] = mapped_column(Float)
    first_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    __table_args__ = (
        Index("uq_taxonomy_terms_facet_term", "facet", "term", unique=True),
        Index("idx_taxonomy_terms_status", "status"),
        Index("idx_taxonomy_terms_parent", "parent_id"),
    )


class TaxonomyAlias(Base):
    __tablename__ = "taxonomy_aliases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    term_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomy_terms.id", ondelete="CASCADE"), nullable=False
    )
    surface_form: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_form: Mapped[str] = mapped_column(String(255), nullable=False)
    locale: Mapped[str | None] = mapped_column(String(80))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "uq_taxonomy_aliases_normalized_locale",
            "normalized_form",
            text("COALESCE(locale, '')"),
            unique=True,
        ),
        Index("idx_taxonomy_aliases_term", "term_id"),
    )


class TaxonomyRule(Base):
    __tablename__ = "taxonomy_rules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    facet: Mapped[str] = mapped_column(
        String(120), ForeignKey("taxonomy_facets.facet", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(80), nullable=False)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    locale: Mapped[str | None] = mapped_column(String(80))
    origin: Mapped[str] = mapped_column(String(40), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hits: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index(
            "uq_taxonomy_rules_facet_kind_pattern_locale",
            "facet",
            "kind",
            "pattern",
            text("COALESCE(locale, '')"),
            unique=True,
        ),
        Index("idx_taxonomy_rules_enabled", "enabled"),
    )


class Capability(Base):
    __tablename__ = "capabilities"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(120), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    configuration: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_capabilities_kind_name", "kind", "name", unique=True),
        Index("idx_capabilities_enabled", "enabled"),
    )


class ReorgOperation(Base):
    __tablename__ = "reorg_operations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operation_type: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    target_type: Mapped[str] = mapped_column(String(80), nullable=False)
    target_id: Mapped[str] = mapped_column(String(255), nullable=False)
    before_state: Mapped[dict | None] = mapped_column(JSONB)
    after_state: Mapped[dict | None] = mapped_column(JSONB)
    rationale: Mapped[str | None] = mapped_column(Text)
    rollback_of_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reorg_operations.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_reorg_operations_status", "status"),
        Index("idx_reorg_operations_target", "target_type", "target_id"),
    )


class GPUResourceSample(Base):
    __tablename__ = "gpu_resource_samples"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gpu_index: Mapped[int] = mapped_column(Integer, nullable=False)
    total_vram_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    used_vram_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    free_vram_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    utilization_percent: Mapped[float | None] = mapped_column(Float)
    temperature_celsius: Mapped[float | None] = mapped_column(Float)
    sampled_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("idx_gpu_resource_samples_gpu_time", "gpu_index", "sampled_at"),)


class Agent(Base):
    __tablename__ = "agents"

    agent_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    department: Mapped[str | None] = mapped_column(String(50))
    specialization: Mapped[str | None] = mapped_column(String(50))
    avatar: Mapped[str | None] = mapped_column(String(10))
    color: Mapped[str | None] = mapped_column(String(7))
    system_prompt: Mapped[str | None] = mapped_column(Text)
    temperature: Mapped[float | None] = mapped_column(Float)
    can_delegate: Mapped[bool | None] = mapped_column(Boolean)
    can_use_tools: Mapped[bool | None] = mapped_column(Boolean)
    available_tools: Mapped[dict | list | None] = mapped_column(JSONB)
    position: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[str | None] = mapped_column(String(20))
    level: Mapped[str | None] = mapped_column(String(20))
    parent_agent: Mapped[str | None] = mapped_column(String(50))
    capabilities: Mapped[dict | list | None] = mapped_column(JSONB)
    current_task: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class AgentTaskRecord(Base):
    __tablename__ = "agent_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_task_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agent_tasks.task_id")
    )
    user_request: Mapped[str | None] = mapped_column(Text)
    task_description: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_to: Mapped[str] = mapped_column(
        String(50), ForeignKey("agents.agent_id"), nullable=False
    )
    assigned_by: Mapped[str | None] = mapped_column(String(50), ForeignKey("agents.agent_id"))
    status: Mapped[str | None] = mapped_column(String(20))
    priority: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[dict | list | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    task_type: Mapped[str | None] = mapped_column(String(20))
    state: Mapped[str | None] = mapped_column(String(30))
    department: Mapped[str | None] = mapped_column(String(50))
    root_task_id: Mapped[str | None] = mapped_column(String(36))
    objective: Mapped[str | None] = mapped_column(Text)
    instructions: Mapped[str | None] = mapped_column(Text)
    expected_output: Mapped[str | None] = mapped_column(Text)
    required_output: Mapped[str | None] = mapped_column(Text)
    acceptance_criteria: Mapped[dict | list | None] = mapped_column(JSONB)
    input_data: Mapped[dict | list | None] = mapped_column(JSONB)
    constraints: Mapped[dict | list | None] = mapped_column(JSONB)
    dependencies: Mapped[dict | list | None] = mapped_column(JSONB)
    confirmation_required: Mapped[bool | None] = mapped_column(Boolean)
    rework_count: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        Index("idx_agent_tasks_status", "status"),
        Index("idx_agent_tasks_assigned_to", "assigned_to"),
        Index("idx_agent_tasks_root", "root_task_id"),
        Index("idx_agent_tasks_state", "state"),
        Index("idx_agent_tasks_type", "task_type"),
    )


class AgentConversation(Base):
    __tablename__ = "agent_conversations"

    conversation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("agent_tasks.task_id"))
    from_agent: Mapped[str] = mapped_column(
        String(50), ForeignKey("agents.agent_id"), nullable=False
    )
    to_agent: Mapped[str] = mapped_column(String(50), ForeignKey("agents.agent_id"), nullable=False)
    message_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("idx_agent_conversations_task", "task_id"),)


class AgentReport(Base):
    __tablename__ = "agent_reports"

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), nullable=False)
    root_task_id: Mapped[str | None] = mapped_column(String(36))
    department_task_id: Mapped[str | None] = mapped_column(String(36))
    agent_id: Mapped[str] = mapped_column(String(50), nullable=False)
    report_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict | list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_agent_reports_task", "task_id"),
        Index("idx_agent_reports_root", "root_task_id"),
    )


class AgentAuditEvent(Base):
    __tablename__ = "agent_audit_events"

    event_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(50))
    task_id: Mapped[str | None] = mapped_column(String(36))
    parent_task_id: Mapped[str | None] = mapped_column(String(36))
    root_task_id: Mapped[str | None] = mapped_column(String(36))
    previous_status: Mapped[str | None] = mapped_column(String(30))
    new_status: Mapped[str | None] = mapped_column(String(30))
    reason: Mapped[str | None] = mapped_column(Text)
    event_metadata: Mapped[dict | list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_agent_audit_root", "root_task_id"),
        Index("idx_agent_audit_task", "task_id"),
        Index("idx_agent_audit_action", "action"),
    )


class AgentMemory(Base):
    __tablename__ = "agent_memory"

    memory_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    root_task_id: Mapped[str | None] = mapped_column(String(36))
    task_id: Mapped[str | None] = mapped_column(String(36))
    department: Mapped[str | None] = mapped_column(String(50))
    agent_id: Mapped[str | None] = mapped_column(String(50))
    knowledge_type: Mapped[str] = mapped_column(String(30), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[dict | list | None] = mapped_column(JSONB)
    created_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_agent_memory_root", "root_task_id"),
        Index("idx_agent_memory_type", "knowledge_type"),
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    root_task_id: Mapped[str] = mapped_column(String(36), nullable=False)
    state: Mapped[str] = mapped_column(String(40), nullable=False)
    checkpoint_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    budget_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_agent_runs_root_task", "root_task_id", unique=True),
        Index("idx_agent_runs_state", "state"),
    )


class AgentDomain(Base):
    __tablename__ = "agent_domains"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("agents.agent_id", ondelete="CASCADE"), nullable=False
    )
    facet: Mapped[str] = mapped_column(
        String(120), ForeignKey("taxonomy_facets.facet", ondelete="CASCADE"), nullable=False
    )
    term_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("taxonomy_terms.id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    origin: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("uq_agent_domains_agent_term", "agent_id", "term_id", unique=True),
        Index("idx_agent_domains_facet", "facet"),
    )


# --- Settings foundation (Brain Second Phase 1a) ---


class SettingDefinition(Base):
    """Typed schema for one setting key (Brain Second Phase 1a).

    The single source of truth for *what* a setting is — its type, bounds,
    allowed values, risk and restart semantics — so neither the UI (Phase 6) nor
    the resolver duplicates that knowledge. ``default_value`` is JSONB so one
    column carries every data_type (bool/int/float/str/enum/json/list) safely.

    ``also_in_categories`` lets a key surface under more than one UI sidebar
    section without duplicating the row (Phase 0 decision Q6: the two keys that
    appear under both Core Behavior and Behavior Rules are stored once).
    """

    __tablename__ = "setting_definitions"

    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    category: Mapped[str] = mapped_column(String(40), nullable=False)
    also_in_categories: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # bool|int|float|str|enum|json|string_list
    data_type: Mapped[str] = mapped_column(String(16), nullable=False)
    default_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    min_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    allowed_values: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    description: Mapped[str] = mapped_column(String(255), nullable=False)
    detailed_description: Mapped[str | None] = mapped_column(Text)
    risk_level: Mapped[str] = mapped_column(
        String(12), nullable=False, server_default=text("'low'")
    )  # low|medium|high|critical
    requires_restart: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    is_sensitive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    is_critical: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    validation_rules: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    source: Mapped[str] = mapped_column(String(60), nullable=False)  # appendix_a|derived_phaseN
    rationale: Mapped[str | None] = mapped_column(Text)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_setting_definitions_category", "category"),
        Index("idx_setting_definitions_critical", "is_critical"),
    )


class SettingValue(Base):
    """The current value of one setting at one scope (Brain Second Phase 1a).

    One table serves all seven priority scopes (system_hard > security > project
    > task > user_preference > temporary > learned); ``scope_ref`` narrows a
    scoped row to a session/task/preference id and is NULL for global rows. The
    resolver (Phase 1b) picks the highest-ranked scope that has a row.

    The unique index uses ``COALESCE(scope_ref, '')`` because Postgres treats
    NULLs as distinct in a plain UNIQUE constraint — without the COALESCE two
    global rows for the same (key, scope) would both be allowed (Phase 1a R12).
    """

    __tablename__ = "setting_values"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(120), nullable=False)
    scope: Mapped[str] = mapped_column(String(24), nullable=False)
    scope_ref: Mapped[str | None] = mapped_column(String(120), nullable=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_by: Mapped[str | None] = mapped_column(String(60))  # seed|api|system|learned|migration
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        # Expression index on COALESCE: a plain UNIQUE(key, scope, scope_ref)
        # would let two global rows (scope_ref IS NULL) coexist, because Postgres
        # treats NULLs as distinct. Wrapping scope_ref in COALESCE makes the
        # "global" case collide as intended (Phase 1a R12).
        Index(
            "uq_setting_values_key_scope_ref",
            text("key"),
            text("scope"),
            text("COALESCE(scope_ref, '')"),
            unique=True,
        ),
        Index("idx_setting_values_key", "key"),
        Index("idx_setting_values_scope", "scope"),
    )


class SettingHistory(Base):
    """Append-only version log for setting changes (Brain Second Phase 1a).

    Every write through ``SettingsStore.set_value`` appends a row carrying the
    previous and new value, who changed it and why — so there is no silent
    change, and ``rollback`` can restore any past version without ever deleting
    history. ``snapshot_id`` is reserved for the Phase 1b snapshot link.
    """

    __tablename__ = "setting_history"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    scope: Mapped[str] = mapped_column(String(24), nullable=False)
    scope_ref: Mapped[str | None] = mapped_column(String(120), nullable=True)
    previous_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    changed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    changed_by: Mapped[str] = mapped_column(String(60), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    snapshot_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    __table_args__ = (
        Index("idx_setting_history_key_version", "key", "version"),
        Index("idx_setting_history_changed_at", "changed_at"),
    )


class SettingsSnapshot(Base):
    """An immutable frozen view of resolved settings (Brain Second Phase 1a).

    Created at task start so a running task keeps using its own snapshot even if
    settings change underneath it mid-run (Phase 1 DoD). The table is created in
    1a so the migration lands once; the resolver that *populates* it is Phase 1b.
    ``resolved_from`` records which scope won each key — the data the Phase 6
    "Effective View" renders (raw vs effective + why).
    """

    __tablename__ = "settings_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    values: Mapped[dict] = mapped_column(JSONB, nullable=False)
    resolved_from: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_settings_snapshots_task", "task_id"),
        Index("idx_settings_snapshots_session", "session_id"),
    )


# --- Database Engine Factory ---


def create_engine(database_url: str | None = None):
    """Create async SQLAlchemy engine."""
    from .config import settings as _settings

    url = database_url or _settings.database_url
    return create_async_engine(url, pool_size=10, max_overflow=20, echo=_settings.debug)


async def _create_vector_index(
    engine,
    index_name: str,
    column_name: str,
    table_name: str = "documents",
) -> None:
    """Create a required vector index, falling back from HNSW to IVFFlat.

    A PostgreSQL transaction cannot be reused after a failed DDL statement, so
    the HNSW attempt and its IVFFlat fallback each run in an independent
    transaction. If both strategies fail the error is fatal: a knowledge
    retrieval service must not start without a usable ANN index.

    Args:
        engine: The async SQLAlchemy engine to open transactions on.
        index_name: Name of the index to create idempotently.
        column_name: The vector column the index is built over.
        table_name: The owning table (defaults to ``documents``).

    Raises:
        DatabaseInitializationError: If neither HNSW nor IVFFlat can be created.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    f"ON {table_name} USING hnsw ({column_name} vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 200)"
                )
            )
    except Exception:
        logger.warning(
            "database.hnsw_index_failed",
            index_name=index_name,
            exc_info=True,
        )
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} "
                        f"ON {table_name} USING ivfflat ({column_name} vector_cosine_ops) "
                        "WITH (lists = 100)"
                    )
                )
        except Exception as ivfflat_error:
            logger.error(
                "database.ivfflat_index_failed",
                index_name=index_name,
                exc_info=True,
            )
            raise DatabaseInitializationError(
                f"Unable to create required vector index {index_name}"
            ) from ivfflat_error


async def init_db(engine=None):
    """Create all tables and enable pgvector extension."""
    if engine is None:
        engine = create_engine()

    # Enable pgvector + pg_trgm extensions
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Ensure the embedding column is a real `vector(768)` type.
    # In earlier versions it was JSONB — detect and fix that in place if needed.
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT data_type, udt_name FROM information_schema.columns "
                    "WHERE table_name = 'documents' AND column_name = 'embedding'"
                )
            )
        ).first()
        if row is None:
            # Table may not exist yet (fresh DB) — declare the column
            await conn.execute(
                text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS embedding vector(768)")
            )
        elif row.udt_name != "vector":
            # Old JSONB column: drop and recreate as vector(768).
            logger.warning("database.dropping_legacy_jsonb_embedding")
            await conn.execute(text("ALTER TABLE documents DROP COLUMN embedding"))
            await conn.execute(text("ALTER TABLE documents ADD COLUMN embedding vector(768)"))

    # Additive 1024-dimension vLLM target column. This deliberately retains
    # legacy 768-dimension data until a verified, reversible manual cutover.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE documents ADD COLUMN IF NOT EXISTS "
                "embedding_vllm_1024 vector(1024), "
                "ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        )

    # --- Giai đoạn 4: session isolation (additive, idempotent) ------------
    # Mirrors alembic f973753a37c2 so deployments that skip migrations still
    # get the scoping columns. NULL session_id = system/global.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE sources ADD COLUMN IF NOT EXISTS session_id UUID, "
                "ADD COLUMN IF NOT EXISTS fingerprint_hash VARCHAR(64)"
            )
        )
        await conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id UUID"))
        await conn.execute(
            text(
                "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS session_id UUID, "
                "ADD COLUMN IF NOT EXISTS fingerprint_hash VARCHAR(64)"
            )
        )
        await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS session_id UUID"))
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_sources_session_id ON sources(session_id)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id)")
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_conversations_session_id "
                "ON conversations(session_id)"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)")
        )
        # Quarantine table — created here too so create_all-less deployments
        # (init_db-only) still get it; create_all covers ORM-mapped cases.
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS quarantined_knowledge ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
                "  session_id UUID,"
                "  table_name VARCHAR(60) NOT NULL,"
                "  row_id VARCHAR(64) NOT NULL,"
                "  reason VARCHAR(40) NOT NULL,"
                "  detail TEXT,"
                "  payload JSONB,"
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_quarantined_session "
                "ON quarantined_knowledge(session_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_quarantined_reason ON quarantined_knowledge(reason)"
            )
        )

    # HNSW and its IVFFlat fallback each run in an independent transaction; see
    # _create_vector_index. Both required vector columns are indexed explicitly.
    await _create_vector_index(engine, "idx_documents_embedding_vllm_1024", "embedding_vllm_1024")
    await _create_vector_index(engine, "idx_documents_embedding", "embedding")

    # Create the FTS function and trigger idempotently. Failure is fatal because
    # hybrid retrieval depends on fresh full-text vectors.
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE OR REPLACE FUNCTION documents_fts_trigger() "
                    "RETURNS trigger AS $$ "
                    "BEGIN "
                    "  NEW.fts_vector := to_tsvector('english', "
                    "    COALESCE(NEW.title, '') || ' ' || COALESCE(NEW.content, '')); "
                    "  RETURN NEW; "
                    "END; "
                    "$$ LANGUAGE plpgsql"
                )
            )
            await conn.execute(text("DROP TRIGGER IF EXISTS trg_documents_fts ON documents"))
            await conn.execute(
                text(
                    "CREATE TRIGGER trg_documents_fts "
                    "BEFORE INSERT OR UPDATE ON documents "
                    "FOR EACH ROW EXECUTE FUNCTION documents_fts_trigger()"
                )
            )
    except Exception as exc:
        logger.error("database.fts_setup_failed", exc_info=True)
        raise DatabaseInitializationError("Unable to create the document FTS trigger") from exc

    # --- Phase 2 schema upgrade: knowledge graph (additive, idempotent) ---
    async with engine.begin() as conn:
        # 1) Source new columns: content_hash, tags, parent_id.
        await conn.execute(
            text(
                "ALTER TABLE sources "
                "ADD COLUMN IF NOT EXISTS content_hash TEXT, "
                "ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'::jsonb, "
                "ADD COLUMN IF NOT EXISTS parent_id UUID "
                "REFERENCES sources(id) ON DELETE SET NULL"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_content_hash "
                "ON sources(content_hash) WHERE content_hash IS NOT NULL"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_sources_parent_id ON sources(parent_id)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type)")
        )

        # 1b) Deep-ingest classification columns (mirrors alembic e51c8a3f7b92).
        # Closed vocabularies from core/taxonomy.py; retrieval filters on them,
        # so they are real columns rather than JSONB keys.
        await conn.execute(
            text(
                "ALTER TABLE sources "
                "ADD COLUMN IF NOT EXISTS doc_type VARCHAR(32), "
                "ADD COLUMN IF NOT EXISTS domain VARCHAR(64), "
                "ADD COLUMN IF NOT EXISTS audience VARCHAR(32), "
                "ADD COLUMN IF NOT EXISTS quality_tier VARCHAR(16), "
                "ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION, "
                "ADD COLUMN IF NOT EXISTS language VARCHAR(8)"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE documents "
                "ADD COLUMN IF NOT EXISTS doc_type VARCHAR(32), "
                "ADD COLUMN IF NOT EXISTS domain VARCHAR(64), "
                "ADD COLUMN IF NOT EXISTS chunk_role VARCHAR(32), "
                "ADD COLUMN IF NOT EXISTS quality_tier VARCHAR(16), "
                "ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION"
            )
        )
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_sources_doc_type ON sources(doc_type)",
            "CREATE INDEX IF NOT EXISTS idx_sources_domain ON sources(domain)",
            "CREATE INDEX IF NOT EXISTS idx_sources_quality_tier ON sources(quality_tier)",
            "CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_chunk_role ON documents(chunk_role)",
            "CREATE INDEX IF NOT EXISTS idx_documents_quality_tier ON documents(quality_tier)",
        ):
            await conn.execute(text(statement))

        # 2) knowledge_links table — normalized, indexed, weighted graph edges.
        # Polymorphic endpoints: source_id/target_id hold the node id named by
        # source_kind/target_kind (source|chunk|unit|entity|tag), so there is no
        # FK to sources.id (dropped in the ingestion upgrade; see alembic
        # ``x1f3a2b9cdd1p``). Integrity is enforced by the writer.
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS knowledge_links ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  source_id UUID NOT NULL, "
                "  target_id UUID NOT NULL, "
                "  relation_type TEXT NOT NULL DEFAULT 'related', "
                "  weight DOUBLE PRECISION NOT NULL DEFAULT 1.0, "
                "  meta JSONB DEFAULT '{}'::jsonb, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_links_pair "
                "ON knowledge_links(source_id, target_id, relation_type)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_links_target "
                "ON knowledge_links(target_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_links_type "
                "ON knowledge_links(relation_type)"
            )
        )

    # --- Learning crawler audit tables (additive, idempotent) ---
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS learn_runs ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  topic TEXT NOT NULL, "
                "  status TEXT NOT NULL DEFAULT 'pending', "
                "  params JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "  summary JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "  log JSONB NOT NULL DEFAULT '[]'::jsonb, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS learn_urls ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  run_id UUID NOT NULL REFERENCES learn_runs(id) ON DELETE CASCADE, "
                "  url TEXT NOT NULL, "
                "  url_norm TEXT NOT NULL, "
                "  depth INTEGER NOT NULL DEFAULT 0, "
                "  status TEXT NOT NULL DEFAULT 'queued', "
                "  title TEXT, "
                "  source_id UUID REFERENCES sources(id) ON DELETE SET NULL, "
                "  links_out JSONB NOT NULL DEFAULT '[]'::jsonb, "
                "  fetched_at TIMESTAMPTZ, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_learn_urls_run_url "
                "ON learn_urls(run_id, url_norm)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_learn_urls_run_status ON learn_urls(run_id, status)"
            )
        )
        # Global normalized-URL history — used for cross-run duplicate checks.
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_learn_urls_url_norm ON learn_urls(url_norm)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_learn_runs_created ON learn_runs(created_at DESC)")
        )

    # --- Robinson registry + audit tables (additive, idempotent) ---
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS registry_entries ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  kind TEXT NOT NULL, "
                "  key TEXT NOT NULL, "
                "  label TEXT NOT NULL, "
                "  description TEXT, "
                "  icon TEXT, "
                "  href TEXT, "
                "  action TEXT, "
                "  group_name TEXT, "
                "  sort_order INTEGER NOT NULL DEFAULT 0, "
                "  enabled BOOLEAN NOT NULL DEFAULT TRUE, "
                "  meta JSONB DEFAULT '{}'::jsonb, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_registry_entries_kind_key "
                "ON registry_entries(kind, key)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_registry_entries_kind_enabled "
                "ON registry_entries(kind, enabled)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_registry_entries_sort "
                "ON registry_entries(kind, sort_order)"
            )
        )
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS audit_logs ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  request_id TEXT, "
                "  actor TEXT, "
                "  service TEXT NOT NULL DEFAULT 'braind', "
                "  module TEXT, "
                "  action TEXT NOT NULL, "
                "  status TEXT NOT NULL DEFAULT 'ok', "
                "  duration_ms INTEGER, "
                "  message TEXT, "
                "  meta JSONB DEFAULT '{}'::jsonb, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON audit_logs(created_at)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_audit_logs_request_id ON audit_logs(request_id)")
        )
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS scheduled_task_runs ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  task_name TEXT NOT NULL, "
                "  task_type TEXT, "
                "  status TEXT NOT NULL DEFAULT 'running', "
                "  started_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "  finished_at TIMESTAMPTZ, "
                "  duration_ms INTEGER, "
                "  message TEXT, "
                "  meta JSONB DEFAULT '{}'::jsonb"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_name "
                "ON scheduled_task_runs(task_name)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_started "
                "ON scheduled_task_runs(started_at)"
            )
        )

    # --- Brain Second semantic layer (Phase 3, additive + idempotent) ---
    # The tables themselves are created by ``Base.metadata.create_all`` above.
    # This block hardens *existing* installs by adding the typed-edge columns
    # to ``knowledge_links`` and the vector indexes to the semantic tables,
    # all with ``IF NOT EXISTS`` so re-runs are no-ops.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE knowledge_links "
                "ADD COLUMN IF NOT EXISTS source_kind TEXT NOT NULL DEFAULT 'source', "
                "ADD COLUMN IF NOT EXISTS target_kind TEXT NOT NULL DEFAULT 'source', "
                "ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0, "
                "ADD COLUMN IF NOT EXISTS evidence JSONB NOT NULL DEFAULT '[]'::jsonb, "
                "ADD COLUMN IF NOT EXISTS evidence_count INTEGER NOT NULL DEFAULT 0, "
                "ADD COLUMN IF NOT EXISTS reason TEXT, "
                "ADD COLUMN IF NOT EXISTS model_version TEXT, "
                "ADD COLUMN IF NOT EXISTS lifecycle TEXT NOT NULL DEFAULT 'active', "
                "ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ, "
                "ADD COLUMN IF NOT EXISTS valid_to TIMESTAMPTZ"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_links_confidence "
                "ON knowledge_links(confidence)"
            )
        )
        # Polymorphic edges: source_id/target_id may reference entity/chunk/unit/
        # tag ids, not just sources. Drop the legacy FK to sources.id so the
        # semantic layer can write entity edges (idempotent — no-op if already
        # dropped). Matches alembic ``x1f3a2b9cdd1p``.
        await conn.execute(
            text(
                "ALTER TABLE knowledge_links "
                "DROP CONSTRAINT IF EXISTS knowledge_links_source_id_fkey"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE knowledge_links "
                "DROP CONSTRAINT IF EXISTS knowledge_links_target_id_fkey"
            )
        )

    # --- Ingestion upgrade (Phase 1): additive provenance + observability +    ---
    # --- search-validation tables. Idempotent; the matching alembic revision   ---
    # --- (x1f3a2b9cdd1p) mirrors exactly this DDL.                              ---
    async with engine.begin() as conn:
        # 1) Additive chunk provenance columns (all nullable — existing rows valid).
        await conn.execute(
            text(
                "ALTER TABLE chunks "
                "ADD COLUMN IF NOT EXISTS root_id UUID, "
                'ADD COLUMN IF NOT EXISTS "order" INTEGER, '
                "ADD COLUMN IF NOT EXISTS page_number INTEGER, "
                "ADD COLUMN IF NOT EXISTS section_number INTEGER, "
                "ADD COLUMN IF NOT EXISTS source_location TEXT, "
                "ADD COLUMN IF NOT EXISTS content_fingerprint TEXT"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_content_fingerprint "
                "ON chunks(document_id, content_fingerprint)"
            )
        )

        # 1b) Deep-ingest classification (mirrors alembic e51c8a3f7b92).
        await conn.execute(
            text("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunk_role VARCHAR(32)")
        )
        await conn.execute(
            text("ALTER TABLE entities ADD COLUMN IF NOT EXISTS normalized_name VARCHAR(255)")
        )
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_role ON chunks(chunk_role)"))
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_entities_normalized_name "
                "ON entities(normalized_name)"
            )
        )

        # 2) search_evidence — one validated web-evidence row per (node, query).
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS search_evidence ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  document_id UUID REFERENCES documents(id) ON DELETE CASCADE, "
                "  chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE, "
                "  unit_id UUID REFERENCES semantic_units(id) ON DELETE CASCADE, "
                "  search_query TEXT NOT NULL, "
                "  web_source_url TEXT, "
                "  web_snippet TEXT, "
                "  web_fetched_text TEXT, "
                "  verdict TEXT NOT NULL DEFAULT 'UNVERIFIED', "
                "  conflict_sentence TEXT, "
                "  external_update_candidate TEXT, "
                "  similarity_input_vs_web DOUBLE PRECISION, "
                "  model_version TEXT, "
                "  meta JSONB DEFAULT '{}'::jsonb, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_search_evidence_node ON search_evidence(chunk_id)")
        )
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_search_evidence_chunk_query "
                "ON search_evidence(chunk_id, search_query)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_search_evidence_verdict ON search_evidence(verdict)"
            )
        )

        # 3) ingestion_jobs — stage-level observability/retry trace.
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS ingestion_jobs ("
                "  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
                "  ingestion_id UUID NOT NULL, "
                "  source_id UUID REFERENCES sources(id) ON DELETE SET NULL, "
                "  stage TEXT NOT NULL, "
                "  status TEXT NOT NULL DEFAULT 'running', "
                "  started_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "  finished_at TIMESTAMPTZ, "
                "  duration_ms INTEGER, "
                "  tool TEXT, "
                "  model TEXT, "
                "  tokens INTEGER, "
                "  error TEXT, "
                "  meta JSONB DEFAULT '{}'::jsonb"
                ")"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_ingestion_id "
                "ON ingestion_jobs(ingestion_id)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_source_id "
                "ON ingestion_jobs(source_id)"
            )
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(status)")
        )

        # 4) freshness_states — seeded taxonomy (idempotent upsert).
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS freshness_states ("
                "  name TEXT PRIMARY KEY, "
                "  table_name TEXT NOT NULL, "
                "  description TEXT, "
                "  llm_probe BOOLEAN NOT NULL DEFAULT FALSE, "
                "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
        )
        await conn.execute(
            text(
                "INSERT INTO freshness_states (name, table_name, llm_probe, description) "
                "VALUES "
                "  ('STATIC', 'semantic_units', FALSE, "
                "'Evergreen reference material; never web-revalidated'), "
                "  ('LONG_LIVED', 'semantic_units', FALSE, "
                "'Changes slowly (years); revalidate only on schedule'), "
                "  ('TIME_SENSITIVE', 'semantic_units', TRUE, "
                "'Changes quickly; primary search-validation target'), "
                "  ('REALTIME', 'semantic_units', TRUE, "
                "'Live/current; highest search-validation priority') "
                "ON CONFLICT (name) DO NOTHING"
            )
        )

    # Vector indexes for the semantic tables (HNSW with IVFFlat fallback).
    await _create_vector_index(
        engine, "idx_chunks_embedding_vllm_1024", "embedding_vllm_1024", "chunks"
    )
    await _create_vector_index(
        engine, "idx_units_embedding_vllm_1024", "embedding_vllm_1024", "semantic_units"
    )
    await _create_vector_index(
        engine, "idx_entities_embedding_vllm_1024", "embedding_vllm_1024", "entities"
    )

    return engine


def create_session(engine=None):
    """Create async session factory."""
    if engine is None:
        engine = create_engine()
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
