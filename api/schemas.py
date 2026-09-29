"""Braind API - Pydantic schemas (v2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --- Chat ---


class ChatRequest(BaseModel):
    """A chat message."""

    message: str = Field(..., min_length=1, max_length=20000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    content: str
    content_html: str = ""
    citations: list[dict] = Field(default_factory=list)
    steps: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)


# --- Ingest ---


class IngestTextRequest(BaseModel):
    content: str = Field(..., min_length=1)
    title: str = ""
    source_type: str = "manual"
    path: str | None = None
    url: str | None = None
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source: str
    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    characters_read: int = 0
    errors: list[str] = Field(default_factory=list)
    job_id: str | None = None


class ObsidianExportResponse(BaseModel):
    """Result of a full Obsidian display-layer rebuild (§3)."""

    documents: int = 0
    entities: int = 0
    canvases: int = 0
    pruned: int = 0
    errors: list[str] = Field(default_factory=list)


class IngestQualityReport(BaseModel):
    """Aggregate data-quality report for the ingestion pipeline (Phase 7)."""

    document_count: int = 0
    source_count: int = 0
    chunk_count: int = 0
    unit_count: int = 0
    entity_count: int = 0
    concept_count: int = 0
    link_count: int = 0
    needs_review_count: int = 0
    resolution_conflict_count: int = 0
    conflicted_count: int = 0
    duplicate_count: int = 0
    failed_stage_count: int = 0
    freshness_breakdown: dict[str, int] = Field(default_factory=dict)
    recent_jobs: list[dict[str, Any]] = Field(default_factory=list)


# --- Search ---


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=50)
    tags: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    results: list[dict[str, Any]]


# --- Writer ---


class WriterRequest(BaseModel):
    """Content generation request where ``topic`` is a title/topic hint."""

    content_type: Literal[
        "blog",
        "social",
        "email",
        "ad",
        "seo",
        "summary",
        "newsletter",
        "case_study",
        "product_desc",
        "press_release",
        "thread",
        "landing_page",
    ]
    topic: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Title/topic hint used to generate the final article title.",
    )
    keywords: list[str] = Field(default_factory=list, max_length=12)
    tone: Literal[
        "chuyên nghiệp",
        "thân thiện",
        "trang trọng",
        "hài hước",
        "thuyết phục",
        "truyền cảm hứng",
        "bình dân",
    ] = "chuyên nghiệp"
    audience: Literal[
        "người mới", "chuyên gia", "lãnh đạo", "sinh viên", "khách hàng B2B", "khách hàng B2C"
    ] = "người mới"
    word_count: int = Field(600, ge=100, le=3000)
    output_format: Literal["markdown", "plain", "html"] = "plain"
    article_count: int = Field(1, ge=1, le=3)
    language: Literal["vi", "en", "vi-en"] = "vi"
    check_web_duplicates: bool = False
    ingest_to_brain: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_writer(self) -> Any:
        """Convert to the writer service's WriterRequest dataclass."""
        from services.writer.service import WriterRequest as WR

        return WR(
            content_type=self.content_type,
            topic=self.topic,
            keywords=self.keywords,
            tone=self.tone,
            audience=self.audience,
            word_count=self.word_count,
            output_format=self.output_format,
            article_count=self.article_count,
            language=self.language,
            check_web_duplicates=self.check_web_duplicates,
            ingest_to_brain=self.ingest_to_brain,
            extra=self.extra,
        )


class WriterGeneratedItem(BaseModel):
    """One generated variant in a batch response."""

    content_id: str
    content_type: str
    topic: str
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)


class WriterBatchResponse(BaseModel):
    """Response for a batch generation request."""

    batch_id: str
    requested_count: int
    completed_count: int
    items: list[WriterGeneratedItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WriterResponse(BaseModel):
    content_id: str
    content_type: str
    topic: str
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)


class RefineRequest(BaseModel):
    content_id: str
    instruction: str = Field(..., min_length=1)


class ContentListItem(BaseModel):
    content_id: str
    content_type: str
    topic: str
    status: str
    tags: list[str] = Field(default_factory=list)
    created_at: str | None = None


# --- Scheduler ---


class SchedulerJob(BaseModel):
    name: str
    task_type: str
    next_run: str | None = None


class RunJobRequest(BaseModel):
    name: str


# --- Deep Learning ---


class DeepLearnRequest(BaseModel):
    """Request to start a deep-learning BFS crawl."""

    topic: str = Field(..., min_length=2, max_length=200)
    keywords: str = Field(default="", max_length=500)
    depth: int = Field(2, ge=1, le=3)
    max_items: int = Field(3, ge=1, le=5)
