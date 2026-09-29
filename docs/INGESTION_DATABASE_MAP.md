# BrainD Data Ingestion — Database Map (Current State)

> Audit deliverable, Phase 0. Every table, column, index, and pgvector usage relevant to ingestion.
> All models live in `core/database.py`; paths relative to `braind/`.

## Schema conventions
- All PKs `UUID(as_uuid=True)` with `default=uuid.uuid4`.
- Timestamps `DateTime(timezone=True)`, `server_default=func.now()`.
- Vectors via `pgvector.sqlalchemy.Vector`.
- **Migration reality:** base tables (`sources`, `documents`, `knowledge_links`, …) are created
  idempotently at startup by `core/database.py init_db()` (`create_all` + `CREATE TABLE IF NOT EXISTS`
  / `ALTER ... ADD COLUMN IF NOT EXISTS` / `CREATE UNIQUE INDEX IF NOT EXISTS`).
  Alembic manages the "Brain Second" and ingestion tables — chain `16dbaf38ae11` → `b2f9c4a7d310`
  → `x1f3a2b9cdd1p` → `9f2c1e4a7b3d` → `c4d7e91a2b60` → `d38b6c05f1a4` (head).

## Ingestion-relevant tables

### `sources` (model `Source`)
`id`, `name`, `source_type` (file|rss|web|obsidian|manual), `path`, `url`,
`content_hash` (Text, **unique** — sha256 dedup fingerprint), `tags` (JSONB), `parent_id` (self-FK,
hierarchy), `meta` (JSONB), `created_at`, `updated_at`.
Indexes: `idx_sources_content_hash` (unique), `idx_sources_parent_id`, `idx_sources_type`.

### `documents` (model `Document`) — legacy chunked docs
`id`, `source_id` (FK→sources CASCADE), `title`, `content` (Text), `content_type`, `chunk_index`,
`chunk_count`, `status`, `embedding` (Vector(768) legacy), **`embedding_vllm_1024` (Vector(1024) active)**,
`fts_vector` (TSVECTOR, trigger), `tags` (JSONB), `meta` (JSONB), timestamps.
Indexes: `idx_documents_content_fts` (gin), `idx_documents_tags` (gin), `idx_documents_source_id`.

### `chunks` (model `Chunk`) — hierarchical semantic nodes
`id`, `document_id` (FK→documents CASCADE), `parent_id` (FK→chunks, SET NULL), `section_path`,
`hierarchy_level`, **`content` (Text — the chunk's own text; offsets alone cannot recover it once
the owning document is replaced)**, `start_offset`, `end_offset`, `prev_chunk_id`, `next_chunk_id`
(FK→chunks), `context_summary`, `embedding_vllm_1024` (Vector(1024)), `embedding_version`,
`meta` (JSONB), timestamps.
Indexes: `idx_chunks_document_id`, `idx_chunks_parent_id`, `idx_chunks_section_path`,
`idx_chunks_embedding_vllm_1024` (HNSW).

### `semantic_units` (model `SemanticUnit`) — atomic claims/facts
`id`, `chunk_id` (FK→chunks CASCADE), `unit_type`, `content`, `polarity` (CHECK -1/0/1),
`valid_from`, `valid_to` (temporal window), `entity_ids` (JSONB), `confidence`,
`span_start`, `span_end`, `embedding_vllm_1024`, `fact_class`, `meta`, `created_at`.
Indexes: `idx_units_chunk_id`, `idx_units_valid_to`, `idx_units_fact_class`, `idx_units_embedding_vllm_1024` (HNSW).

### `entities` (model `Entity`)
`id`, `canonical_name` (unique), `aliases` (JSONB), `entity_type` (default `"concept"`),
`description`, `embedding_vllm_1024`, `resolution_state`, `merge_policy`, `confidence`, `meta`, timestamps.
Indexes: `uq_entities_canonical_name` (unique), `idx_entities_type`, `idx_entities_resolution_state`,
`idx_entities_embedding_vllm_1024` (HNSW).

### `tags` (model `Tag`)
`id`, `name`, `parent_id` (self-FK), `synonyms` (JSONB), `description`, `domain`, `version`, timestamps.
Unique `(name, parent_id)`.

### `knowledge_links` (model `KnowledgeLink`) — typed graph edges
`id`, `source_id`/`target_id` (**no FK — polymorphic node ids named by `source_kind`/`target_kind`;
the source-only FKs were dropped in `9f2c1e4a7b3d`**), `relation_type`, `weight`, `source_kind`
(source|chunk|unit|entity|tag), `target_kind`, `confidence`, `evidence` (JSONB), `evidence_count`,
`reason`, `model_version`, `lifecycle` (active|review|superseded), `valid_from`, `valid_to`, `meta`.
Unique directed edge per `(source_id, target_id, relation_type)`.

### Versioning / state
- `document_versions` (model `DocumentVersion`): `source_id`, `version`, `content_hash`, `diff_summary`,
  `reason`, `meta`; unique `(source_id, version)`. Populated by `IngestService._record_version`.
- `embedding_versions` (model `EmbeddingVersion`): `model_name`, `dimensions`, `status`, `config`;
  unique `(model_name, dimensions)`.
- `maintenance_runs`, `dirty_flags` (partial index `processed_at IS NULL`),
  `model_records`, `feedback_events`, `learn_runs`, `learn_urls` (unique `(run_id, url_norm)`),
  `registry_entries`, `audit_logs`, `scheduled_tasks`, `scheduled_task_runs`, `retrieval_feedback`.

## pgvector usage
- Extensions: `vector`, `pg_trgm`.
- Vector columns (all `Vector(1024)`): `documents.embedding_vllm_1024`, `chunks.embedding_vllm_1024`,
  `semantic_units.embedding_vllm_1024`, `entities.embedding_vllm_1024` (+ legacy `documents.embedding` 768).
- HNSW indexes `vector_cosine_ops` (m=16, ef_construction=200), ivfflat fallback in `init_db`.
- Similarity query (RAG): `1 - (embedding_vllm_1024 <=> :vec)` fused with FTS + trigram (RRF).
- Dedup query (`IngestService.dedup_check`): `1 - (embedding_vllm_1024 <=> CAST(:vec AS vector))`.

## Tables from the spec
- `search_evidence`, `ingestion_jobs`, `freshness_states` — created by `x1f3a2b9cdd1p`.
- `nodes`, `claims`, `relationships`, `concepts`, `knowledge_versions` — **absent by design** (the
  spec's equivalents are `chunks`, `semantic_units`, `knowledge_links`, `entities`/`tags`,
  `document_versions` respectively).

## Adding a migration
New alembic revision with `down_revision` set to the current head, additive-only, mirrored by
idempotent DDL in `init_db()`. See `docs/schema-evolution.md`.