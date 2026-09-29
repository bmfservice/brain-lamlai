# BrainD Data Ingestion — Architecture (Current State)

> Audit deliverable, Phase 0 of the ingestion upgrade. Describes what exists today,
> read from source (not guessed). All paths relative to `braind/`.

## 1. Overview

BrainD ingests text, uploaded files, and its Obsidian vault into PostgreSQL + pgvector.
There are **two parallel pipelines**, both composed in `api/main.py` and sharing the
same public entry points:

1. **Legacy flat pipeline** — `services/ingest/service.py` `IngestService` → `sources` + `documents`.
2. **"Brain Second" semantic pipeline** — `services/ingest/semantic.py` `SemanticIngestService`
   → `chunks`, `entities`, `semantic_units`, typed `knowledge_links`.

The semantic pipeline runs **after** the legacy commit as a best-effort collaborator, so a
semantic failure never rolls back the stored source.

## 2. Wire-in (`api/main.py`)

```python
semantic = SemanticIngestService(vllm, settings.chunk_size)   # line 79
ingest   = IngestService(session_factory, vllm, semantic=semantic)  # line 80
web_tool = WebSearchTool(ingest)                               # line 91
```

`IngestService` is registered on `app.state.ingest`; routes consume it via
`api/deps.py get_ingest`.

## 3. Public API entry points (`api/routes_ingest.py`, prefix `/api/ingest`)

| Route | Handler | Path |
|---|---|---|
| `POST /api/ingest/text` | `ingest_text` | `IngestService.ingest_text` |
| `POST /api/ingest/file` | `ingest_file` | docx/text extraction, then `ingest_text` |
| `POST /api/ingest/vault` | `sync_vault` | full Obsidian resync |
| `GET  /api/ingest/stats` | `get_stats` | source/chunk counts + recent |

### Format support (files)
- Plain text extensions: `.md .txt .json .csv .log .py .js .ts .html .css` → `utf-8` decode.
- `.docx` (+ DOCX MIME): `_extract_docx_text` (zipfile + ElementTree over `word/document.xml`).
- Extensionless non-binary → text. Otherwise rejected (`"This file format is not supported."`).
- **No OCR, no PDF parsing.**

## 4. `IngestService.ingest_text` — full chain

```
content.trim
 ├─ len < 120            → skip (skip_short)
 ├─ contains_chinese()   → skip (skip_chinese)          [absolute Han prohibition]
 ├─ dedup_check(0.92)    → embedding cosine vs documents → skip if match
 ├─ sha256 content_hash  → skip if Source.path == hash   [exact dup]
 ├─ TextAnalyzer: summarize / extract_keywords / detect_topic   [LLM]
 ├─ insert Source(source_type, path=content_hash, meta{summary,keywords,auto_topic,...})
 ├─ TextChunker.chunk(content) → list[str]              [chunk_size 800 / overlap 120]
 ├─ for each chunk: vllm.embed(chunk) → insert Document(
 │      chunk_index, chunk_count, embedding_vllm_1024, tags, meta)
 ├─ _auto_link → knowledge_links edges (shared keywords, upsert)
 ├─ commit
 └─ SemanticIngestService.semantic_pass(...)            [best-effort]
```

## 5. Vault sync (`sync_vault` / `_ingest_file`)

- Walks `vault_dir.rglob('*.md')`, skipping `.obsidian`.
- Frontmatter parsed (`title`, `tags`, `type`, `parent`).
- Dedup by sha256; unchanged files skipped via in-memory `_file_hashes`.
- `Source` keyed by absolute file path in `Source.path`.
- **Delete-and-reinsert** child `documents` per source (idempotence by deletion).
- `_resolve_parent`: `parent:` / `parent_note:` frontmatter or `[[wikilink]]` → `source.parent_id`
  + a `knowledge_links` edge `relation_type="parent_child"`.
- After commit, runs the semantic collaborator.

## 6. Semantic collaborator (`SemanticIngestService.semantic_pass`)

1. Writes an owning `documents` row (content = whole text).
2. `HierarchicalChunker.chunk` → write `chunks` (per node: parent_id, prev/next, section_path,
   hierarchy_level, offsets, context_summary, embedding).
3. `EntityExtractor` + `EntityResolver` → `entities` (merge decisions honored, alias additions).
4. `RelationValidator` + `RelationExtractor` → typed, evidence-gated `knowledge_links` edges.
5. `ClaimExtractor` + `extract_temporal_window` → `semantic_units` (atomic claims, spans,
   fact_class, polarity, valid_from/valid_to, embedding).

Each step wrapped independently; a failure records the step and returns stats,
never rolling back earlier committed work.

## 7. Chunking (`core/chunking.py`)

- `TextChunker` (legacy): paragraph blocks (`\n\n`), token budget, overlap, sentence hard-split.
- `HierarchicalChunker` (semantic): heading→section parse, section_path from heading stack,
  paragraph nodes under open section, sentence sub-split only when over budget,
  prev/next + parent links, context_summary.

## 8. Embedding + persistence

- `VLLMClient` (`core/vllm.py`): `embed` / `embed_batch` → `BAAI/bge-m3`, **1024-dim**, at `:8002`.
- Active vector column across tables: `embedding_vllm_1024` (`Vector(1024)`).
  - `documents`, `chunks`, `semantic_units`, `entities`.
- HNSW indexes `vector_cosine_ops` (m=16, ef_construction=200) created in `init_db`.
- RAG retrieval is raw-`text()` pgvector cosine + FTS + trigram (parameterized tag and
  source-type filters) fused by Reciprocal Rank Fusion in `services/rag/service.py`;
  `services/rag/pipeline.py` adds the analyzer, entity and graph streams plus the feature
  reranker and is the single entry point every consumer receives.

## 9. Background processing

- APScheduler jobs (`services/scheduler/service.py`): `rss-fetch` (`RSS_REFRESH_MINUTES`,
  default 60 min), `vault-sync` (5 min), `daily-digest` (cron `DIGEST_HOUR`:03), `maintenance`
  (cron `MAINTENANCE_TIME`). No task backlog; ingestion is in-process sequential.
- Startup vault sync in `main.py` lifespan (`asyncio.wait_for(..., timeout=300)`).
- Trigger one job on demand with `BraindScheduler.run_job(name)` /
  `POST /api/scheduler/run`.
- No dedicated job/retry queue for ingestion. `ingestion_jobs` records stage outcomes for
  observability; `POST /api/ingest/{job_id}/retry` only re-marks a stage as queued — nothing
  consumes that flag yet.

## 10. Metadata stored today

- `Source.meta`: `summary`, `keywords`, `auto_topic`, user `meta`, `links` (after auto-link);
  vault adds `relative_path` + frontmatter.
- `Document.meta`: `{"source_type": ...}` merged with the above.
- `Chunk.content`: the chunk's own text (added 2026-08-20; offsets alone could not recover it
  once the owning document was replaced).
- `Chunk.meta`: `{"chunk_kind": ..., "detected_language": ..., "language_confidence": ...}`.
- `semantic_units`: spans, polarity, validity window, `fact_class`, `entity_ids`, confidence.

## 11. Background / jobs and state

- `document_versions` is populated by `IngestService._record_version` on every ingest.
- `maintenance_runs`, `dirty_flags`, `embedding_versions`, `model_records` exist
  (schema-level) but the ingest path does not populate them.