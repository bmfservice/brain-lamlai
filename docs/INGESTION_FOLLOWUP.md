# INGESTION_FOLLOWUP.md — Out-of-scope items (recorded, not implemented)

> Per the ingestion-upgrade scope rule: improvements required **elsewhere** (or in ingestion but
> beyond this delivery) are recorded here with issue / reason / affected subsystem / proposed
> solution / risk. Do not implement without a separate decision.

## 1. OCR + native PDF parsing
- **Issue:** Uploaded scanned PDFs/images and native PDF text cannot be ingested (only `.md/.txt/.docx`).
- **Reason:** Requires Poppler/PyMuPDF/pdfplumber (+ possibly OCR models); the ingestion upgrade is
  text-path only.
- **Affected subsystem:** `api/routes_ingest.py` `_ALLOWED_TEXT_EXTENSIONS` + a new parser service.
- **Proposed solution:** Add parsers behind the existing `IngestService.ingest_text` (parse → feed the
  same pipeline); OCR as an optional plugin gated by explicit enablement.
- **Risk:** Dependency/GPU cost; security (macros, embedded objects) needs validation.

## 2. Dedicated background/message queue for ingestion
- **Issue:** Long ingest jobs run inline in the request and can block the API for large files.
- **Reason:** Current pipeline is in-process sequential; a queue is an architecture change.
- **Affected subsystem:** `services/ingest/service.py`, `api/routes_ingest.py`, `api/main.py` lifespan.
- **Proposed solution:** ARQ/Taskiq/Celery worker with retries and per-stage checkpointing once
  `ingestion_jobs` stage rows exist.
- **Risk:** New runtime dependency; operational complexity on this self-hosted box.

## 3. Standalone `concepts` table
- **Issue:** The spec asks for concepts distinct from keywords; we represent concepts as
  `entities` with `entity_type='concept'` + existing `tags` machinery.
- **Reason:** Avoids a parallel taxonomy and a new retrieval surface; the entity/tag graph already
  supports linking.
- **Affected subsystem:** `core/database.py`, retrieval (`services/rag/pipeline.py`).
- **Proposed solution:** If retrieval demonstrates a need, add `concepts` with its own edges/browse.
- **Risk:** Schema/retrieval churn; low value before retrieval pressure.

## 4. Full incremental reprocessing engine
- **Issue:** Re-ingest of a changed file reprocesses everything; only content-fingerprint + unit
  `(span_start, content)` skips exist.
- **Reason:** Field-level diffing + dirty-flag cascades for all node types is a large subsystem.
- **Affected subsystem:** `services/ingest/semantic.py`, `services/maintenance/`.
- **Proposed solution:** Build on `dirty_flags` + `document_versions.diff_summary` when a retrieval
  need emerges.
- **Risk:** Complex correctness surface; silent partial updates if wrong.

## 5. Periodic (re-)search validation scheduler
- **Issue:** Freshness/search validation runs at ingest only; existing TIME_SENSITIVE nodes are not
  re-validated on a schedule.
- **Reason:** A re-validation cron is separate scope from the ingest path.
- **Affected subsystem:** `services/scheduler/service.py`, `services/ingest/search_validate.py`.
- **Proposed solution:** Add a periodic job that re-runs `SearchValidator` over `freshness=TIME_SENSITIVE`
  units, reusing `search_evidence` rows.
- **Risk:** Network cost; needs the outbound cap + dedup on `(chunk_id, search_query)`.

## 6. Freshness LLM/hybrid as default
- **Issue:** `freshness_method` ships heuristic by default; LLM per-node is slower/costlier.
- **Reason:** Heuristic is deterministic and safe for the 5-min vault sync cadence.
- **Affected subsystem:** `services/ingest/freshness.py`, `core/config.py`.
- **Proposed solution:** Flip `freshness_method` to `llm`/`hybrid` after a quality benchmark.
- **Risk:** Latency + vLLM load on every ingest.

## 7. Cross-system (non-ingestion) issues noticed during audit
- **Issue:** `FetchURLTool`/`scrape_url`/`fetch_feed` were not SSRF-guarded unlike `WebSearchTool`.
- **Affected subsystem:** `services/tools/rss.py`, `services/scheduler/rss.py`.
- **Resolved 2026-08-20:** every outbound fetch now shares `core/net.py:fetch_text_bounded`, which
  applies the host check before the request and after each redirect plus content-type, byte-cap and
  timeout limits. Covered by `tests/test_web_tool.py` and `tests/test_web_safety.py`.