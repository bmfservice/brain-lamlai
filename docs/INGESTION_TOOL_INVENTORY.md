# BrainD Data Ingestion — Tool Inventory (Current State)

> Audit deliverable, Phase 0. Everything the ingestion path can reuse. Paths relative to `braind/`.

## Agent tool registry

Defined in `core/agent/tool.py` (`BaseTool`, `ToolRegistry`); driven by `core/agent/loop.py`.
Four tools are registered at startup in `api/main.py` (lifespan).

### 1. `web_search` — `services/tools/web.py` `WebSearchTool`
- **Purpose:** DuckDuckGo web search (falls back to Bing RSS); optional save-to-brain.
  Also ships the **SSRF-safe** fetch utilities the ingestion upgrade must reuse.
- **Input:** `query` (str, required), `max_results` (int, default 10, capped 15), `save_to_brain`.
- **Output:** `{query, results:[{title,url,snippet}], saved_to_brain}`.
- **Read-only helpers to reuse:**
  - `search(query, max_results=5)` — search, returns results (SSRF-filtered URLs).
  - `fetch_page_text(url, ..., max_chars, max_bytes, ...)` — SSRF-safe page fetch → `{title,text,url,...}`.
  - `extract_candidate_links(...)`.
- **Guard:** `core/net.py` `is_safe_url` / `is_public_host` reject loopback/private/link-local/
  reserved, before the request and again after every redirect.
- **Dependencies:** `httpx`, `bs4`, `core.net`; config `writer_web_fetch_timeout_seconds`,
  `writer_web_max_page_chars` (30000), `writer_web_max_response_bytes` (1500000),
  `outbound_fetch_timeout_seconds`, `outbound_max_response_bytes`.
- **Callers:** registered tool; injected into `WriterService`, `LearningService`;
  `ExternalDuplicateChecker` (`services/writer/duplicate.py`); route `POST /api/tools/search_external`.

### 2. `fetch_rss` — `services/tools/rss.py` `FetchRSSTool`
- **Purpose:** fetch RSS/Atom feed items into brain.
- **Input:** `feed_url` (required), `limit` (default 5).
- **Callers:** registered tool; `scheduler._job_fetch_rss`. `fetch_feed` routes through
  `core.net.fetch_text_bounded`, so feed URLs get the same SSRF/size/timeout guard.

### 3. `fetch_url` — `services/tools/rss.py` `FetchURLTool`
- **Purpose:** fetch + scrape a URL's readable content into brain.
- **Input:** `url`, `limit_chars`. SSRF-guarded since 2026-08-20 — `scrape_url` routes through
  `core.net.fetch_text_bounded`.
- **Callers:** registered tool.

### 4. `knowledge_retrieve` — `services/tools/knowledge.py` `KnowledgeRetrieveTool`
- **Purpose:** RAG lookup; returns brief excerpts.
- **Input:** `query`, `top_k` (≤8).
- **Backing service:** `RAGService.retrieve` (hybrid vector + FTS + trigram, RRF fusion).
- **Callers:** registered tool.

## Reusable extraction services (used by the semantic ingest collaborator)

| Service | File | Capability |
|---|---|---|
| `EntityExtractor` | `services/entity/extractor.py` | `extract(text) -> [EntityMention{text,entity_type,start,end}]`; `chat_json` first, heuristic fallback |
| `EntityResolver` | `services/entity/resolver.py` | `resolve(mention, *, known, ...)` deterministic merge: AUTO_MERGE / DO_NOT_MERGE; `normalize_name` |
| `ClaimExtractor` | `services/extract/claims.py` | `extract(text, *, chunk_id) -> [SemanticUnitDraft]`; LLM-first, deterministic sentence fallback |
| `temporal` | `services/extract/temporal.py` | `extract_temporal_window(text, reference_year=...) -> TemporalWindow` |
| `RelationExtractor` | `services/extract/relations.py` | verb-cue + co-occurrence relation candidates |
| `RelationValidator` | `services/relation/validator.py` | evidence-gated edge acceptance; `VALID_EDGE_TYPES` taxonomy; `RelationCandidate`/`RelationVerdict` |

## Inference primitives (`core/vllm.py`, `VLLMClient`)
- `chat(...)` — OpenAI-compatible chat (local `Qwen/Qwen2.5-7B-Instruct-AWQ` on `:8001`).
- `chat_json(...)` — **JSON-mode chat** (default temp 0.2); the primitive to use for any new
  structured-LLM extraction (freshness, search queries), salvages JSON robustly.
- `embed` / `embed_batch` — `BAAI/bge-m3`, 1024-dim, validated, on `:8002`.
- `generate(...)` — completion endpoint.
- Availability: `is_available`, `is_llm_available`, `is_embedding_available`.

## Duplicate detection that exists
- `IngestService.dedup_check(content, threshold)` — embedding-cosine vs `documents` (the pattern
  the search-validation verdict will reuse).
- Exact sha256 `content_hash` dedup on `Source.path`.
- `services/writer/similarity.py` — pure-Python lexical (shingles, Jaccard, containment, cosine).
- `ExternalDuplicateChecker` (`services/writer/duplicate.py`) — web-level duplicate check using
  `WebSearchTool` + `chat_json` query generation + `vllm.embed` semantic scoring.

## Capabilities that DO NOT exist (gaps)
- **No OCR / image→text.**
- **No general language detector** (only the Chinese-Han prohibition `contains_chinese`).
- **No standalone document classifier** (only heuristic `TextAnalyzer.detect_topic`).
- **No simhash/fingerprint primitive** beyond sha256 + lexical/embedding-duplicate detection.
- **No dedicated `concepts` table** (concepts currently map to `entities.entity_type='concept'` + `tags`).