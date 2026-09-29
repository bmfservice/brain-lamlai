# Braind Architecture

## Purpose

Braind is a local-first AI second-brain server. PostgreSQL with pgvector owns durable knowledge. The local Obsidian vault owns Markdown notes. Two localhost-only vLLM servers provide chat and embeddings.

## Runtime Composition

`api/main.py` is the composition root. Its FastAPI lifespan creates the database engine and shared session factory, creates the vLLM client, then constructs and registers all application services on `app.state`.

```
FastAPI routes -> api/deps.py -> app.state services
                         |
                         +-> core: config, database, vLLM, errors, logging
                         +-> services: domain workflows
```

The deployed dependency chain is:

```
PostgreSQL + pgvector <- vLLM chat :8001 + embed :8002 <- FastAPI :8000
```

## Directory Ownership

| Path | Owner and responsibility |
| --- | --- |
| `api/` | FastAPI routes, request/response schemas, dependency access, static UI. Keep business workflows out of routes. |
| `core/` | Shared infrastructure: settings, database schema/bootstrap, vLLM client, outbound HTTP safety (`net.py`), document parsing (`documents.py`), the ingest quality gate (`quality.py`), the closed classification vocabularies (`taxonomy.py`), chunking, text analysis, logging, typed errors, agent runtime primitives. |
| `services/` | Domain workflows. Each package receives its dependencies explicitly from `api/main.py`. |
| `services/ingest/` | Quality gate, one-round-trip classification (`classify.py`), chunking, embeddings, persistence, deduplication, and knowledge-link creation. |
| `services/rag/` | Retrieval. `RAGService` owns the three raw SQL streams (vector, full-text, trigram); `UnifiedRetrievalService` wraps it with the query analyzer, entity and graph streams, RRF and the feature reranker, and is the single retrieval entry point every consumer receives. |
| `services/learn/` | Two crawlers over one page pipeline: `run_deep_learn` (search-driven, topic research) and `crawl_site` (one website, follows its own links, up to 10 tiers). Both stream SSE progress and write the run/URL audit tables. |
| `services/writer/` | RAG-backed content generation, external similarity checks, and generated-content history. |
| `services/agent/` and `core/agent/` | Tool registry, ReAct loop, and agent-oriented response assembly. |
| `services/scheduler/` | APScheduler jobs for RSS, vault synchronization, digest, and the daily maintenance window. Trigger one on demand with `run_job(name)`. |
| `services/maintenance/` | Duplicate and contradiction scans plus re-embedding of documents with a NULL vector. Driven by the scheduler's `maintenance` job. |
| `services/tools/` | Reusable agent tools for knowledge retrieval, RSS, URLs, and web search. |
| `scripts/` | Explicit operator-only maintenance commands. They are never run by application startup. |
| `config/` | systemd user-unit templates. Keep their model settings aligned with `.env`. |
| `tests/` | Deterministic mocked tests for logic. They must not require live vLLM, filesystem data, or the public internet. |
| `tests/integration/` | SQL-level tests against a real PostgreSQL, isolated in a `braind_test` schema that is created and dropped per session. Skipped automatically when no database is reachable. |
| `vault/` | Stateful Obsidian content. Preserve separately from source-code cleanup. |
| `data/`, `logs/`, `models/` | Intentional runtime locations. Do not bulk-delete them even when empty. |

## Core Workflows

- **Ingest:** route or tool -> `IngestService` -> quality gate (`core.quality.assess`: strip site chrome, reject binary, snippets, menus and Han ideographs) -> classification (one LLM round-trip: doc_type, domain, audience, language, summary, keywords, claims) -> dedup -> chunking -> vLLM embeddings -> PostgreSQL sources/documents/links. Rejected content is logged with its reason and never stored.
- **Chunking:** the retrieval-facing `documents` rows are cut by the structure-aware `HierarchicalChunker` (heading/section/paragraph/list/table), not fixed character windows (§1/§2). It runs with `compact=True, prune=False`: coalescing merges the one-block-per-table-cell fragments that Office and PDF extraction emit, while the lossy prune half is skipped so no prose is dropped from a row retrieval must still return.
- **Topic-shift segmentation:** a prose node longer than `segment_trigger_chars` has no internal heading to cut on, so `services/ingest/segment.py::TopicShiftSegmenter` asks the chat model which sentences start a new topic (§2) — never an embedding-cosine threshold. The model only proposes indices into a sentence list; out-of-range, duplicate, non-integer and too-thin boundaries are discarded, so every stored chunk is a contiguous run of the original sentences. Lists and tables are never segmented (their rows are the unit). If the model is unreachable, returns nothing usable, or the node exceeds `segment_max_sentences` (a 7B prompt on one 12GB GPU), the node is packed at sentence boundaries instead.
- **Chunk provenance:** each row records how its boundary was decided in `meta.chunk_kind` — `structure` (the document's own structure), `topic_shift` (LLM boundary), `sentence_pack` (segmenter degraded), `fallback` (whole-document fixed-size net, event `ingest.fixedsize_fallback`, reached only when the structure pass yields no prose node) — plus `meta.section_path`. Every ingest logs `ingest.chunked` with `characters_read`/`characters_chunked`/`coverage`/`segmented_nodes` (§3), and the API `IngestResponse` returns `characters_read`.
- **Entity resolution (§5, three layers):** `services/entity/blocking.py` first narrows the comparison set to entities sharing a normalized token or acronym key with something the document mentions (recall-first: a candidate the blocker drops can never be merged). `services/entity/resolver.py` then decides identity from orthographic evidence only — exact/alias/suffix/subsumption — with knowledge-graph neighbours (`_entity_neighbors`, evidence-gated edges) as an independent *structural* signal that can raise confidence or queue a review but can never manufacture a merge. A name that normalizes to nothing ("platform", "BINARY") has no identity key and merges onto nothing. What still needs a human is not dropped: `canonical_name` is globally unique, so a same-name-different-domain mention cannot become a second row — the collision is recorded as a `lifecycle="review"` `POSSIBLE_RELATION` self-edge on the existing entity carrying both domain assertions in `meta`, and `review_clusters` groups normalization-collision rows for one human decision. The report exposes both backlogs: `/api/ingest/report` → `needs_review_count` (entities never confirmed) and `resolution_conflict_count` (specific pairs awaiting a ruling).
- **Classify:** `core/taxonomy.py` defines every allowed label. Model output is coerced onto it — an unrecognised value becomes the default rather than a new de-facto category. Quality tier and score come from the gate's measurements, never from the model's self-assessment.
- **Retrieve:** route, agent, or writer -> `UnifiedRetrievalService` -> query analysis -> retrieval preferences (`services/rag/preferences.py`: which passage role and document type answer this shape of question) -> vector + FTS + trigram + entity + graph candidates -> weighted RRF -> feature rerank including `role_match` and `source_authority` -> source chunks. The reranker is the heuristic `FeatureReranker`, not a cross-encoder — an accepted trade-off to keep the 12GB VRAM budget for chat + embedding vLLM (`bge-reranker-v2-m3` is deferred to a future phase with a co-residency benchmark).
- **Learn (topic):** route -> `LearningService.run_deep_learn` -> web search -> SSRF-safe fetch -> relevance/SEO analysis -> ingest -> run and URL audit tables.
- **Learn (site):** route -> `LearningService.crawl_site` -> SSRF-safe fetch of the start URL -> breadth-first over links that stay on the same site -> ingest each readable page. Bounded by `crawl_max_pages` and `crawl_hard_timeout_sec`, not by depth alone.
- **Upload:** route -> `core.documents.extract_text` -> `IngestService` (same path as pasted text).
- **Write:** route -> `WriterService` -> retrieve knowledge -> local vLLM generation -> generated-content record -> optional web similarity and ingest.
- **Agent:** route -> `AgentService` -> pre-retrieval from `UnifiedRetrievalService` -> `AgentLoop` -> registered tools -> answer and trace.
- **Maintain:** scheduler `maintenance` job -> `MaintenanceService` -> duplicate/contradiction report -> re-embed NULL vectors.
- **Obsidian display layer (§3):** `services/obsidian/` renders the committed semantic layer into `vault/<obsidian_export_dirname>/` — one note per entity, one note plus a JSON `.canvas` per source. `render.py` is pure (no DB, no I/O) so the exact bytes are unit-testable; `exporter.py` owns the reads and writes atomically (temp file + `os.replace`), skipping files whose content is unchanged so Obsidian's watcher and git stay quiet. It runs automatically after each semantic pass and rebuilds wholesale via `POST /api/ingest/obsidian-export`. Filenames use only keys that survive re-ingest (`entities.normalized_name`, `sources.path`) because document, chunk and unit UUIDs are all regenerated by every pass. Canvas edges are co-mention with the typed `knowledge_links` overlaid and labelled — typed edges alone would draw an almost empty graph. Unconfirmed entities export too, tagged `braind/review` so the backlog is filterable rather than hidden, and negated units (`polarity = -1`) render an explicit **NOT** so a claim never loses its sign in the one view a human actually reads. Export is one-way: nothing reads these notes back into the database.

## Dependency Rules

1. Routes call services through `api/deps.py`; services do not import routes.
2. `core/` has no dependency on `api/` or service workflows.
3. Services may use `core/` and other services only when the dependency is passed in explicitly from the composition root.
4. Database access belongs in services or `core/database.py`, not in UI code.
5. Extend a feature within its existing service package before introducing a new package.

## Data and Safety Invariants

- The active embedding column is `documents.embedding_vllm_1024`; its dimension must remain 1024.
- Content with Han ideographs in U+3400-U+4DBF or U+4E00-U+9FFF is rejected before ingest or web-learning persistence.
- Generated writer content is excluded from ordinary RAG retrieval by default to avoid recursive pollution.
- The vault is the *input* to ingestion, so the Obsidian display layer's own output must never be scanned back in. Two independent guards close that loop: `sync_vault` skips `settings.obsidian_export_dirname` by path part, and `_ingest_file` skips any file carrying `braind_generated: true` — the second check runs *before* the skip-hash is recorded, so a generated note moved into a user folder is re-checked every sync instead of being remembered as "seen". Both guards are structural, because the skip-hash cache is in-memory and a restart re-scans the whole vault.
- The exporter deletes only files that are inside the export directory *and* carry the generated marker. An unexpected leftover file costs a tidy-up; deleting something a person wrote costs their work.
- The semantic layer is derived data owned by the semantic pass. Vault re-sync deletes only the flat chunk rows; the semantic `documents` row (and the chunks, units, and entity edges cascading from it) is replaced inside the semantic pass's own transaction, so a failed pass leaves the previous layer intact.
- `chunks.content` stores the chunk's own text. Offsets alone cannot recover it once the owning document is replaced.
- Classification lives in columns (`doc_type`, `domain`, `audience`, `quality_tier`, `quality_score`, `language`, `chunk_role`), not in JSONB `meta`. Retrieval filters and ranks on them, which a JSONB key cannot index per-value.
- Entity names pass through `services/entity/normalize.py` before storage: whitespace folded, stopwords and fragments rejected, type coerced onto the closed vocabulary, `normalized_name` set so case variants collapse. Every entity carries a description and an embedding.
- Entities are never resolved on embedding cosine alone (§13): identity comes from orthographic rules plus graph-structure corroboration, blocking decides who is comparable at all, and anything uncertain is recorded as a review edge rather than merged or forked into a duplicate. The classifier's `domain` flows into every entity row so same-name-different-domain stays a review, not a silent merge.
- `semantic_units.entity_ids` is populated by matching claim text against the entities resolved for that document. Without it the unit→entity join is empty and graph expansion has no data.
- All outbound HTTP goes through `core.net.fetch_text_bounded`, which enforces the SSRF host check before the request and after every redirect, plus content-type, size and timeout limits. Never issue a raw `httpx` request to a user- or model-supplied URL.
- There is no API authentication. The default bind is `127.0.0.1` and CORS is a loopback whitelist; exposing the API requires an authenticating proxy in front.
- Database schema bootstrap in `core/database.py` is additive operational code. Schema changes require a migration plan, database backup, and tests.
