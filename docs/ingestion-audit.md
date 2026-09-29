# Ingestion & Knowledge Architecture — Phase 0 Audit

> Deliverable for Phase 0 (Discovery) of the ingestion-upgrade CLAUDE.md.
> **No source code was changed in this phase** — this is a read-from-source audit.
> All paths relative to `braind/`. Evidence commands and their real output are
> inline. Date: 2026-08-22.

---

## A. Current architecture (as it actually runs)

### A.1 Two parallel ingest pipelines, plus a third half-wired one

There are **three** chunking code paths in the tree. Two are live; one is dormant.

| Path | Entry | Writes | Chunking unit | Status |
|---|---|---|---|---|
| **V1 flat** | `services/ingest/service.py::_ingest_text_v1` (line 200) | `sources` + `documents` | `core/chunking.py::TextChunker` — fixed 800 tok / 120 overlap | **LIVE** (default for every API call — see A.4) |
| **Semantic collaborator** | `services/ingest/semantic.py::SemanticIngestService.semantic_pass` | `chunks`, `entities`, `semantic_units`, typed `knowledge_links` | `core/chunking.py::HierarchicalChunker` — heading→section structure-aware | **LIVE** (runs after V1 commit, best-effort) |
| **Ingest V2 adaptive** | `services/ingest_v2/service.py::IngestV2Service` via `services/ingest/service.py::_ingest_text_v2` (line 537) | `sources` + hierarchical `documents` nodes only | `AdaptiveStrategySelector` (hierarchical / semantic-cosine) | **DORMANT** — never invoked in practice (see A.4); also has a runtime bug (B.1) |

Composition root: `api/main.py` lines 113–123. `IngestService` is built with
`semantic=SemanticIngestService(...)` and `use_ingest_v2=getattr(settings,'use_ingest_v2',False)`.

### A.2 What the LIVE path does per document (V1 + semantic collaborator)

`_ingest_text_v1` chain (verified against source):

```
quality.assess(content)              # core/quality.py gate; strips chrome, rejects Han, snippets, binary
 → dedup_check(0.92)                 # embedding cosine vs existing documents
 → sha256 content_hash               # exact-dup guard (Source.path == hash)
 → DocumentClassifier (1 LLM call)   # doc_type/domain/audience/quality_tier/language + summary + keywords
 → INSERT Source (classification denormalised onto columns)
 → TextChunker.chunk(content)        # FIXED 800/120  ← the core problem, see C
 → per chunk: vllm.embed → INSERT Document(embedding_vllm_1024, chunk_role, ...)
 → _auto_link (shared-keyword knowledge_links edges)
 → commit
 → _run_semantic_pass(...)           # best-effort; never rolls back the commit
```

`semantic_pass` (the structure-aware half) then writes the real knowledge layer:
`HierarchicalChunker` → `chunks` (parent/prev/next, `section_path`, offsets,
`context_summary`, embedding) → `EntityExtractor`+`EntityResolver` → `entities`
→ `RelationValidator`/`RelationExtractor` → typed `knowledge_links` →
`ClaimExtractor`+temporal → `semantic_units` (polarity, validity window,
`fact_class`).

**Live evidence** — a real ingest through the running API (`POST /api/ingest/text`,
"CUDA Notes", 53 words) produced:

```
source meta: doc_type=explainer, domain="gpu computing", audience=technical,
             key_claims=[3 claims], keywords=[6]      ← V1 classification path
documents for source: 2 rows, content_type=markdown
chunks for source (semantic layer): 1
semantic_units for source: 3
```

So the live path **does** build a semantic layer — but the flat `documents` rows
that retrieval actually ranks on are cut by the **fixed-size** `TextChunker`.

### A.3 Storage model (PostgreSQL + pgvector, `core/database.py`)

- `sources` — raw source + denormalised classification (`doc_type`, `domain`,
  `audience`, `quality_tier`, `quality_score`, `language`), `content_hash`, `meta` JSONB.
- `documents` — **flat chunk rows** (`chunk_index`, `chunk_count`, `content`,
  `embedding_vllm_1024` Vector(1024), FTS `fts_vector`, `doc_type`/`domain`/`chunk_role`/`quality_*`).
  Legacy `embedding` Vector(768) still present, unused by active retrieval.
- `chunks` — **structure-aware** semantic chunks (`parent_id`, `prev/next_chunk_id`,
  `section_path`, `hierarchy_level`, `start/end_offset`, `context_summary`,
  `chunk_role`, `root_id`, `page_number`, `source_location`, `content_fingerprint`).
- `semantic_units` — atomic claims (`unit_type`, `polarity` ∈ {-1,0,1}, `valid_from/to`,
  `entity_ids`, `fact_class` ∈ SOURCE_FACT/INFERRED_FACT/MODEL_GENERATED/USER_PROVIDED/SYSTEM_DERIVED/UNCERTAIN).
- `entities` — `canonical_name` (unique), `normalized_name`, `aliases`, `entity_type`,
  `resolution_state` (resolved/manual/ambiguous/review), `merge_policy` (auto/review/do_not_merge).
- `knowledge_links` — polymorphic typed edges (`source_kind`/`target_kind` ∈
  source|chunk|unit|entity|tag), `relation_type`, `confidence`, `evidence` JSONB,
  `evidence_count`, `lifecycle`, validity window. **No edge without evidence** (enforced in writer).

**Live counts (whole DB):** `chunks=414`, `semantic_units=2560`, `entities=2409`,
`knowledge_links` = 397 source↔source + 18 entity↔entity.

### A.4 Retrieval (`services/rag/`)

`UnifiedRetrievalService` (`services/rag/pipeline.py`) is the single entry point —
`api/deps.py::get_unified_retrieval`, consumed by chat, search, learn, writer,
marketing, tools (verified: 12 consumers, none touch bare `RAGService`).

Streams: pgvector cosine + Postgres FTS `tsvector` + `pg_trgm` + entity + bounded
graph traversal (`services/graph/service.py`, hard-capped by `graph_max_*`),
fused by **Reciprocal Rank Fusion** with per-query-type weights, then a
**`FeatureReranker`** (heuristic: term overlap, entity match, recency, source
trust, graph distance, role match — **not** a cross-encoder model).

### A.5 Document parsing (`core/documents.py`)

Real parsers already present: PDF (`pypdf`), DOCX/PPTX/ODT/EPUB (zip+XML),
XLSX (`openpyxl`), plain text/code. **No OCR** for scanned PDFs. No Docling/MarkItDown.

### A.6 Vault / Canvas

`vault/` has 10 `.md` files, **zero `.canvas` files**. Vault sync
(`sync_vault`/`_ingest_file`, line ~1135) walks `*.md`, parses frontmatter,
dedups by sha256, delete-and-reinserts child `documents`, resolves `parent:`/
`[[wikilink]]` into a `parent_child` edge, then runs the semantic pass. There is
**no Canvas generation and no reverse-sync** (Obsidian edit → DB) today.

### A.7 Serving stack (matches CLAUDE.md §0)

vLLM chat `Qwen2.5-7B-Instruct-AWQ` @`:8001`; vLLM embedding `BGE-M3` 1024-dim
@`:8002`; FastAPI @`:8000`. `pip list` confirms: `vllm 0.27.1`, `pgvector 0.5.0`,
`networkx 3.6.1`, `sentencepiece`. **Absent:** docling, markitdown, kuzu,
sqlite-vec, llama-index, FlagEmbedding, redis client, cross-encoder reranker.

---

## B. Bugs found while auditing (not yet fixed — Phase 0 is read-only)

### B.1 `IngestV2Service` crashes whenever `enrich=True`

`services/ingest_v2/service.py` lines 117/167/234 gate enrichment on
`config.use_ai_enhancement`, but `AdaptiveChunkConfig` has **no such attribute**
(verified: `hasattr(AdaptiveChunkConfig(), 'use_ai_enhancement') == False`; the
real flag is `use_semantic_split` / the per-feature booleans). Any call with
`enrich=True` raises `AttributeError`. This is one reason the V2 path is unused.

### B.2 `IngestV2` produces zero chunks for short content

`tests/test_ingest_v2_service.py::TestEdgeCases::test_very_short_content` fails:
`ingest_text("Hello world.")` → `len(result.chunks) == 0`. Short docs fall through
the adaptive selector with no fallback node, violating CLAUDE.md §1 (fixed-size
must be the *safety net* — here there is no net at all).

### B.3 Broader `ingest_v2` test rot

`pytest -k "chunk or ingest or semantic"` → **13 failed, 132 passed, 22 errors**.
All failures/errors are inside `tests/test_ingest_v2_*` (analyzers, enrichers,
graph, service). The **live** path is green: `test_ingest_semantic.py`,
`test_hierarchical_chunking.py`, `test_entity_resolution.py`,
`test_graph_service.py`, `test_semantic_models.py` → **80 passed**. So the rot is
confined to the dormant V2 subsystem.

---

## C. Gap analysis vs the ingestion CLAUDE.md

| # | Requirement | Current state | Gap |
|---|---|---|---|
| §1 | No fixed-size as primary unit; fixed only as logged fallback | **LIVE `documents` rows are fixed 800/120** via `TextChunker`. `HierarchicalChunker` (structure-aware) writes only the parallel `chunks` table. | **HIGH.** The rows retrieval ranks on (`documents`) are fixed-size. No "fallback used" flag is logged. |
| §2 | Structure-aware first; LLM topic-shift for unstructured; **not** embedding-similarity cuts | `HierarchicalChunker` is structure-aware ✓. But V2's `SemanticChunker` cuts on cosine `semantic_threshold` (line 4/125), and no LLM topic-shift segmenter exists for unstructured docs. | **MEDIUM.** Missing LLM-assisted boundary detection; existing semantic cut is cosine-only (explicitly discouraged). |
| §3 | Read whole content; report `characters_read` | No `characters_read`/`character_count` field anywhere (grep: none in ingest/chunking/ingest_v2). `word_count` exists in quality report. | **MEDIUM.** Ingestion report lacks the mandated char-count reconciliation. |
| §4 | Read whole content; **raw source preserved** separate from derived | `sources` keeps raw; `document_versions` snapshots per ingest ✓. | **LOW.** Largely satisfied. |
| §4-schema | `node_id/document_id/parent_node_id/path/node_type/confidence/source_location`; unified DOCUMENT→SECTION→UNIT | `chunks` has parent/section_path/offsets/source_location ✓; but **`path` is `section_path` on `chunks` only**, absent on the `documents` rows retrieval uses. No single `node_type` enum spanning SECTION vs UNIT. | **MEDIUM.** Schema close but split across two tables (`documents` flat vs `chunks` hierarchical). |
| §5 | 5-signal evidence provenance | `semantic_units.fact_class` has 6 classes incl. SOURCE_FACT/INFERRED_FACT/MODEL_GENERATED/USER_PROVIDED ✓. | **LOW.** Present and enforced. |
| §5-ER | 3-layer entity resolution (blocking / multi-signal scoring / clustering), no single cosine | `EntityResolver` is deterministic multi-rule (exact/alias/suffix/subsumption/ambiguity), never cosine-only ✓; `merge_policy=review` for uncertain ✓. But **no explicit blocking stage** and structural (shared-relationship) evidence isn't part of scoring yet. | **MEDIUM.** Rules good; missing blocking + graph-structural signal + `NEEDS_REVIEW` count in report. |
| §3-Obsidian | Each unit → `.md` with frontmatter; edges as `[[wikilink]]`+`relationships:`; per-doc `.canvas`; reverse-sync | **Zero `.canvas` files; no unit-per-file export; no reverse-sync.** Vault is input-only + parent wikilink. | **HIGH.** Entire Obsidian display/Canvas layer unbuilt. |
| Retrieval | hybrid semantic+FTS+metadata+graph+**bge-reranker-v2-m3** | Hybrid + RRF + graph ✓, but reranker is a **heuristic `FeatureReranker`**, not the cross-encoder bge-reranker. | **MEDIUM.** Works, but not the specified model; VRAM budget in §0 assumes a reranker model that isn't loaded. |
| §0-VRAM | reranker model loaded within 12GB budget | Only chat + embedding vLLM run; no reranker model process. | **LOW/DECISION.** Frees VRAM, but diverges from the stated stack. |
| Parsers | Docling/MarkItDown structure-aware | `core/documents.py` uses pypdf/openpyxl/zip-XML → **plain text**, losing heading/table structure that structure-aware chunking depends on. No OCR. | **MEDIUM.** Structure is re-detected from text, not preserved from the parser. |
| Graph store | Kuzu embedded (spec suggestion) | Graph lives in `knowledge_links` rows in Postgres, bounded traversal in `GraphService`. | **DECISION, not gap.** CLAUDE.md §2 says *do not rewrite* an existing Postgres/graph store; the Postgres-native graph is a legitimate choice to keep. |

---

## D. Proposed architecture (incremental, respects §2 "don't rewrite")

The live semantic layer (`chunks`/`entities`/`semantic_units`/`knowledge_links`)
is already close to the spec and well-tested. The right move is **converge on the
structure-aware path and retire the fixed-size primary**, not a rewrite.

1. **Make structure-aware the primary, fixed-size the logged fallback (§1/§2).**
   Stop cutting the retrieval-facing rows with `TextChunker`. Either (a) point
   retrieval at the `chunks` table, or (b) have the live path populate `documents`
   from `HierarchicalChunker` nodes, tagging each node `chunk_kind=structure|fallback`
   and logging when the fixed-size net is used.
2. **Add an LLM topic-shift segmenter for unstructured docs (§2)** (transcripts,
   OCR, raw email) — prefer Claude API for boundary calls per §0; never cosine-only.
3. **Add `characters_read` to the ingestion report (§3)** and reconcile against
   raw source length; surface it in the MUST-SHOW-EVIDENCE report.
4. **Decide the fate of `ingest_v2`.** It is dormant, buggy (B.1/B.2), and
   duplicates the live `HierarchicalChunker`. Options: fix+adopt, or delete and
   fold its useful analyzers (table/code detection) into the live path. Recommend
   **fold-and-delete** to restore a single source of truth.
5. **Entity resolution §5**: add a blocking stage (by `entity_type`+`domain`+lang),
   add graph-structural evidence to scoring, and emit `NEEDS_REVIEW` counts.
6. **Obsidian display layer §3**: export each `entity`/`semantic_unit`/section to a
   `.md` with the §4 frontmatter, edges as `relationships:` + `[[wikilink]]`, and a
   per-document `.canvas`; design reverse-sync as watch→re-validate→update (never
   blind overwrite).
7. **Reranker decision**: either load `bge-reranker-v2-m3` within the §0 VRAM
   budget, or explicitly amend CLAUDE.md to record `FeatureReranker` as the chosen
   trade-off (§7 "state limits, don't hide them").
8. **Parsers**: evaluate Docling/MarkItDown to *preserve* heading/table structure
   from PDF/DOCX rather than re-detecting from flattened text.

---

## E. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Re-pointing retrieval from `documents`→`chunks` changes what every consumer sees | HIGH — chat/search/writer output shifts | Phase behind a flag; keep `documents` populated during transition; golden-query regression set before cutover |
| Deleting `ingest_v2` removes code some route imports (`/ingest/v2`, `api/routes_ingest_v2.py`) | MEDIUM | Grep-confirmed consumers: `routes_ingest_v2.py`, `ingest_v2_adapter.py`, `service.py`, `config.py`. Remove route + adapter together; the live path never depends on it |
| Non-additive schema change (unifying `documents`/`chunks`) | HIGH — data migration on 414 chunks / 2560 units | Follow `docs/schema-evolution.md`; DB backup first; additive columns over destructive DDL |
| LLM topic-shift segmentation is slow on 1×12GB for large docs | MEDIUM (§0/§7) | Route large-doc boundary calls to Claude API; cap local-model segmentation size; log fallback |
| Obsidian reverse-sync overwrites derived data or loops with vault-sync | MEDIUM | Watch→validate→update only; never auto-overwrite; guard against re-ingest of exported files |
| bge-reranker + chat-7B-AWQ + BGE-M3 co-resident OOM on 12GB | MEDIUM (§0) | Enforce `--gpu-memory-utilization 0.5–0.6` on chat vLLM; benchmark before enabling reranker; or keep heuristic reranker |
| Han-ideograph reject rule must survive any new ingest path | HIGH (CLAUDE.md hard rule) | Every new path calls `core/quality.py::assess` + `sanitize_model_text`; add a test |

---

## F. Evidence appendix (real command output)

- **Live path proof**: `POST /api/ingest/text` "CUDA Notes" → `added=1`; DB shows
  `documents=2`, `chunks=1`, `semantic_units=3` for that source; source `meta`
  carries V1 classification (`doc_type=explainer`, 3 `key_claims`).
- **V2 dormant**: route `use_v2` default `False` (`api/routes_ingest.py:39`)
  overrides the service's `use_ingest_v2=True`, so no API call reaches V2.
- **V2 broken**: `hasattr(AdaptiveChunkConfig(),'use_ai_enhancement')==False`;
  `test_very_short_content` → 0 chunks.
- **Test split**: live semantic suite **80 passed**; `ingest_v2` suite
  **13 failed + 22 errors**.
- **Fixed-size sites**: `core/config.py:47-48` (800/120), `core/chunking.py:55`,
  `services/ingest/service.py:131`, `services/ingest_v2/config.py` (min/target/max).
- **Stack**: `vllm 0.27.1`, `pgvector 0.5.0`, `networkx 3.6.1`; no docling/kuzu/
  llama-index/FlagEmbedding installed.

---

## G. Phase gate

Phase 0 is complete and **read-only** — no source changed. Awaiting approval to
proceed to **Phase 1 (Schema)**. The first decision needed from you:

1. **Converge or keep two paths?** Recommend converging on the structure-aware
   live path and folding/deleting `ingest_v2` (restores single source of truth).
2. **Retrieval target** — re-point consumers at `chunks`, or backfill `documents`
   from hierarchical nodes? (Determines Phase 1 schema work and migration risk.)
3. **Reranker** — load bge-reranker-v2-m3, or record `FeatureReranker` as the
   accepted trade-off in CLAUDE.md?

### Decisions taken (2026-08-22)

1. **Converge — DONE.** `ingest_v2` (dormant + buggy) was deleted in the prior
   cleanup; the structure-aware `HierarchicalChunker` is the single ingest path.
2. **Backfill `documents` from hierarchical nodes (option 1a).** Consumers and the
   `documents` table are unchanged; `IngestService._retrieval_chunks` now populates
   `documents` from `HierarchicalChunker(compact=False)` prose nodes, tagging each
   row `meta.chunk_kind=structure` + `meta.section_path`. Fixed-size `TextChunker`
   remains only as a logged fallback (`chunk_kind=fallback`,
   `ingest.fixedsize_fallback`). `characters_read` is now on `IngestResult` /
   `IngestResponse` and logged as `ingest.chunked` coverage (§3). No schema DDL, no
   migration — old rows convert on their next re-ingest/re-sync.
3. **Keep `FeatureReranker`, recorded as an accepted trade-off** (§7). `bge-reranker-v2-m3`
   is deferred to a future phase gated on a 12GB VRAM co-residency benchmark
   (chat-7B-AWQ + BGE-M3 + reranker). Documented in `docs/architecture.md`.

## H. Phase 2 — LLM topic-shift segmentation (2026-08-22)

Phase 2 closes the §2 gap ("structure-aware first; LLM topic-shift for
unstructured; **not** embedding-similarity cuts"). Two problems were measured on
the real corpus first, then fixed.

### H.1 The measured problem

Of 89 sources, only 19 carry markdown headings — **70 are heading-less**, 41 of
those over 1500 characters. Phase 1's `compact=False` chunking failed on them in
two opposite directions at once:

| Source | chars | nodes | median | tiny (<200) |
|---|---|---|---|---|
| `H200_AI_Financial_Report_24M_U` | 35203 | 1551 | 10 | 1524 |
| `文档 (1)` | 10938 | 380 | 24 | 380 |
| `terms_of_service` | 12509 | 191 | 50 | 177 |
| `Premium Mobile Oasis™ – Model A-1P` | 8422 | **3** | **3706** | 0 |
| `Contract/Rental Agreement` | 16040 | **6** | **3537** | 0 |

Office/PDF extraction emits one blank-line block per table cell, so
fragment-heavy documents shredded into unretrievable labels (`Yes`, `6379`),
while heading-less prose stayed in ~3700-character blobs — 783 tokens of mixed
topics in one embedding.

### H.2 What changed

1. **Coalesce without prune.** `HierarchicalChunker.chunk()` gained
   `prune: bool = True`; the retrieval path passes `compact=True, prune=False`.
   Coalescing merges cell fragments and *raises* coverage; the prune half is
   lossy (it folds undersized nodes into sections and drops sections that
   absorbed nothing) so it stays off for rows retrieval must return.
2. **`services/ingest/segment.py::TopicShiftSegmenter`.** A prose node over
   `segment_trigger_chars` (2000) is handed to the chat model, which returns the
   sentence indices that start a new topic. Boundaries are validated against the
   real sentence list — out-of-range, duplicate, non-integer, and boundaries
   thinner than `segment_min_sentences` are discarded — so a hallucination
   degrades the cut, never the content. Lists and tables are never segmented.
   Fallbacks: model error, unusable payload, or a node over
   `segment_max_sentences` (120) → sentence-boundary packing, logged.
   **No embedding cosine is consulted anywhere in the boundary decision.**
3. **Config knobs** in `core/config.py`: `segment_enabled`,
   `segment_trigger_chars`, `segment_min_sentences`, `segment_max_sentences`.
4. **`meta.chunk_kind`** now records provenance across four values:
   `structure` | `topic_shift` | `sentence_pack` | `fallback`.

### H.3 Measured result

Whole-corpus re-chunk of all 57 documents ≥1500 chars (57 docs, no DB write):

```
A (Phase 1, compact=False)      nodes=4253  tiny=3561
B (Phase 2, coalesce+segment)   nodes=1222  tiny=131   oversized(>2k)=76 → LLM
```

`H200_AI_Financial_Report_24M_U` 1551 → 130 nodes, median 10 → 207 chars,
coverage 0.91 → 0.95, tiny 1524 → 16. `文档 (1)` 380 → 49, median 24 → 211.
`terms_of_service` 191 → 46, tiny 177 → 2. Coverage never fell on any document.

Live boundary quality against the running vLLM (`Qwen2.5-7B-Instruct-AWQ`):

```
Premium Mobile Oasis™ – Model A-1P  3706 chars / 16 sentences → 3 segments in 0.9s
  [0:5]   product header + rating          (1247 chars)
  [5:14]  deployment/infrastructure spec    (1975 chars)
  [14:16] usage & maintenance instructions  (482 chars)
Contract/Rental Agreement           3537 chars / 45 sentences → 5 segments in 0.9s
  [0:4] policies preamble  [4:15] reservation & deposit  [15:25] invoicing
  [25:31] remedies on non-payment  [31:45] on-site movement restrictions
```

Every segmentation was verified lossless (`rejoined == original sentences`) and
cost <1s per node — well inside the §0 hardware budget, no extra VRAM.

### H.4 Verification

```
.venv/bin/python -m pytest -q                 → 554 passed
.venv/bin/python -m ruff check api core services scripts tests
                                              → 12 UP042 only (intentional repo style)
.venv/bin/python -m mypy core/config.py       → Success
```

16 new tests: `tests/test_topic_shift_segmenter.py` (10 — boundary honouring,
hallucinated/duplicate/non-integer index rejection, exact content preservation,
thin-boundary rejection, LLM error, malformed payload, too-few/too-many
sentences, token budget) and `tests/test_ingest_structure_documents.py` (6 —
structure chunks, table-cell coalescing, topic-shift wiring, sentence-pack
degradation, fixed-size fallback).

### H.5 Still open (next phase gate)

Obsidian display/Canvas layer + reverse-sync (§3), entity-resolution blocking
stage + graph-structural signal + `NEEDS_REVIEW` counts (§5), Docling/MarkItDown
structure-preserving parsers, `documents`/`chunks` schema unification, and
`bge-reranker-v2-m3` gated on a 12GB co-residency benchmark. Existing rows keep
their Phase 1 chunking until their next re-ingest or vault re-sync.

## I. §5 — Three-layer entity resolution: blocking, structural evidence, review counts (2026-08-22)

### I.1 The measured problem

`resolve_entities` loaded **every** entity in the database for every document
(`select(db.Entity)`, unfiltered) and rebuilt the resolver's lookup tables per
mention. Live measurement, 2470 entities:

```
resolve() 60 mentions x 2470 known : 519.5 ms  (8.66 ms/mention)
resolve() 60 mentions x  155 known :  32.3 ms  (16.1x)
SELECT * FROM entities             : 400 ms   (the 1024-dim vector dominates;
select(...) load_only lean columns  :  28 ms    resolution never reads it)
```

The cost is quadratic in corpus size and degrades with every ingest. The audit's
§5-ER gap also stood unchanged: no explicit blocking stage, no graph-structural
signal in scoring, no `NEEDS_REVIEW` count anywhere.

Two correctness defects surfaced while measuring:

- **Empty-norm auto-merge.** `normalize_name("platform") == ""` (every token is
  a generic suffix), so `resolve("system", known=[Entity("platform")])` returned
  `AUTO_MERGE` at confidence **0.98** — two unrelated generic words collapsing
  into one entity. Both live rows existed.
- **Domain never reached entity rows.** `_run_semantic_pass` did not pass the
  classifier's `domain`, so every entity carried `meta.domain=null` and the
  same-name-different-domain review gate could never fire.

### I.2 What changed

1. **Blocking (layer 1)** — new `services/entity/blocking.py`. Blocking keys are
   normalized tokens (≥3 chars, technical acronyms exempt) plus multi-word
   acronyms (`Legal Integrity Analytics` → `lia`), derived from the resolver's
   own normalizer so a name cannot block differently from how it is compared.
   `block_candidates(mentions, known)` keeps only key-sharing candidates; a
   pathological block truncates at 400 and logs `truncated=True`.
2. **Structural scoring (layer 2)** — `EntityResolver.resolve` accepts
   `graph_neighbors` + `context_ids`. Shared entity→entity `knowledge_links`
   neighbours (edges that already passed `RelationValidator`, never cosine)
   raise merge confidence to a 0.99 ceiling, enrich subsumption hints, and can
   promote token overlap to `REVIEW_REQUIRED`. Structure alone never merges:
   "Nomad" vs "Kubernetes" with three shared neighbours stays `DO_NOT_MERGE`.
3. **Review clustering + counts (layer 3)** — `review_clusters` groups rows whose
   normalized names collide (13 groups live, e.g. `Machine learning` /
   `machine-learning`, caused by the keyword path and resolver path deduplicating
   with different normalizers). Domain-conflict and shared-neighbour mentions are
   recorded as `lifecycle="review"` `POSSIBLE_RELATION` edges on the colliding
   entity, carrying both domain assertions plus `matched_by` and
   `structural_support`, instead of vanishing. A second entity row is impossible
   here — `canonical_name` is globally unique, and a live probe proved that
   inserting one raises `UniqueViolationError` and rolls back the whole semantic
   pass, destroying the chunks and units alongside it.
   `semantic_pass` stats carry `mentions/candidates_considered/candidates_blocked/
   needs_review`; `/api/ingest/report` returns `needs_review_count` (entities
   never confirmed) and `resolution_conflict_count` (pairs awaiting a ruling).
4. **Rule 0** — a mention with no identity tokens after normalization merges onto
   nothing (confidence 0.2), closing the empty-norm defect.
5. **Wiring** — extraction runs before resolution so blocking builds once per
   document; `_load_entities` projects six lean columns (400ms → 28ms); the graph
   query runs only when the document anchors on a known entity; linear scans over
   `known` became dict lookups; `_run_semantic_pass` propagates the classifier's
   `domain`; `logger` switched to structlog `get_logger` (the old stdlib logger
   raised `TypeError` on any kwargs-style event).

### I.3 Measured result

Recall proof — 920 synthetic mentions (exact/case/article/suffix/acronym/
subsumption/novel) across 23 documents, blocked decisions compared against
full-corpus decisions on all 2470 live entities:

```
full-scan time  : 6695.4 ms
blocked time    :  417.6 ms   speedup 16.0x
recall differences: 0
review clusters on live data: 12
```

First live ingest after restart (README vault re-sync):

```
semantic.entity_blocking  candidates=50 considered=2470 context=14 keys=15 mentions=14
ingest.semantic_pass      stats={'chunks': 3, 'entities': 0, ..., 'candidates_considered': 2470,
                                 'candidates_blocked': 50, 'needs_review': 0, ...}
/api/ingest/report        needs_review_count=2232  (the pre-existing ambiguous backlog,
                          now visible instead of silent)
```

Same-name-different-domain probe — two unrelated documents both calling their
subject "Kestrel" (a falconry bird, a wind turbine model):

```
POST /api/ingest/text  "Kestrel Falconry Handbook"        -> added=1
POST /api/ingest/text  "Kestrel Turbine Maintenance Log"  -> added=1
resolver decision: REVIEW_REQUIRED, "same name as 'Kestrel' but conflicting
                   domain 'falconry' vs 'wind energy'"
```

The first attempt at this persisted the review as a second `entities` row and hit
`duplicate key value violates unique constraint "uq_entities_canonical_name"`,
rolling back the entire pass. That is what drove the review-edge design: the
review has to be non-destructive to the rest of the semantic layer.

After the fix, a third document about the turbine fleet resolved cleanly:

```
semantic.entities_need_review  mentions=12 needs_review=1
ingest.semantic_pass  stats={'chunks': 1, 'entities': 15, ..., 'needs_review': 1, 'units': 8}

knowledge_links row (lifecycle=review, POSSIBLE_RELATION, confidence=0.4):
  {"domain": "energy infrastructure", "mention": "Kestrel", "review_of": "Kestrel",
   "matched_by": "canonical", "candidate_domain": "falconry",
   "structural_support": 0, "entity_resolution_review": true}

entities named "Kestrel": exactly 1 (the falconry bird) — no duplicate, no merge
/api/ingest/report: needs_review_count=2255  resolution_conflict_count=1
```

Retrieval separates the two senses without any entity merge:
`POST /api/search "Kestrel turbine bearing lead time"` ranks the turbine document
first (0.552) above the falconry one (0.432).

### I.4 Verification

- `.venv/bin/python -m pytest -q` → **592 passed**
- `.venv/bin/python -m ruff check api core services scripts tests` → 12 UP042 only (intentional repo style)
- `.venv/bin/python -m mypy core/config.py` → Success
- Restarted `braind.service`; `/` 200; `/api/ingest/stats` sources=92 chunks=492; report shows the new field.

### I.5 Still open

- The 2232-row ambiguous backlog predates this change; `review_clusters` output
  gives the merge order for a future operator pass (explicit approval required).
- Clustering is reported, not acted on: no automatic merge of normalization
  collisions (deliberate — some collisions are genuinely distinct entities).

---

## J. §3 — Obsidian display + Canvas export layer (2026-08-24)

### J.1 The measured problem

The §C gap table rated `§3-Obsidian` **HIGH**: *"Zero `.canvas` files; no
unit-per-file export; no reverse-sync. Vault is input-only + parent wikilink."*

A rich semantic layer existed and was invisible outside SQL:

```
sources 108 | semantic documents 70 | chunks 428 | semantic_units 2667 | entities 2513
.canvas files in vault: 0
```

Two hazards had to be settled before writing a single file.

**Hazard 1 — the ingestion feedback loop.** `vault/` is the *input* to
ingestion. `sync_vault` (`services/ingest/service.py:695`) ingests every `*.md`
under it, on a 5-minute scheduler job plus once on every boot
(`api/main.py:238`). Writing generated notes there would feed derived content
back in as source material on every cycle. The skip-hash cache
(`service.py:171`) is a plain in-memory dict, so "already ingested once" is not
protection — a restart re-scans everything. The only existing exclusion was
`".obsidian" in path.parts`; `RAGService.default_excludes` filters
`writer_generated` at *retrieval* time only, so the writer's precedent does not
transfer (the writer never writes files to disk).

**Hazard 2 — almost no id is stable.** `semantic_pass` deletes the prior
`documents` row and CASCADE removes its chunks and units, and `SemanticChunk`
mints a fresh `uuid4()` per run (`core/chunking.py:199`). Document, chunk and
unit UUIDs all change on every re-ingest. Only `entities.normalized_name`
(unique index) and `sources.path` (row updated in place) survive.

Measurement also shaped what was worth rendering:

| Signal | Live value | Consequence |
|---|---|---|
| `chunks.parent_id` populated | **1 / 428** | hierarchy must come from `section_path` + `order` |
| `semantic_units.entity_ids` non-empty | 1759 / 2667 | the unit→entity wikilink is real |
| typed `entity→entity` edges | ~20 | a Canvas on typed edges alone would be empty |
| negated units (`polarity = -1`) | 469 | negation must survive rendering |

### J.2 What changed

1. **New package `services/obsidian/`.** `render.py` is pure — no DB, no I/O —
   so the exact bytes landing in the vault are unit-testable. `exporter.py` owns
   the reads and writes atomically (temp file + `os.replace`) and skips files
   whose content is unchanged, so Obsidian's watcher and git stay quiet.
2. **Two independent guards** close the feedback loop: a path guard on
   `settings.obsidian_export_dirname` in `sync_vault`, and a
   `braind_generated: true` frontmatter marker checked in `_ingest_file`
   **before the skip-hash is recorded** — so a generated note moved into a user
   folder is re-checked every sync rather than remembered as "seen". Both are
   structural, because a restart loses the hash cache.
3. **Stable filenames only**: `entities.normalized_name` and `sources.path`.
4. **Canvas edges are co-mention** with typed `knowledge_links` overlaid and
   labelled; an edge whose other end is not on the Canvas is dropped rather than
   left dangling.
5. **Nothing is hidden**: unconfirmed entities export tagged `braind/review`,
   and `polarity = -1` renders an explicit **NOT**.
6. Export runs after each semantic pass (best-effort — a write failure never
   rolls back committed knowledge) and rebuilds via
   `POST /api/ingest/obsidian-export`.

### J.3 Measured result

Full export against the live database:

```
obsidian.export_all  sources=108 documents=70 entities=934 canvases=70 pruned=0 errors=0  (1.8s)
vault/braind: 1004 files (934 entity notes, 70 document notes, 70 canvases), 0 stray .tmp
canvases: 1004 nodes / 940 edges total; typed edges drawn: {'IS_A': 6}
entity notes tagged braind/review: 830 / 934
document notes rendering an explicit **NOT**: 46
```

Re-running the export immediately after wrote **nothing** —
`documents=0 entities=0 canvases=0` — so an unchanged corpus produces no
filesystem churn.

### J.4 Verification

- `.venv/bin/python -m pytest -q` → **645 passed** (592 + 53 new)
- `.venv/bin/python -m ruff check api core services scripts tests` → 12 UP042
  (intentional repo style) + one pre-existing E501 in `services/agents/office.py`
- `.venv/bin/python -m mypy core/config.py` → Success

**Feedback loop, proven live** against the real vault. `sync_vault` was run with
the ingest step instrumented, over a vault now holding 1014 markdown files:

```
markdown files in vault      : 1014
  of which generated (braind/): 1004
  user notes                  : 10
handed to ingest by sync_vault: 10
generated files that leaked   : 0
```

**Marker guard, proven against the real `_ingest_file`** (the scan above stubs
that method, so it cannot see this guard). A generated note was copied into
`vault/references/`:

```
stray generated note -> added=0 updated=0 errors=0
entered the skip-hash cache: False
session.execute called      : False
```

**Prune safety, proven on the real export tree.** A hand-written note and a
stale generated note were both placed in `vault/braind/entities/`:

```
obsidian.prune_skipped_foreign path=vault/braind/entities/zzz-my-own-note.md
pruned=1
hand-written note survived : True
stale generated note gone  : True
```

**Mutation-tested.** With both guards replaced by `if False:`, all 7 tests in
`tests/test_ingest_vault_export_guard.py` fail; restored, all 7 pass. The
generated fixtures are deliberately long and prose-like, because a short note is
rejected by `core.quality.assess` as `too_short` — which would make the tests
pass whether or not the guards existed.

### J.5 Still open

- **Reverse-sync is not built** (watch → re-validate → update). Deferred
  deliberately: export cannot corrupt data, reverse-sync can.
- Entity extraction noise is now visible in the vault — many entity notes are
  extraction artefacts (`detect`, `must-be`, `over`). The export did not create
  this; it surfaced it. Worth a cleanup pass against the 830 review-tagged notes.
- Document slugs embed the content hash for text-ingested sources
  (`sources.path` is the hash there), so they are stable but not readable.
