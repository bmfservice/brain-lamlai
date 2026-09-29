# BrainD Data Ingestion — Risks (Current State + Upgrade)

> Audit deliverable, Phase 0. Risks the ingestion upgrade must respect. Paths relative to `braind/`.

## Operational / runtime risks

1. **GPU VRAM + vLLM max-model-len.** The writer requests up to `MAX_OUTPUT_TOKENS=8000`
   (`services/writer/service.py`), but vLLM enforces `input_tokens + max_tokens ≤ max_model_len`.
   A chat-vLLM restarted at `--max-model-len 16384` + `--gpu-memory-utilization 0.7` currently
   serves this; the config template `config/braind-vllm-llm.service` carries those values.
   Embedding (bge-m3, `:8002`) shares the same 12 GB card — parallel startup OOMs if both reserve
   too much; start chat first, keep embed util modest.
2. **Embedding-model/version mixing.** All active vectors are `embedding_vllm_1024` (bge-m3, 1024-dim).
   A changed embedding model must not silently mix with older vectors — record `embedding_version`,
   and use `embedding_versions` for the registry. Re-embedding path = `scripts/migrate_vllm_embeddings.py`.
3. **Network cost of search validation.** Blindly web-searching every node would explode latency and
   duplicate/conflict noise. Any newly added search validation MUST be gated by freshness + source
   type + hard outbound cap, reuse the SSRF-safe `WebSearchTool`, and treat results as evidence, not truth.

## Data-safety risks (CLAUDE.md — hard rules)

4. **Han-ideograph rejection.** `core/textanalysis.py contains_chinese` (CJK U+3400–U+4DBF and
   U+4E00–U+9FFF) is the authoritative pre-ingest rejection gate (already in `ingest_text`, `_ingest_file`,
   `services/learn/service.py`). Keep it authoritative; language detection supplements, never overrides.
5. **Additive-only DDL / never drop.** DB changes use `init_db()` idempotent DDL + an additive alembic
   revision. Never `DROP TABLE`/`TRUNCATE`/`RESET` as a shortcut. Backup before non-additive DDL;
   new revision must be strictly additive.
6. **Source overwrite by web evidence.** Authoritative/internal input must be preserved. If web results
   conflict, store `external_update_candidate` and mark CONFLICTED — never silently overwrite the source.
7. **No data loss on semantic failure.** A failure in any enrichment step must not delete original
   content or roll back the committed source (semantic pass is already best-effort + independently
   committed).
8. **SSRF.** All external page fetching goes through `core/net.py:fetch_text_bounded`
   (`is_safe_url` / `is_public_host`, re-checked after every redirect, plus content-type/size/timeout
   caps). `FetchURLTool`/`scrape_url`/`fetch_feed` were unguarded until 2026-08-20 and now share that
   fetcher. Never issue a raw `httpx` request against a user- or model-supplied URL.

## Scope / architecture risks

9. **Two parallel pipelines.** The legacy `Source`+`Document` path and the semantic
   `chunks`/`units`/`entities` path both run. Extend both coherently; do not break the legacy retrieval
   (RAG reads `documents`). Any new columns must be nullable/additive so existing rows stay valid.
10. **Avoid duplicate tool implementation.** Reuse `WebSearchTool`, `TextAnalyzer`,
    `EntityExtractor`, `ClaimExtractor`, `RelationValidator`, `chat_json`, `embed_batch`. Do not add
    parallel web-search or embedding stacks.
11. **No out-of-scope redesign.** Remember hydrate: no changes to chat, agent loop, business logic,
    model serving, fine-tuning, etc. Record any needed cross-system change in `INGESTION_FOLLOWUP.md`.
12. **Tests are fully mocked.** Tests must not require live PostgreSQL, vLLM, filesystem state, or
    public network. New modules must be import-safe and use `tests/conftest.py` fixtures.

## Performance risks
- Long single `ingest_text` calls block the request; chunking + `chat_json` extraction should stay
  bounded (`TextAnalyzer.max_chars`, `ClaimExtractor` LLM-only >220 chars, `dedup_check` over
  `content[:4000]`).
- HNSW index build on large uploads is expensive; batch embeddings via `embed_batch` where possible.