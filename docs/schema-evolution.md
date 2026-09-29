# Schema Evolution and Embeddings

## Current State

`core/database.py` defines the SQLAlchemy models and applies additive database bootstrap work during application startup. PostgreSQL with `pgvector` is the source of truth.

The active vector representation is:

- column: `documents.embedding_vllm_1024`
- dimensions: `1024`
- provider: local vLLM embedding server
- model: `BAAI/bge-m3`

The legacy 768-dimensional embedding column is retained only for controlled migration rollback. Production retrieval and ingest use `embedding_vllm_1024`.

## Required Procedure for Schema Changes

1. Back up PostgreSQL and record the deployed Braind revision.
2. Write a focused migration or a clearly idempotent bootstrap change with tests.
3. Test it against a disposable database before live execution.
4. Validate indexes, vector dimensions, FTS behavior, and existing row counts after migration.
5. Deploy one component at a time and retain a rollback path.

Do not bundle destructive DDL, vector cutover, or data reprocessing into an unrelated source cleanup.

## 768-to-1024 Re-embedding Procedure

Run only after both vLLM endpoints are healthy:

```bash
PYTHONPATH=. .venv/bin/python scripts/migrate_vllm_embeddings.py --batch-size 32
```

Use the script's validation report to confirm every active document has a 1024-dimensional embedding. The `--cutover` option changes legacy column naming and must be treated as a separate, backed-up operation.

## Maintenance Scripts

`backfill_phase2.py` and `cleanup_knowledge.py` mutate existing rows. They require a database backup, a dry-run where available, and an explicit change review. They are not part of normal startup or cache cleanup.

## Future Direction

The current bootstrap is acceptable for additive local development but is not a substitute for an auditable migration history. Before introducing non-additive DDL, adopt a versioned migration runner and move schema transitions out of runtime startup.
