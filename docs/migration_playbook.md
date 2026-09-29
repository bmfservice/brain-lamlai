# Brain Second Migration & Rollback Playbook (Phase 15)

This document describes how to deploy the Brain Second upgrade on a live Braind
instance, how to verify each phase, and how to roll back if a regression is
found.

## Pre-requisites

- PostgreSQL 15+ with `pgvector` and `pg_trgm` extensions created.
- Both vLLM servers reachable on `127.0.0.1:8001` (chat) and `127.0.0.1:8002`
  (embedding).
- A fresh database backup: `pg_dump -Fc braind -f data/backups/braind_$(date +%F).dump`.
- `.env` reviewed and any new Brain Second variables applied.

## Deployment steps

1. **Stop ingest/writer jobs** so no new data arrives during schema changes.
   ```bash
   systemctl --user stop braind
   ```
2. **Apply database migrations** (new tables/columns are additive unless noted
   in `docs/schema-evolution.md`).
   ```bash
   .venv/bin/python -m alembic current
   .venv/bin/python -m alembic upgrade head
   ```
3. **Run the evaluation regression suite** against the committed golden cases.
   ```bash
   .venv/bin/python scripts/run_evaluation.py
   ```
4. **Start vLLM services** with the updated `config/*.service` units that point
   `HF_HOME` at `braind/models/hf`.
5. **Start Braind** and verify the health endpoints.
   ```bash
   systemctl --user start braind
   curl http://127.0.0.1:8000/
   curl http://127.0.0.1:8001/v1/models
   curl http://127.0.0.1:8002/v1/models
   ```
6. **Smoke-test** the endpoints:
   - `POST /api/search/unified`
   - `GET /api/search/debug?q=...`
   - `GET /api/debug/metrics`
   - `GET /api/ingest/stats`

## Rollback

If any golden case fails or the service is unstable:

1. Stop Braind: `systemctl --user stop braind`.
2. Restore the pre-upgrade database backup.
3. Revert the source tree to the previous tag/branch.
4. Restart Braind with the previous service units.

## Checkpoints

| Checkpoint                     | Pass criteria                                  |
| ------------------------------ | ---------------------------------------------- |
| Golden regression suite          | `recall_at_k >= 0.50`, no new failures        |
| Health endpoints                 | All three return 200 / model lists             |
| Ingest smoke test                | One vault sync completes without error         |
| Integration tests                | `pytest tests/integration -q` green against the live database |
| Semantic layer intact            | `chunks` and `semantic_units` counts non-zero after the vault sync |

## Notes

- Do not run `scripts/backfill_phase2.py`, `scripts/cleanup_knowledge.py`,
  `scripts/migrate_vllm_embeddings.py`, or `scripts/relearn_semantic.py` as part
  of routine deployment; they are operator-only and require explicit approval
  and a backup.
- Keep `vault/`, `data/`, `logs/`, and `models/` intact across rollback.
