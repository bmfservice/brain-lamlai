# Phase tracking for the approved CLAUDE.md remediation plan.
# See the approved remediation plan (local plan file, not committed).

STATUS: IN PROGRESS

## Step 0 — Foundation (owner: me, blocking for all 3 lanes) [DONE]
- core/database.py: add ORM for taxonomy_facets/terms/aliases/rules, capabilities,
  reorg_operations, gpu_resource_samples, agent_runs, agent_domains, and close
  drift for agents, agent_tasks, agent_conversations, agent_reports,
  agent_audit_events, agent_memory -> [x] (verified: Base.metadata has all 15 models, 47 tables total)
- alembic revision d6a91f3c8e42 (down_revision='7c1e9d4a2f80') applying all of
  the above (additive, zero seed data, COALESCE unique indexes) -> [x] (verified: upgrade head -> downgrade -1 -> upgrade head roundtrip on real PostgreSQL)
- core/textanalysis.py: normalize_label() extracted from taxonomy.coerce -> [x] (verified: tests/test_settings_schema.py)
- settings spec rows for the new thresholds (catalog.py) -> [x] (verified: 17 STEP0_FOUNDATION specs, bounds checked, gpu_reserve_free_mib >= 5120 MiB per CLAUDE.md §3)
- test_step0_foundation.py -> [x] (verified: 7/7 passing in tests/integration/test_step0_foundation.py, full suite 226 passed, ruff clean)

## Lane 1 — Dynamic knowledge substrate
- services/taxonomy/ service (empty-start, model-proposed categories)  -> [ ]
- core/taxonomy.py -> facade, no frozensets                            -> [ ]
- services/relation/validator.py: VALID_EDGE_TYPES + _SIGNAL_WEIGHTS   -> [ ]
- services/extract/relation_llm.py: open vocabulary                    -> [ ]
- services/ingest/classify.py: dynamic prompt, no language default     -> [ ]
- services/ingest/{service,freshness,semantic}.py: stopword/hint -> DB,
  embedding_version derived                                             -> [ ]
- services/entity/normalize.py                                         -> [ ]
- services/learn/service.py query suffixes -> DB                       -> [ ]
- concurrency: rag/service.py gather, ingest embed_batch, graph, learn -> [ ]
- model ids -> env/DB only (core/config.py)                            -> [ ]

## Lane 2 — Runtime + self-org + Settings UI
- core/resource_guard.py (nvidia-smi over WSL, auto-degrade)            -> [ ]
- services/selforg/{merge,split,discover,journal}.py                   -> [ ]
- scheduler hook, non-blocking                                          -> [ ]
- Phase 6 Settings Console (api/static/main + view-settings)            -> [ ]
- api/routes_settings.py: POST /definitions (user-owned keys)           -> [ ]
- registry / nav entries                                                -> [ ]

## Lane 3 — Agents into braind second (real DB, real data)
- DELETE api/static/office/ + GET /office route                         -> [ ]
- services/agents/context_plane.py (RAG+graph+memory grounding)         -> [ ]
- agent_runs durable checkpoint/resume replacing _runs                  -> [ ]
- agent_domains capability routing                                      -> [ ]
- ReAct loop for specialists                                            -> [ ]
- chat seam: auto-create workflow from action-bearing input             -> [ ]
- verification + quality gate wired to the workflow result              -> [ ]

## Step 9 — Restart + full test (real data, real DB)                       -> [ ]
