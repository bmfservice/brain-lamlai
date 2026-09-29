# Braind Operations

## Required Local Services

Braind runs as native processes. It does not require Docker or Dify.

| Component | Bind address | Purpose |
| --- | --- | --- |
| PostgreSQL + pgvector | `localhost:5432` | Durable sources, document chunks, vectors, graph links, generated content, and learning audit history. |
| vLLM chat | `127.0.0.1:8001` | `Qwen/Qwen2.5-7B-Instruct-AWQ` chat generation. |
| vLLM embeddings | `127.0.0.1:8002` | `BAAI/bge-m3` embeddings with 1024 dimensions. |
| FastAPI | `127.0.0.1:8000` | Braind REST API and `/ui` console. **No authentication layer** — see below before widening the bind address. |

## Configuration

1. Copy `.env.example` to `.env`.
2. Set real `DATABASE_URL` credentials. Do not commit `.env`.
3. Keep these values consistent with the deployed vLLM units:

```dotenv
VLLM_LLM_BASE_URL=http://127.0.0.1:8001
VLLM_EMBED_BASE_URL=http://127.0.0.1:8002
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
EMBED_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSIONS=1024
```

The service-unit templates in `config/` use absolute project paths (including a machine-specific `CUDA_HOME`). If Braind moves, update all three unit files together before installing or restarting them.

## Site Crawl and Uploads

`POST /api/learn/site` reads one website: it fetches the start URL, follows the
links it finds, and ingests every page that yields enough text — breadth-first,
staying on the same site (subdomains included), up to `CRAWL_MAX_DEPTH` tiers.

Depth alone does not bound a real site, so the limits that actually stop a run
are `CRAWL_MAX_PAGES` (default 300) and `CRAWL_HARD_TIMEOUT_SEC` (default 1800).
A timeout is not a failure: everything already ingested is kept and the run is
recorded as done.

```bash
curl -N -X POST http://127.0.0.1:8000/api/learn/site \
  -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
  -d '{"url":"https://example.com/docs","max_pages":300}'
```

Pass `topic` to skip off-topic pages (each page is scored by the LLM, which
roughly doubles per-page cost); omit it to learn the whole site.

Every crawled page becomes a `web_crawl` source; the per-URL audit trail is in
`learn_urls`, joined to `learn_runs` by `run_id`.

`POST /api/ingest/file` accepts .pdf, .docx, .pptx, .xlsx, .odt, .epub, plain
text, Markdown, and source code, up to `UPLOAD_MAX_BYTES` (default 25MB).
Parsers live in `core/documents.py`.

## Deep Ingest

Everything entering the brain passes a quality gate before it is stored. The
gate strips repeated site chrome, then rejects binary payloads, search-result
snippets, navigation menus and Han ideographs. Trusted local sources (vault,
uploads, manual paste, chat) skip the prose thresholds — a config file is
legitimately not prose. Rejections are logged as `ingest.rejected` with the
reason; nothing is silently dropped.

What survives is classified in one LLM round-trip into columns retrieval can
filter on:

| Column | Vocabulary |
| --- | --- |
| `doc_type` | reference, tutorial, explainer, news, analysis, discussion, marketing, data, personal, other |
| `chunk_role` | definition, procedure, example, data, claim, context, reference, navigation |
| `audience` | general, technical, business, academic, legal |
| `quality_tier` | high, medium, low — measured by the gate, not claimed by the model |

Chat and the writer read those labels back: a "what is X" question prefers
definitions from reference material, "how do I X" prefers procedures from
tutorials, and the writer demotes marketing copy so generated text is not
grounded in sales language.

Vocabularies are closed (`core/taxonomy.py`). Model output is coerced onto
them — an unrecognised label becomes the default rather than a new category.
Add a term there, never at a call site.

## Starting the brain from scratch

`scripts/purge_brain.py` deletes every trace of ingested knowledge: sources,
documents, chunks, semantic units, entities, graph edges, learning runs,
generated content and chat history. It keeps configuration (UI registry,
scheduler, audit log, alembic version) and never touches the Obsidian vault on
disk, which re-syncs on next start.

```bash
PYTHONPATH=. .venv/bin/python scripts/purge_brain.py --dry-run
pg_dump -Fc braind -f data/backups/braind_pre_purge_$(date +%F_%H%M%S).dump
systemctl --user stop braind.service
PYTHONPATH=. .venv/bin/python scripts/purge_brain.py --apply
systemctl --user start braind.service
```

The script refuses to run without a dump newer than an hour: a stale backup
cannot contain what is about to be destroyed.

## Network Exposure

There is no API authentication. `HOST` and `braind.service` bind `127.0.0.1`, and `CORS_ALLOW_ORIGINS` is a loopback whitelist. Every endpoint — including ingest, vault sync and scheduler triggers — is available to anyone who can reach the port, so put an authenticating reverse proxy in front before changing either value.

## Local Start and Health Checks

From the `braind/` directory:

```bash
curl --fail http://127.0.0.1:8001/v1/models
curl --fail http://127.0.0.1:8002/v1/models
.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8010
```

On Linux/macOS the interpreter path is `.venv/bin/uvicorn` instead. Prefer the
orchestrator, which also waits for health before starting the app:
`powershell -File scripts\start_bill.ps1`.

Then verify:

```bash
curl --fail http://127.0.0.1:8010/
curl --fail http://127.0.0.1:8010/api/agent/tools
curl --fail http://127.0.0.1:8010/api/ingest/stats
curl --fail -X POST http://127.0.0.1:8010/api/search/unified \
  -H 'Content-Type: application/json' -d '{"query":"braind","top_k":3}'
```

Open `http://127.0.0.1:8010/ui` for the built-in UI.

## systemd User Units

These supervise **only the two vLLM servers**, which run inside WSL2. The FastAPI
app runs natively on Windows (port `8010`, launched by `scripts/start_bill.ps1`),
so there is deliberately no `braind.service` unit — that earlier draft assumed an
all-Linux deploy.

The unit files live in the repo:

- `config/braind-vllm-embed.service` — embedding server (`start_vllm.sh embed`)
- `config/braind-vllm-llm.service` — chat server (`start_vllm.sh chat`)
- `config/braind-vllm.target` — one handle for the pair

Why units and not a detached `wsl.exe ... &`: WSL2 tears down a session's process
tree when the launching `wsl.exe` exits, so anything backgrounded that way dies
with it (observed: ports drop to `000`, VRAM returns to idle, and `/tmp` output
vanishes because `/tmp` is RAM-backed). Units are owned by the distro's `systemd
--user` manager instead, so they survive every `wsl.exe` client exiting.

Install (idempotent; enables without starting, so a live stack is untouched):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_units.ps1
```

Everyday operation — prefer the orchestrators, which also handle health waits:

```powershell
powershell -File scripts\start_bill.ps1   # vLLM via systemd + app on Windows
powershell -File scripts\stop_bill.ps1    # stops units, legacy holder, and app
powershell -File scripts\status_bill.ps1  # HTTP + unit state + VRAM
```

Or directly, from Windows (`systemctl --user` works through `wsl.exe -e` because
`XDG_RUNTIME_DIR` and the session bus are already set):

```powershell
wsl -d Ubuntu -e systemctl --user start braind-vllm.target
wsl -d Ubuntu -e systemctl --user is-active braind-vllm-llm.service
wsl -d Ubuntu -e journalctl --user -u braind-vllm-llm.service -f
```

Details the units depend on:

- **Embedding starts first** (`After=braind-vllm-embed.service` on the chat unit).
  Both servers are pinned to different cards via `CUDA_VISIBLE_DEVICES` from
  `.env`, so this is startup ordering rather than a hard VRAM dependency.
- **`KillMode=control-group`** is required: vLLM forks `EngineCore` workers that
  would otherwise survive a stop and keep holding VRAM, breaking the next start.
- **No `LD_LIBRARY_PATH` is needed.** WSL writes `/etc/ld.so.conf.d/ld.wsl.conf`
  containing `/usr/lib/wsl/lib`, so `libcuda`/`libnvidia-ml` resolve through the
  ldconfig cache even in systemd's clean environment.
- **`TimeoutStartSec=900`**: vLLM binds its port long before the model finishes
  loading, and JIT-compiling flashinfer kernels makes the first start slow.

By default user units stop at logout and do not start until you log in. They do
survive closing terminals and `wsl.exe` clients without further setup. To also
start them at WSL boot, enable lingering once (needs your password):

```bash
sudo loginctl enable-linger "$USER"
```


## Operator-Only Scripts

These scripts may modify PostgreSQL. They are not cleanup commands and must not be run during ordinary maintenance.

| Script | Purpose | Required safeguard |
| --- | --- | --- |
| `scripts/purge_brain.py` | Delete ALL ingested knowledge, leaving an empty brain. | Requires `--apply` and a dump newer than 60 minutes. Irreversible. |
| `scripts/backfill_phase2.py` | Backfill source hashes, tags, graph links, and parent relationships. | Requires `--apply`; refuses to run without a dump in `data/backups/` newer than 24h. |
| `scripts/cleanup_knowledge.py` | Re-analyze existing knowledge and embeddings. | Run `--dry-run` first; back up PostgreSQL before `--apply`. |
| `scripts/migrate_vllm_embeddings.py` | Re-embed existing documents for the 1024-dimensional vLLM column. | Run validation until missing and wrong dimensions are zero; do not run `--cutover` without a backup and explicit approval. |
| `scripts/relearn_semantic.py` | Rebuild the derived semantic layer for existing sources. | Run `--dry-run` first; back up PostgreSQL before `--apply`. |

## Safe Workspace Cleanup

The following are reproducible developer artifacts and can be removed while the application is stopped or running:

```bash
rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage
find . -path './.venv' -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
find . -path './.venv' -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
```

Never include `.venv`, `.env`, `vault/`, `data/`, `logs/`, `models/`, PostgreSQL storage, or installed systemd unit files in this command. The virtual environment is currently referenced by absolute path in `config/*.service`; replace and validate it before removing it.

## Validation

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check api core services scripts tests
.venv/bin/python -m mypy core/config.py
```

Most of the suite is mocked by design. `tests/integration/` additionally runs the retrieval and cascade SQL against a real PostgreSQL, inside a `braind_test` schema that is created and dropped per session — the application tables are never touched. Those tests skip themselves when no database is reachable, and point at `BRAIND_TEST_DATABASE_URL` when set.

Neither layer proves a live vLLM. Run the health checks above before accepting a deployment.

## Database Migrations

```bash
.venv/bin/python -m alembic current
.venv/bin/python -m alembic upgrade head
```

Take a `pg_dump -Fc` backup into `data/backups/` before any migration. See `docs/schema-evolution.md`.
