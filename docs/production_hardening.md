# Production Hardening (Phase 17)

This guide covers the hardening steps for running Braind in production on the
RTX 3060 12GB machine.

## Asset isolation

- Models are cached under `braind/models/hf` instead of the global
  `~/.cache/huggingface` directory. This makes backups portable and keeps the
  project self-contained.
- Download models once with:
  ```bash
  .venv/bin/python scripts/download_models.py
  ```
- The systemd units set `HF_HOME` to the local `models/hf` directory.

## Service hardening

- vLLM chat and embedding services run under separate user units and restart on
  failure (`Restart=on-failure`, `RestartSec=5`).
- The main Braind service starts after both vLLM units are available.
- VRAM budget on the 12GB card: chat runs `--max-model-len 16384 --max-num-seqs 2
  --gpu-memory-utilization 0.7`, embeddings run `--max-model-len 8192
  --gpu-memory-utilization 0.25`. Both figures are measured against the running
  deployment; raising either one OOMs once both models are resident.
- `braind.service` sets `NoNewPrivileges`, `PrivateTmp`, `ProtectSystem=full`,
  `ProtectControlGroups`, `ProtectKernelTunables` and `RestrictSUIDSGID`.
- Start the embedding unit before the chat unit: the 7B model claims the larger
  share of VRAM, and starting it first can leave too little headroom for the
  embedding model on the 12GB card.
- The API's `ExecStartPre` polls both vLLM `/v1/models` endpoints (up to ten
  minutes) rather than probing once — vLLM binds its port well before the model
  finishes loading, so a single probe would fail on every boot.

## Configuration

- All runtime behaviour is config-driven via `.env` and
  `core.config.settings`.
- Do not hard-code paths in service logic; use `settings.*` values.
- Sensitive values (`DATABASE_URL`, API keys) live only in `.env` and are never
  logged.

## Security

- **There is no API authentication layer.** `HOST` and the systemd unit bind
  `127.0.0.1` by default, and `CORS_ALLOW_ORIGINS` is a loopback whitelist.
  Before exposing the API on a LAN or the internet, put an authenticating
  reverse proxy in front of it — widening the bind address alone leaves every
  endpoint, including ingest and scheduler triggers, open to anyone who can
  reach the port.
- Every outbound fetch — the web tool, RSS, the deep-learn crawler, and the
  agent's `fetch_url`/`fetch_rss` tools — goes through `core.net.fetch_text_bounded`,
  which enforces the SSRF host check (before the request and again after every
  redirect), a content-type check, a byte cap, and a timeout.
- Content containing Han ideographs in U+3400-U+4DBF or U+4E00-U+9FFF is
  rejected before ingest or web-learning persistence.
- Request bodies and private ingested content are never logged.

## Backup

- Take a database backup before any non-additive DDL or embedding cutover.
- `scripts/backfill_phase2.py` refuses to run without `--apply` and without a
  dump under `data/backups/` newer than 24 hours.
- Preserve `vault/`, `data/`, `logs/`, and `models/` as operational assets.

## Monitoring

- Use `GET /api/debug/metrics` for retrieval counters and latency histograms and
  `GET /api/search/debug?q=...` for per-stream retrieval diagnostics.
- The maintenance window runs daily at `MAINTENANCE_TIME` as the scheduler's
  `maintenance` job: duplicate detection, contradiction scan, and re-embedding of
  documents whose vector column is NULL. Trigger it manually with
  `POST /api/scheduler/run {"name": "maintenance"}`.
