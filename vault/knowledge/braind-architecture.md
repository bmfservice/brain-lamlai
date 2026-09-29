---
title: Cau truc Braind Server
type: knowledge
tags: [braind, architecture, ai, python, postgres, vector-db]
topic: kien truc he thong
status: done
date: 2026-08-17
---

# Kien truc Braind Server

## Nguyen tac

- Khong Docker, khong n8n, khong Dify runtime; chay native tren Linux.
- PostgreSQL + pgvector la nguon du lieu ben vung; Obsidian vault la nguon Markdown ben vung.
- AI la lop tong hop: RAG + local vLLM de tra loi, research, va tao noi dung.
- Moi external web fetch phai qua WebSearchTool SSRF-safe.

## Thanh phan

| Thanh phan | Vai tro |
| --- | --- |
| PostgreSQL + pgvector + pg_trgm | sources, chunks, embeddings, full-text search, graph links, writer history, learning audit |
| vLLM chat (`Qwen/Qwen2.5-7B-Instruct-AWQ`) | sinh noi dung, agent, phan tich |
| vLLM embedding (`BAAI/bge-m3`) | embedding 1024 chieu |
| FastAPI | REST, SSE va UI tai `/ui` |
| APScheduler | RSS, vault sync va digest |

## Luong phu thuoc

`api/main.py` la composition root: khoi tao database, VLLMClient va cac service; sau do gan chung vao `app.state`.

```text
FastAPI routes -> api/deps -> app.state services
                          -> core (config, DB, vLLM, errors, logging)
                          -> services (ingest, rag, learn, writer, agent, scheduler)
```

## Ranh gioi thu muc

- `api/`: HTTP routes, schemas va static UI.
- `core/`: ha tang dung chung va agent primitives.
- `services/`: domain workflows. Service nhan dependency tu `api/main.py`.
- `scripts/`: lenh van hanh co the thay doi PostgreSQL; khong chay tu dong.
- `vault/`: du lieu Obsidian; khong xoa trong dot don dep source.
- `docs/`: tai lieu van hanh chuan cua project.

Tai lieu chi tiet: `docs/architecture.md` trong root Braind.
