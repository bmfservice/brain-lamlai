---
title: Braind Server - du an chinh
type: project
tags: [braind, ai, python, project]
status: in-progress
date: 2026-08-17
---

# Du an Braind Server

## Muc tieu

Xay bo nao so local-first: ingest tu Obsidian, file, RSS va web; tim kiem hybrid; agent chat; writer duoc ho tro boi RAG; web learning co audit history.

## Trang thai nen tang

- [x] PostgreSQL + pgvector + pg_trgm
- [x] Local vLLM: Qwen 7B AWQ cho chat va BGE-M3 1024 chieu cho embedding
- [x] Core: config, database, VLLM client, chunking, typed errors, logging
- [x] Services: ingest, RAG, chat, agent, writer, learning, scheduler
- [x] FastAPI API, SSE va built-in UI `/ui`
- [x] Systemd templates cho hai vLLM server va FastAPI
- [ ] Hoan tat writer flow: title hint -> Braind title -> content -> global web similarity -> structured ingest
- [ ] Nang cap auto-learning, SEO learning va structure-aware ingestion theo ke hoach rieng
- [ ] Dua schema changes sang migration history versioned truoc khi co DDL khong cong them

## Tai lieu lien quan

- [[braind-architecture]]
- [[rag-hybrid-search]]
- `docs/architecture.md`
- `docs/operations.md`
- `docs/schema-evolution.md`
