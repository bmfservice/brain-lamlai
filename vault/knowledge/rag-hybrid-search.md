---
title: RAG va tim kiem lai
type: knowledge
tags: [rag, ai, pgvector, fulltext-search, python]
topic: retrieval augmented generation
status: done
date: 2026-08-17
---

# Retrieval-Augmented Generation voi tim kiem lai

## Cach hoat dong

Braind lay ung vien bang ba kenh roi gop diem voi Reciprocal Rank Fusion (RRF):

1. **Vector / pgvector**: cosine similarity tren `embedding_vllm_1024`.
2. **Full-text search**: PostgreSQL `ts_rank` tren `fts_vector`.
3. **Fuzzy search**: `pg_trgm` similarity cho typo va tu gan dung.

Diem RRF la tong `1 / (60 + rank)` theo moi kenh, sau do chon `top_k` chunks cao nhat.

## Mo hinh hien tai

- Embedding: `BAAI/bge-m3` qua local vLLM, 1024 chieu.
- Chat/generation: `Qwen/Qwen2.5-7B-Instruct-AWQ` qua local vLLM.
- Vector index: HNSW cosine, co IVFFlat fallback khi khoi tao index.

## Invariants

- Generated writer content khong duoc dua vao RAG mac dinh de tranh tu nhien tinh tu sinh.
- Bat ky Han tu Trung Quoc trong cac range bi cam se bi tu choi truoc ingest.
- RAG la retrieval layer; khong bo qua validation, dedupe, hay SSRF-safe fetch o ingest/learn flow.
