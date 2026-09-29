---
title: FastAPI + Pydantic v2 ghi chú
type: reference
tags: [python, fastapi, pydantic, api]
topic: FastAPI
source: fastapi.tiangolo.com
status: done
date: 2026-08-13
---

# FastAPI notes (Pydantic v2)

## Điểm chính
- FastAPI sinh OpenAPI docs tự động tại `/docs`.
- Dùng `Annotated[T, Depends(...)]` cho dependency (recommended).
- Tất cả endpoint nên `async def` để hợp với I/O-bound (DB async, LLM async).

## Pydantic v2
- Dùng `Field(..., min_length=1)` cho bắt buộc + ràng buộc.
- `model_dump()` thay cho `dict()` cũ.
- JSONB + pydantic-settings hoạt động tốt với `extra="ignore"`.

## Với async DB
- `async_sessionmaker` + `AsyncSession` để chạy query không block event loop.