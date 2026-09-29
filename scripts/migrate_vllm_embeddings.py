"""Manually re-embed Braind documents with vLLM's 1024-dimension model.

This is intentionally separate from application startup. Back up PostgreSQL,
run the default resumable re-embedding pass until it reports zero pending rows,
then run with ``--cutover`` only after reviewing the validation report.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import create_engine
from core.logging import get_logger
from core.textanalysis import contains_chinese
from core.vllm import VLLMClient

logger = get_logger(__name__)


async def _pending_rows(session: AsyncSession, batch_size: int) -> Sequence[Any]:
    """Return one stable batch of active documents lacking vLLM embeddings."""
    result = await session.execute(
        text(
            "SELECT id, content FROM documents "
            "WHERE status = 'active' AND embedding_vllm_1024 IS NULL "
            "ORDER BY id LIMIT :batch_size"
        ),
        {"batch_size": batch_size},
    )
    return result.all()


async def reembed_pending(
    session: AsyncSession,
    vllm: VLLMClient,
    *,
    batch_size: int,
) -> dict[str, int]:
    """Re-embed active rows in verified, independently committed batches."""
    totals = {"embedded": 0, "quarantined": 0}
    while rows := await _pending_rows(session, batch_size):
        allowed = [row for row in rows if not contains_chinese(row.content or "")]
        blocked = [row for row in rows if contains_chinese(row.content or "")]
        for row in blocked:
            await session.execute(
                text(
                    "UPDATE documents SET status = 'cjk_quarantined', "
                    "embedding_vllm_1024 = NULL WHERE id = :id"
                ),
                {"id": row.id},
            )
        if allowed:
            embeddings = await vllm.embed_batch([str(row.content) for row in allowed])
            if len(embeddings) != len(allowed):
                raise RuntimeError("vLLM batch count changed before database update")
            for row, vector in zip(allowed, embeddings, strict=True):
                if len(vector) != settings.embedding_dimensions:
                    raise RuntimeError("vLLM returned a wrong-dimension vector")
                literal = "[" + ",".join(str(value) for value in vector) + "]"
                await session.execute(
                    text(
                        "UPDATE documents SET embedding_vllm_1024 = "
                        "CAST(:vector AS vector) WHERE id = :id"
                    ),
                    {"id": row.id, "vector": literal},
                )
        await session.commit()
        totals["embedded"] += len(allowed)
        totals["quarantined"] += len(blocked)
        logger.info("vllm_migration.batch_committed", **totals)
    return totals


async def validate_reembedding(session: AsyncSession) -> dict[str, int]:
    """Return invariant counts that must be zero before cutover."""
    result = await session.execute(
        text(
            "SELECT "
            "count(*) FILTER (WHERE status = 'active') AS active, "
            "count(*) FILTER (WHERE status = 'active' "
            "  AND embedding_vllm_1024 IS NULL) AS missing, "
            "count(*) FILTER (WHERE status = 'active' "
            "  AND vector_dims(embedding_vllm_1024) <> :dimensions) AS wrong_dimensions, "
            "count(*) FILTER (WHERE status = 'cjk_quarantined') AS quarantined "
            "FROM documents"
        ),
        {"dimensions": settings.embedding_dimensions},
    )
    row = result.one()
    return {
        "active": int(row.active),
        "missing": int(row.missing),
        "wrong_dimensions": int(row.wrong_dimensions),
        "quarantined": int(row.quarantined),
    }


async def cutover(session: AsyncSession, validation: dict[str, int]) -> None:
    """Promote verified 1024 vectors while retaining the 768 column reversibly."""
    if validation["missing"] or validation["wrong_dimensions"]:
        raise RuntimeError(f"Refusing cutover: validation failed: {validation}")
    await session.execute(text("DROP INDEX IF EXISTS idx_documents_embedding"))
    await session.execute(text("DROP INDEX IF EXISTS idx_documents_embedding_vllm_1024"))
    await session.execute(
        text("ALTER TABLE documents RENAME COLUMN embedding TO embedding_ollama_768")
    )
    await session.execute(
        text("ALTER TABLE documents RENAME COLUMN embedding_vllm_1024 TO embedding")
    )
    await session.execute(
        text(
            "CREATE INDEX idx_documents_embedding ON documents USING hnsw "
            "(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)"
        )
    )
    await session.commit()


async def main(*, batch_size: int, perform_cutover: bool) -> None:
    """Run the resumable migration and optional validated cutover."""
    engine = create_engine()
    vllm = VLLMClient()
    try:
        if not await vllm.is_embedding_available():
            raise RuntimeError("vLLM embedding endpoint is unavailable")
        async with AsyncSession(engine, expire_on_commit=False) as session:
            totals = await reembed_pending(session, vllm, batch_size=batch_size)
            validation = await validate_reembedding(session)
            print({**totals, **validation})
            if perform_cutover:
                await cutover(session, validation)
                print("Cutover complete; legacy embedding_ollama_768 was retained.")
    finally:
        await vllm.close()
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed documents with vLLM safely.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cutover", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    asyncio.run(main(batch_size=args.batch_size, perform_cutover=args.cutover))
