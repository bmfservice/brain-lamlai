"""Feedback capture and hard-negative mining (Brain Second Phase 11).

Provides a small, deterministic framework for recording retrieval feedback
(clicks, up/down votes) and mining hard negatives: candidates that look
relevant to a query but are known not to be useful.

The default store is in-memory.  When a ``session_factory`` is supplied,
feedback is persisted to ``core.database.RetrievalFeedback`` rows so future
re-ranker training / evaluation can use them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sqlalchemy

if TYPE_CHECKING:
    from collections.abc import Callable

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class FeedbackRecord:
    """A single feedback event."""

    query: str
    chunk_id: str
    label: str  # click | upvote | downvote
    user_id: str | None = None
    meta: dict = field(default_factory=dict)


class FeedbackService:
    """Capture feedback and mine hard negatives without an LLM."""

    _VALID_LABELS = {"click", "upvote", "downvote"}

    def __init__(self, session_factory: Callable | None = None) -> None:
        self.session_factory = session_factory
        self._memory: list[FeedbackRecord] = []

    async def record(
        self,
        query: str,
        chunk_id: str,
        label: str,
        user_id: str | None = None,
        meta: dict | None = None,
    ) -> None:
        """Record one feedback event.

        Args:
            query: The original search query.
            chunk_id: Identifier of the chunk that received feedback.
            label: One of ``click``, ``upvote``, ``downvote``.
            user_id: Optional user identifier.
            meta: Optional metadata (e.g. rank position, source).
        """
        if label not in self._VALID_LABELS:
            raise ValueError(f"Invalid feedback label: {label}")
        record = FeedbackRecord(
            query=query.strip(),
            chunk_id=chunk_id,
            label=label,
            user_id=user_id,
            meta=meta or {},
        )
        self._memory.append(record)
        if self.session_factory is not None:
            await self._persist(record)

    async def _persist(self, record: FeedbackRecord) -> None:
        from core.database import RetrievalFeedback

        if self.session_factory is None:
            return
        # Import lazily to avoid circular imports at module load.
        session: AsyncSession
        async with self.session_factory() as session:
            session.add(
                RetrievalFeedback(
                    id=uuid.uuid4(),
                    query_text=record.query,
                    chunk_id=record.chunk_id,
                    label=record.label,
                    user_id=record.user_id,
                    meta=record.meta,
                )
            )
            try:
                await session.commit()
            except sqlalchemy.exc.IntegrityError:
                await session.rollback()

    def for_query(self, query: str) -> list[FeedbackRecord]:
        """Return all feedback records matching ``query`` (exact)."""
        query_norm = query.strip().lower()
        return [r for r in self._memory if r.query.lower() == query_norm]

    def positives(self, query: str) -> set[str]:
        """Return chunk ids with positive feedback for ``query``."""
        return {
            r.chunk_id
            for r in self.for_query(query)
            if r.label in {"click", "upvote"}
        }

    def negatives(self, query: str) -> set[str]:
        """Return chunk ids with explicit negative feedback for ``query``."""
        return {r.chunk_id for r in self.for_query(query) if r.label == "downvote"}

    def hard_negatives(
        self,
        query: str,
        candidates: list[dict],
        positive_ids: set[str] | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Mine hard negatives from ``candidates``.

        A hard negative is a candidate that:

        - is NOT in the known-positive set,
        - has high token overlap with the query (looks relevant), and
        - is NOT already marked negative (if present).

        Args:
            query: Original query.
            candidates: List of chunk dicts with at least ``chunk_id`` and ``content``.
            positive_ids: Optional explicit positive set (falls back to recorded positives).
            top_k: Maximum hard negatives to return.

        Returns:
            Up to ``top_k`` candidate dicts sorted by descending pseudo-relevance.
        """
        positives = positive_ids or self.positives(query)
        negatives = self.negatives(query)
        query_tokens = self._token_set(query)

        scored: list[tuple[float, dict]] = []
        for cand in candidates:
            cid = str(cand.get("chunk_id", ""))
            if not cid or cid in positives or cid in negatives:
                continue
            content = str(cand.get("content", ""))
            overlap = self._overlap(query_tokens, self._token_set(content))
            # Only surface candidates that look plausibly relevant but were passed over.
            if overlap <= 0:
                continue
            scored.append((overlap, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return {t.lower() for t in text.split() if len(t) > 2}

    @classmethod
    def _overlap(cls, a: set[str], b: set[str]) -> float:
        if not a:
            return 0.0
        return len(a & b) / len(a)
