"""Braind Learning — Live event hub for SSE streaming.

Bounded ``asyncio.Queue`` per run; ``emit`` pushes typed events toward
the HTTP response, ``stream`` yields them as SSE frames.
"""

from __future__ import annotations

import asyncio
import datetime
import json
from collections.abc import AsyncIterator
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)

_MAX_QUEUE = 2048


class LearnEventHub:
    """Per-run event bus for deep-learning progress.

    Usage::

        hub = LearnEventHub()
        # … in a background task …
        await hub.emit("stage", {"stage": "search", "label": "Đang tìm kiếm…"})
        await hub.emit("log", {"level": "info", "text": "Found 5 results"})
        await hub.emit("done", {})
        hub.close()

        # … in the SSE response generator …
        async for event in hub.stream():
            yield event
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=_MAX_QUEUE)
        self._closed = False
        self._terminal_event_emitted = False

    @property
    def terminal_event_emitted(self) -> bool:
        """Whether an ``error`` or ``done`` terminal event has been queued."""
        return self._terminal_event_emitted

    async def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Push one SSE frame ``event`` + ``data`` to the stream."""
        if self._closed:
            return
        try:
            self._queue.put_nowait(
                {"event": event_type, "data": json.dumps(data, ensure_ascii=False, default=str)}
            )
            if event_type in {"error", "done"}:
                self._terminal_event_emitted = True
        except asyncio.QueueFull:
            logger.warning("learn.event_queue_full", event_type=event_type)

    async def emit_log(self, level: str, text: str, **extra: Any) -> None:
        """Convenience: emit a ``log`` event with timestamp."""
        await self.emit(
            "log",
            {
                "level": level,
                "text": text,
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                **extra,
            },
        )

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield SSE-frame dicts until the hub is closed and drained."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
            self._queue.task_done()

    def close(self) -> None:
        """Signal the stream to end after all queued events are consumed."""
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            # Never crash cleanup if the bounded queue is saturated. Drop the
            # sentinel is not acceptable (stream would hang), so drain one slot.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(None)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass
