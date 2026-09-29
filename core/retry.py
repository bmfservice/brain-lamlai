"""Retry with exponential backoff for transient outbound failures (Giai đoạn 2).

Transport-agnostic: wrap any async fetch callable. Only 429/5xx/timeout/connect
errors are retried — a 404 or 400 is a definitive answer, not a transient one.
Sleep is injectable so tests never wait real time.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import httpx

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _status_of(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _retry_after_of(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", None)
    if header is None:
        return None
    try:
        return float(header.get("Retry-After"))
    except (TypeError, ValueError):
        return None


def is_retryable(exc: BaseException) -> bool:
    """True for 429/5xx, transport timeouts and connect errors."""
    timeout_types: tuple[type[BaseException], ...] = (
        httpx.TimeoutException,
        httpx.TransportError,
        asyncio.TimeoutError,
        TimeoutError,
    )
    if isinstance(exc, timeout_types):
        return True
    status = _status_of(exc)
    return status in _RETRYABLE_STATUS


async def fetch_with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    should_retry: Callable[[BaseException], bool] = is_retryable,
) -> T:
    """Run ``fn()`` retrying transient failures with exponential backoff.

    Args:
        fn: zero-arg async callable performing one attempt.
        retries: retries AFTER the first attempt (default from settings, cap 2).
        base_delay/max_delay: backoff bounds in seconds (defaults from settings).
        sleep: injectable awaitable sleep for tests (default asyncio.sleep).
        should_retry: predicate deciding whether an exception warrants a retry.

    Raises:
        The last exception when every attempt fails or the failure is not
        retryable.
    """
    attempts = max(0, min(retries if retries is not None else settings.outbound_retry_attempts, 2))
    base = base_delay if base_delay is not None else settings.outbound_retry_base_delay
    ceiling = max_delay if max_delay is not None else settings.outbound_retry_max_delay
    do_sleep = sleep or asyncio.sleep

    last: BaseException | None = None
    for attempt in range(attempts + 1):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - retryable check decides
            last = exc
            if attempt >= attempts or not should_retry(exc):
                raise
            delay = _retry_after_of(exc)
            if delay is None:
                delay = rng_jitter(min(base * (2**attempt), ceiling))
            logger.warning(
                "retry.transient_failure",
                attempt=attempt + 1,
                next_delay=round(delay, 2),
                error=str(exc)[:120],
            )
            result = do_sleep(delay)
            if result is not None:
                await result
    raise last  # pragma: no cover - loop always returns or raises


def rng_jitter(delay: float) -> float:
    """Full jitter within [delay/2, delay] to avoid thundering herds."""
    return delay / 2 + random.random() * (delay / 2)
