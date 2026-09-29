"""Retry/backoff tests with injected sleep. No real waiting, no network."""

from __future__ import annotations

import pytest

from core.retry import fetch_with_retry, is_retryable


class HttpishError(RuntimeError):
    def __init__(self, status=None, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        if retry_after is not None:
            self.response = type("R", (), {"headers": {"Retry-After": str(retry_after)}})()


def test_is_retryable_matrix():
    assert is_retryable(HttpishError(429))
    assert is_retryable(HttpishError(503))
    assert not is_retryable(HttpishError(404))
    assert not is_retryable(HttpishError(400))
    assert is_retryable(TimeoutError())


@pytest.mark.asyncio
async def test_retries_then_succeeds_with_backoff():
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise HttpishError(503)
        return "ok"

    result = await fetch_with_retry(fn, retries=2, base_delay=1.0, max_delay=8.0, sleep=fake_sleep)
    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert all(0 < s <= 8.0 for s in sleeps)
    assert sleeps[1] >= sleeps[0] / 2  # exponential-ish growth, jittered


@pytest.mark.asyncio
async def test_retry_after_header_honored():
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fn():
        raise HttpishError(429, retry_after=3)

    with pytest.raises(HttpishError):
        await fetch_with_retry(fn, retries=1, sleep=fake_sleep)
    assert sleeps == [3.0]


@pytest.mark.asyncio
async def test_non_retryable_single_attempt():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise HttpishError(404)

    with pytest.raises(HttpishError):
        await fetch_with_retry(fn, retries=2, sleep=lambda d: None)
    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_all_attempts_fail_raises_last():
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        raise HttpishError(500)

    with pytest.raises(HttpishError):
        await fetch_with_retry(fn, retries=2, sleep=lambda d: None)
    assert calls["n"] == 3
