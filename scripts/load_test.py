#!/usr/bin/env python3
"""Lightweight load test for the Braind API (Phase 16).

Runs concurrent HTTP requests against the local FastAPI instance and reports
latency percentiles.  Does not require a live vLLM; it can be pointed at any
running Braind endpoint.  By default it hammers ``POST /search/unified``.

Usage:
    .venv/bin/python scripts/load_test.py --url http://127.0.0.1:8000/api --duration 30 --workers 8
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time


def _now() -> float:
    return time.perf_counter()


try:
    import httpx  # type: ignore

    async def _request(client: httpx.AsyncClient, url: str, payload: dict) -> tuple[int, float]:
        start = _now()
        try:
            response = await client.post(url, json=payload, timeout=30.0)
            return response.status_code, _now() - start
        except Exception:
            return 0, _now() - start
except ImportError:
    httpx = None  # type: ignore


async def _worker(
    url: str,
    payload: dict,
    duration: float,
    results: list[tuple[int, float]],
    stop: asyncio.Event,
    start_time: float,
) -> None:
    async with httpx.AsyncClient() as client:
        while not stop.is_set():
            status, elapsed = await _request(client, url, payload)
            results.append((status, elapsed))
            if _now() - start_time > duration:
                break


async def main() -> int:
    parser = argparse.ArgumentParser(description="Braind API load test.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api", help="Base API URL.")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers.")
    parser.add_argument("--query", default="What is PostgreSQL?", help="Query payload.")
    args = parser.parse_args()

    if httpx is None:
        print("httpx is not installed; install it to run load tests.", file=sys.stderr)
        return 1

    endpoint = f"{args.url.rstrip("/")}/search/unified"
    payload = {"query": args.query, "top_k": 5}
    results: list[tuple[int, float]] = []
    stop = asyncio.Event()

    start = _now()
    tasks = [
        asyncio.create_task(_worker(endpoint, payload, args.duration, results, stop, start))
        for _ in range(args.workers)
    ]
    await asyncio.sleep(args.duration)
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = _now() - start

    if not results:
        print("No requests completed.")
        return 1

    latencies = sorted(r[1] for r in results)
    total = len(results)
    ok = sum(1 for status, _ in results if status == 200)
    errors = total - ok
    throughput = total / elapsed if elapsed else 0.0

    print("=" * 60)
    print("Braind Load Test Report")
    print("=" * 60)
    print(f"Duration (s): {elapsed:.2f}")
    print(f"Total requests: {total}")
    print(f"Successful (2xx): {ok}")
    print(f"Errors: {errors}")
    print(f"Throughput (req/s): {throughput:.2f}")
    print(f"Min latency (ms): {latencies[0] * 1000:.2f}")
    print(f"P50 latency (ms): {latencies[int(total * 0.5)] * 1000:.2f}")
    print(f"P95 latency (ms): {latencies[int(total * 0.95)] * 1000:.2f}")
    print(f"Max latency (ms): {latencies[-1] * 1000:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
