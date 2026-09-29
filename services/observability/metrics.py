"""Lightweight observability metrics (Brain Second Phase 13).

A small, dependency-light metrics registry.  In production this is intended to
feed Prometheus-style scrapers or structured logs; during tests it is fully
in-memory and resettable.

Supported metric kinds:

- ``Counter`` — monotonically increasing integer events.
- ``Histogram`` — binned latency / size observations.
- ``Gauge`` — point-in-time numeric values.

The registry is thread-safe enough for async use (single event loop) and has no
external dependencies beyond the standard library.
"""

from __future__ import annotations

import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class _Histogram:
    name: str
    unit: str
    bins: list[float]
    counts: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * (len(self.bins) + 1)

    def observe(self, value: float) -> None:
        self.values.append(value)
        for i, bound in enumerate(self.bins):
            if value <= bound:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def summary(self) -> dict:
        return {
            "unit": self.unit,
            "bins": ["<=" + str(b) for b in self.bins] + ["+inf"],
            "counts": self.counts,
            "count": len(self.values),
            "mean": statistics.mean(self.values) if self.values else 0.0,
            "p95": (sorted(self.values)[int(len(self.values) * 0.95)] if self.values else 0.0),
        }


class MetricsRegistry:
    """In-memory registry for Brain Second observability metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, _Histogram] = {}

    def counter(self, name: str, *, increment: int = 1) -> None:
        """Increment the named counter by ``increment``."""
        self._counters[name] += increment

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge to ``value``."""
        self._gauges[name] = value

    def histogram(
        self,
        name: str,
        value: float,
        *,
        bins: tuple[float, ...] = (10, 50, 100, 250, 500, 1000),
        unit: str = "ms",
    ) -> None:
        """Observe ``value`` into a histogram. Creates the histogram on first use."""
        if name not in self._histograms:
            self._histograms[name] = _Histogram(name=name, unit=unit, bins=list(bins))
        self._histograms[name].observe(value)

    def timer(self, name: str):
        """Context manager that observes elapsed time into a histogram."""
        return _Timer(self, name)

    def snapshot(self) -> dict:
        """Return the current state of all metrics as a plain dict."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {n: h.summary() for n, h in self._histograms.items()},
        }

    def reset(self) -> None:
        """Clear all metrics (useful between tests)."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


class _Timer:
    def __init__(self, registry: MetricsRegistry, name: str) -> None:
        self.registry = registry
        self.name = name
        self.start: float | None = None

    def __enter__(self) -> _Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self.start is not None:
            elapsed_ms = (time.perf_counter() - self.start) * 1000
            self.registry.histogram(self.name, elapsed_ms)
