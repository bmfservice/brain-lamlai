"""Settings cache — in-memory TTL caching with invalidation (Brain Second Phase 1c).

Provides high-performance in-memory caching for raw and resolved settings values
to reduce DB load on hot execution paths, with instantaneous key-level or global
invalidation when setting values mutate.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_expired(self, now: float) -> bool:
        return now > self.expires_at


class SettingsCache:
    """In-memory cache for setting definitions, raw values, and resolved hierarchies."""

    def __init__(self, default_ttl_seconds: float = 60.0) -> None:
        self.default_ttl = float(default_ttl_seconds)
        self._entries: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    def _now(self) -> float:
        return time.monotonic()

    def get(self, key: str) -> Any | None:
        """Get cached value by key. Returns None if absent or expired."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        now = self._now()
        if entry.is_expired(now):
            self._entries.pop(key, None)
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        """Cache a value for key with TTL."""
        ttl = float(ttl_seconds) if ttl_seconds is not None else self.default_ttl
        expires_at = self._now() + ttl
        self._entries[key] = _CacheEntry(value, expires_at)

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache.

        If key is None, invalidates all cached entries.
        If key is a string, invalidates exact matches and any compound keys
        starting with `key:` or `key#`.
        """
        if key is None:
            count = len(self._entries)
            self._entries.clear()
            logger.debug("settings.cache_cleared", cleared_entries=count)
            return

        prefix_colon = f"{key}:"
        prefix_hash = f"{key}#"
        keys_to_remove = [
            k for k in self._entries
            if k == key or k.startswith(prefix_colon) or k.startswith(prefix_hash)
        ]
        for k in keys_to_remove:
            self._entries.pop(k, None)

        logger.debug(
            "settings.cache_invalidated",
            key=key,
            removed_entries=len(keys_to_remove),
        )

    def clear(self) -> None:
        """Clear all entries."""
        self.invalidate(None)

    def __len__(self) -> int:
        """Return number of unexpired entries."""
        now = self._now()
        return sum(1 for e in self._entries.values() if not e.is_expired(now))
