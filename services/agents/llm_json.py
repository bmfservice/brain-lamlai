"""Shared helpers for agent LLM calls that must return structured JSON."""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    """Best-effort parse of a JSON object from an LLM response.

    Handles markdown code fences and leading/trailing prose. Returns ``None``
    if nothing parseable is found so callers can fall back deterministically.
    """
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to the first balanced {...} or [...] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    return None
