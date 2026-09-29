"""Ingestion classifiers: language + freshness.

Phase 2/3 of the ingestion upgrade. Everything here is best-effort and
non-destructive:
- ``LanguageClassifier`` adds ``detected_language`` / ``language_confidence``
  to node ``meta``; ``contains_chinese`` remains the hard pre-ingest rejection.
- ``FreshnessClassifier`` stamps ``meta["freshness"]`` on semantic units and
  AGES the owning document. Heuristic by default; an optional LLM ``chat_json``
  probe only runs for ``llm_probe`` taxonomy classes when the heuristic is
  too uncertain or the method is ``llm``/``hybrid``.
"""

from __future__ import annotations

import datetime
import re

from core.config import settings
from core.logging import get_logger
from core.textanalysis import detect_language
from core.vllm import VLLMClient

logger = get_logger(__name__)

_ASCII_DIGITS = set("0123456789")

# Heuristic markers that incline a node toward TIME_SENSITIVE / REALTIME.
_CHANGE_MARKERS = (
    "updated", "released", "version", "current", "as of", "hiện tại",
    "发布", "已更新", "截至", "最新",
)
# Explicit evergreen/definitional cues -> STATIC.
_STATIC_MARKERS = (
    "definition", "is defined as", "means", "refers to", "là định nghĩa",
    "được định nghĩa", "là khái niệm", "evergreen",
)

# Time-sensitivity buckets (ascending).
FRESHNESS_BUCKETS = ("STATIC", "LONG_LIVED", "TIME_SENSITIVE", "REALTIME")


class LanguageClassifier:
    """Detect the language of an ingested text.

    Heuristic first; only when heuristic confidence is below the configured
    threshold and an LLM is available does it consult ``chat_json``. The result
    is a hint for ``meta`` — never a hard gate on its own.
    """

    # Language -> known date / number token hints (heuristic boosters).
    _LANG_HINTS: dict[str, re.Pattern[str]] = {
        "vi": re.compile(r"\b(tiếng việt|hôm nay|ngày|tháng|hiện tại|vào năm)\b", re.I),
        "en": re.compile(r"\b(today|current|as of|released|updated)\b", re.I),
    }

    def __init__(self, vllm: VLLMClient | None = None) -> None:
        self.vllm = vllm

    async def classify(self, text: str) -> tuple[str, float]:
        """Return ``(iso_code, confidence)``, upgrading to LLM when uncertain."""
        iso, conf = detect_language(text)
        if conf >= settings.language_heuristic_confidence_threshold:
            return iso, conf

        # Heuristic is ambiguous (mostly Latin, low stopword coverage) — try hints.
        sample = text[:2000].lower()
        best_iso, best_score = iso, conf
        for lang, pattern in self._LANG_HINTS.items():
            hits = len(pattern.findall(sample))
            if hits:
                boosted = conf + 0.2 * min(hits, 3)
                if boosted > best_score:
                    best_iso, best_score = lang, boosted
        if best_score >= settings.language_heuristic_confidence_threshold:
            return best_iso, min(0.95, best_score)

        # Still ambiguous and a vLLM is available: structured probe.
        if self.vllm is not None and await self.vllm.is_llm_available():
            try:
                out = await self.vllm.chat_json(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Detect the primary language of this text. "
                                f"Return only {{\"language\": \"<iso-639-1>\", "
                                f"\"confidence\": <0-1>}}.\n\n{text[:1200]}"
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=40,
                )
                if out and isinstance(out, dict) and out.get("language"):
                    return str(out["language"]).lower(), float(out.get("confidence", 0.8))
            except Exception as exc:
                logger.warning("ingest.language_llm_failed", error=str(exc))

        return iso, conf


class FreshnessClassifier:
    """Classify a knowledge node into a freshness bucket.

    Heuristic buckets (default method) are deterministic and safe for the
    5-minute vault-sync cadence:
      - explicit time window open to ``now`` (any)          -> REALTIME
      - 4-digit year / explicit date window                 -> TIME_SENSITIVE
      - change markers                                      -> TIME_SENSITIVE
      - evergreen/definitional cues                         -> STATIC
      - otherwise                                          -> LONG_LIVED
    When ``method in {\"llm\", \"hybrid\"}`` and the heuristic is unsure for an
    ``llm_probe`` class, ``chat_json`` decides.
    """

    def __init__(self, vllm: VLLMClient | None = None) -> None:
        self.vllm = vllm

    def classify_heuristic(
        self,
        text: str,
        *,
        has_open_window: bool = False,
        has_year: bool = False,
    ) -> tuple[str, float, list[str]]:
        """Deterministic freshness bucket, no IO."""
        reasons: list[str] = []
        sample = text[:3000].lower()

        if has_open_window:
            reasons.append("open temporal window (valid until now)")
            return "REALTIME", 0.9, reasons

        if has_year:
            reasons.append("explicit 4-digit year / date window")
            return "TIME_SENSITIVE", 0.75, reasons

        if any(marker in sample for marker in _CHANGE_MARKERS):
            reasons.append("change markers present")
            return "TIME_SENSITIVE", 0.7, reasons

        if any(marker in sample for marker in _STATIC_MARKERS):
            reasons.append("definitional/evergreen cues")
            return "STATIC", 0.8, reasons

        # Heuristic -> highest bucket with sensible bounding.
        reasons.append("no time or change signal; defaults to long-lived")
        return "LONG_LIVED", 0.55, reasons

    async def classify(
        self,
        text: str,
        *,
        has_temporal: datetime.datetime | tuple | bool = False,
        method: str | None = None,
    ) -> tuple[str, float, list[str]]:
        """Heuristic classification, optionally upgraded by LLM for LLM-probe classes."""
        method = method or settings.freshness_method
        has_open_window = False
        has_year = False
        if has_temporal is True:
            has_open_window = True
        elif isinstance(has_temporal, tuple) and len(has_temporal) >= 1:
            has_open_window = has_temporal[0] is None
            has_year = True
        elif isinstance(has_temporal, datetime.datetime):
            # A bare datetime is treated as a bounded window (contains a date).
            has_year = True

        cls, conf, reasons = self.classify_heuristic(
            text, has_open_window=has_open_window, has_year=has_year
        )

        needs_llm = method in ("llm", "hybrid") and cls in ("TIME_SENSITIVE", "REALTIME")
        needs_llm = needs_llm or (
            method == "hybrid" and conf < settings.language_heuristic_confidence_threshold
        )
        if needs_llm and self.vllm is not None and await self.vllm.is_llm_available():
            try:
                out = await self.vllm.chat_json(
                    [
                        {
                            "role": "user",
                            "content": (
                                "Classify how often this claim changes. "
                                "Choose one of STATIC | LONG_LIVED | TIME_SENSITIVE | REALTIME. "
                                f'Return only {{"freshness": "<CLASS>", "confidence": <0-1>, '
                                f'"reason": "<short>"}}.\n\n{text[:1200]}'
                            ),
                        }
                    ],
                    temperature=0.0,
                    max_tokens=60,
                )
                if out and isinstance(out, dict) and out.get("freshness"):
                    chosen = str(out["freshness"]).upper()
                    if chosen in FRESHNESS_BUCKETS:
                        reasons.append(f"llm: {out.get('reason', '')}")
                        return chosen, min(0.98, float(out.get("confidence", 0.85))), reasons
            except Exception as exc:
                logger.warning("ingest.freshness_llm_failed", error=str(exc))

        return cls, conf, reasons

    @staticmethod
    def _bump_bucket(cls: str, delta: int) -> str:
        try:
            idx = FRESHNESS_BUCKETS.index(cls)
        except ValueError:
            return cls
        idx = max(0, min(len(FRESHNESS_BUCKETS) - 1, idx + delta))
        return FRESHNESS_BUCKETS[idx]