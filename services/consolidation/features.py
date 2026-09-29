"""Feature engineering for the consolidated relation classifier (Giai đoạn 5).

This is the *single source of truth* for the feature vector: both the nightly
training job and the live inference path build features through
:func:`featurize`, so a row trained yesterday is comparable to a candidate
scored today. The vector is deliberately small and dense — engineered from the
candidate's lexical/structural signals, never from raw embeddings — because the
master blueprint caps server RAM/CPU and the classifier's whole point is to be
cheaper than the LLM it distils.

Design rules (kept here so the encoding never drifts):

- Fixed feature order (``FEATURE_NAMES``). A model trained on this layout must
  be re-trained if the layout changes; the classifier checks the dimension at
  predict time and defers to the LLM on any mismatch.
- Pure functions, no I/O, no randomness — identical inputs always yield an
  identical vector.
- ``REJECT`` is a first-class label: the model learns "this sentence asserts no
  relation" just as it learns each taxonomy type, mirroring the LLM's
  supported=False verdict so the consolidated tier can drop noise too.
"""

from __future__ import annotations

# Lexical proposed types emitted by RelationExtractor._classify (relations.py).
# One-hot over this fixed set; an unseen type falls through to all-zeros (the
# model then leans on the signal features).
_PROPOSED_TYPES: tuple[str, ...] = (
    "DEPENDS_ON",
    "IS_A",
    "PART_OF",
    "CAUSES",
    "USES",
    "IMPLEMENTS",
    "SUPERSEDES",
    "REFERENCES",
    "CONTRADICTS",
    "POSSIBLE_RELATION",
)

# Independent sub-scores the extractor puts in ``candidate.signals``.
_SIGNAL_KEYS: tuple[str, ...] = (
    "structural",
    "entity",
    "lexical",
    "dependency",
    "explicit_reference",
)

# Normalization ceilings — chosen so typical values land in (0, 1) without
# saturating. Sentences longer than these are clamped, not scaled further.
_MAX_SENT_CHARS = 500.0
_MAX_SENT_WORDS = 80.0

# Feature order. Anything added must be appended, never inserted, and bumps the
# stored model's expected dimension (handled by the classifier).
FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"pt_{t.lower()}" for t in _PROPOSED_TYPES)
    + tuple(f"sig_{k}" for k in _SIGNAL_KEYS)
    + (
        "sent_len_norm",
        "sent_words_norm",
        "nonzero_signal_frac",
        "has_dependency_cue",
        "has_reference_cue",
    )
)

#: Label emitted when a sentence asserts no relation (LLM supported=False).
REJECT_LABEL = "REJECT"

FEATURE_DIM = len(FEATURE_NAMES)


def featurize(
    *,
    proposed_type: str | None,
    signals: dict[str, float] | None,
    sentence: str = "",
) -> dict[str, float]:
    """Build the ordered feature dict for one candidate.

    Accepts plain values (not a ``RelationCandidate``) so this module stays
    dependency-free and unit-testable without importing the validator.
    """
    signals = signals or {}
    features: dict[str, float] = {}

    proposed = (proposed_type or "").upper()
    for name in _PROPOSED_TYPES:
        features[f"pt_{name.lower()}"] = 1.0 if name == proposed else 0.0

    for key in _SIGNAL_KEYS:
        try:
            value = float(signals.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        features[f"sig_{key}"] = max(0.0, min(1.0, value))

    text = sentence or ""
    words = text.split()
    features["sent_len_norm"] = max(0.0, min(1.0, len(text) / _MAX_SENT_CHARS))
    features["sent_words_norm"] = max(0.0, min(1.0, len(words) / _MAX_SENT_WORDS))

    nonzero = sum(
        1
        for key in _SIGNAL_KEYS
        if _safe_float(signals.get(key)) > 0.0
    )
    features["nonzero_signal_frac"] = nonzero / len(_SIGNAL_KEYS)
    features["has_dependency_cue"] = 1.0 if _safe_float(signals.get("dependency")) > 0.0 else 0.0
    features["has_reference_cue"] = (
        1.0 if _safe_float(signals.get("explicit_reference")) > 0.0 else 0.0
    )
    return features


def to_vector(features: dict[str, float]) -> list[float]:
    """Order a feature dict into the dense vector the model consumes."""
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
