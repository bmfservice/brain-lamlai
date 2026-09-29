"""Edge weight decay + reuse tests (Giai đoạn 5, B4). Pure math, no DB/LLM."""

from __future__ import annotations

import math

from services.consolidation.weight import (
    compute_weight,
    decay_factor,
    reuse_factor,
)

# ---------------------------------------------------------------------------
# decay_factor — exponential freshness
# ---------------------------------------------------------------------------


def test_decay_no_idle_is_one():
    assert decay_factor(0.0, half_life_days=90.0) == 1.0


def test_decay_at_half_life_is_about_half():
    # exp(-x/x) = 1/e ≈ 0.3679 — the half-life is the *time constant*, the
    # value at exactly one half-life is 1/e. Document the real shape.
    assert decay_factor(90.0, half_life_days=90.0) == math.exp(-1.0)


def test_decay_monotonic_decreasing():
    a = decay_factor(10.0, half_life_days=90.0)
    b = decay_factor(50.0, half_life_days=90.0)
    c = decay_factor(200.0, half_life_days=90.0)
    assert a > b > c > 0.0  # always positive, never exactly zero


def test_decay_negative_idle_clamps_to_one():
    assert decay_factor(-5.0, half_life_days=90.0) == 1.0


def test_decay_nonpositive_half_life_means_no_decay():
    assert decay_factor(1000.0, half_life_days=0.0) == 1.0
    assert decay_factor(1000.0, half_life_days=-1.0) == 1.0


# ---------------------------------------------------------------------------
# reuse_factor — sub-linear reinforcement
# ---------------------------------------------------------------------------


def test_reuse_zero_is_one():
    assert reuse_factor(0) == 1.0


def test_reuse_sublinear():
    # Each extra use adds less than the last (diminishing returns).
    g1 = reuse_factor(1) - reuse_factor(0)
    g2 = reuse_factor(10) - reuse_factor(9)
    g3 = reuse_factor(100) - reuse_factor(99)
    assert g1 > g2 > g3 > 0.0


def test_reuse_matches_log1p():
    assert reuse_factor(10) == 1.0 + math.log1p(10)


def test_reuse_negative_clamps_to_one():
    assert reuse_factor(-3) == 1.0


# ---------------------------------------------------------------------------
# compute_weight — the combined law
# ---------------------------------------------------------------------------


def test_weight_fresh_reused_high_confidence():
    # Brand-new, heavily used, certain → near 1.0 but capped.
    w = compute_weight(confidence=1.0, days_unused=0.0, use_count=50, half_life_days=90.0)
    assert 0.0 < w <= 1.0
    assert w == 1.0  # confidence 1 × decay 1 × reuse>1, clamped to 1.0


def test_weight_idle_low_confidence_decays():
    fresh = compute_weight(confidence=0.8, days_unused=0.0, use_count=0, half_life_days=90.0)
    stale = compute_weight(confidence=0.8, days_unused=365.0, use_count=0, half_life_days=90.0)
    assert fresh > stale
    assert fresh == 0.8  # decay 1 × reuse 1 → exactly the confidence
    assert stale < 0.8 * math.exp(-4) + 1e-6  # ~4 time constants → ~0.0146


def test_weight_reuse_revives_idle_edge():
    # Two equally idle edges: the reused one weighs more.
    cold = compute_weight(confidence=0.6, days_unused=120.0, use_count=0, half_life_days=90.0)
    warm = compute_weight(confidence=0.6, days_unused=120.0, use_count=20, half_life_days=90.0)
    assert warm > cold


def test_weight_confidence_is_the_ceiling_for_idle_edge():
    # An unused edge can never exceed its confidence (reuse=1, decay≤1).
    for conf in (0.2, 0.5, 0.9):
        w = compute_weight(confidence=conf, days_unused=0.0, use_count=0, half_life_days=90.0)
        assert w == conf


def test_weight_always_in_unit_interval():
    for conf in (0.0, 0.3, 1.0, 1.5):  # 1.5 must clamp down
        for days in (0.0, 90.0, 1000.0):
            for uses in (0, 1, 1000):
                w = compute_weight(
                    confidence=conf, days_unused=days, use_count=uses, half_life_days=90.0
                )
                assert 0.0 <= w <= 1.0


def test_weight_zero_confidence_stays_zero():
    # Reuse and freshness cannot manufacture certainty from a zero-confidence edge.
    w = compute_weight(confidence=0.0, days_unused=0.0, use_count=999, half_life_days=90.0)
    assert w == 0.0


def test_weight_is_rounded_to_four_places():
    w = compute_weight(confidence=0.7, days_unused=37.0, use_count=3, half_life_days=90.0)
    assert w == round(w, 4)
