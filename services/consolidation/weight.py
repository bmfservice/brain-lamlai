"""Edge weight decay + reuse reinforcement (Giai đoạn 5, B4).

An edge's ``weight`` is how strongly the graph asserts it. The master blueprint
wants weights to *breathe*: rise when retrieval keeps using an edge, fall when
it sits idle. That is spaced-repetition applied to knowledge — a fact you keep
touching is more alive than one you never look at.

Formula (pure, deterministic, unit-tested — the job in B6 just applies it):

    weight = clamp(confidence × decay(days_unused) × reuse(use_count), 0, 1)
    decay(d) = exp(-d / half_life_days)          # half-life in days
    reuse(u) = 1 + log(1 + u)                    # sub-linear, never explodes

Why these shapes:

- ``decay`` is exponential with a configurable half-life, so an unused edge
  halves its time-freshness every ``half_life_days`` but never hits exactly zero
  (it can always be revived by reuse or new evidence).
- ``reuse`` is logarithmic, not linear: the 100th use of an edge should not make
  it 100× heavier than the 1st. Diminishing returns keep a single hot edge from
  drowning the graph.
- ``confidence`` is the base — decay/reuse *modulate* a judgement that already
  happened, they never manufacture certainty from nothing (an edge the LLM rated
  0.3 stays capped near 0.3 even if heavily reused).

Kept dependency-free (only ``math``) so both the job and the tests import it
without pulling SQLAlchemy or numpy.
"""

from __future__ import annotations

import math


def decay_factor(days_unused: float, *, half_life_days: float) -> float:
    """Exponential freshness decay in (0, 1].

    ``days_unused`` ≤ 0 (a brand-new or just-reused edge) → 1.0 (no decay).
    A negative/zero half-life is treated as "no decay" rather than dividing by
    zero — the job passes the configured value, which is normally positive.
    """
    if half_life_days <= 0:
        return 1.0
    d = max(0.0, float(days_unused))
    return math.exp(-d / half_life_days)


def reuse_factor(use_count: int) -> float:
    """Sub-linear reinforcement from how often retrieval used the edge (≥ 1.0).

    ``use_count`` ≤ 0 → 1.0 (no reinforcement, no penalty). Log1p keeps a hot
    edge from dominating: 1 use → ~1.69, 10 → ~3.4, 100 → ~5.6.
    """
    u = max(0, int(use_count or 0))
    return 1.0 + math.log1p(u)


def compute_weight(
    *,
    confidence: float,
    days_unused: float,
    use_count: int,
    half_life_days: float,
) -> float:
    """Combine confidence, idle-decay and reuse into one weight in [0, 1].

    This is the single place the weight law lives; the consolidation job (B6)
    recomputes every active edge through it each night, so a reused edge climbs
    and an abandoned one fades — automatically, with no per-edge special-casing.
    """
    base = max(0.0, min(1.0, float(confidence)))
    decayed = base * decay_factor(days_unused, half_life_days=half_life_days)
    raw = decayed * reuse_factor(use_count)
    return round(max(0.0, min(1.0, raw)), 4)
