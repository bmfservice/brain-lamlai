"""Consolidated classifier tests (Giai đoạn 5, B2). No network, no real LLM."""

from __future__ import annotations

import asyncio

from core.config import settings
from services.consolidation.classifier import ConsolidatedRelationClassifier
from services.consolidation.dataset import decision_label
from services.consolidation.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    REJECT_LABEL,
    featurize,
    to_vector,
)
from services.extract.relation_llm import LLMRelationVerifier
from services.extract.relations import RelationExtractor
from services.ingest.semantic import SemanticIngestService

# ---------------------------------------------------------------------------
# features.py — determinism, dimension, ordering
# ---------------------------------------------------------------------------


def test_feature_dim_matches_names():
    assert len(FEATURE_NAMES) == FEATURE_DIM


def test_featurize_is_deterministic_and_full():
    args = {
        "proposed_type": "DEPENDS_ON",
        "signals": {"structural": 0.5},
        "sentence": "A depends on B",
    }
    f1 = featurize(**args)
    f2 = featurize(**args)
    assert f1 == f2
    assert set(f1) == set(FEATURE_NAMES)
    assert f1["pt_depends_on"] == 1.0
    assert f1["pt_is_a"] == 0.0
    assert f1["has_dependency_cue"] == 0.0  # no "dependency" signal key given
    assert len(to_vector(f1)) == FEATURE_DIM


def test_featurize_clamps_signals():
    f = featurize(proposed_type="IS_A", signals={"structural": 5.0, "entity": -2.0}, sentence="x")
    assert f["sig_structural"] == 1.0
    assert f["sig_entity"] == 0.0


def test_decision_label_reject_vs_type():
    assert decision_label(supported=False, relation_type="DEPENDS_ON") == REJECT_LABEL
    assert decision_label(supported=True, relation_type="depends_on") == "DEPENDS_ON"


# ---------------------------------------------------------------------------
# classifier.py — cold start, train, predict, persist, threshold
# ---------------------------------------------------------------------------


def _toy_ledger(n: int = 60):
    """A separable toy dataset: dependency-cue → DEPENDS_ON, reference-cue →
    REFERENCES, neither → REJECT. Enough rows/signal for GBDT to learn."""
    # (proposed_type, signals, label, sentence) per class.
    recipes = [
        ("DEPENDS_ON", {"structural": 0.5, "dependency": 0.9}, "DEPENDS_ON", "A depends on B"),
        (
            "REFERENCES",
            {"structural": 0.4, "explicit_reference": 0.9},
            "REFERENCES",
            "A references B",
        ),
        ("POSSIBLE_RELATION", {"structural": 0.1, "entity": 0.5}, REJECT_LABEL, "A and B listed"),
    ]
    feats, labels = [], []
    for i in range(n):
        pt, sig, label, sent = recipes[i % len(recipes)]
        feats.append(featurize(proposed_type=pt, signals=sig, sentence=sent))
        labels.append(label)
    return feats, labels


def test_classifier_cold_returns_none(tmp_path):
    c = ConsolidatedRelationClassifier(tmp_path)
    assert not c.is_ready
    feats, _ = _toy_ledger(3)
    assert c.predict_batch(feats) == [None, None, None]


def test_classifier_train_then_predict(tmp_path):
    c = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=0.5)
    feats, labels = _toy_ledger(120)
    result = c.train(feats, labels)
    assert result.trained and result.n_classes >= 2
    assert c.is_ready and c.version.startswith("consolidated-v")
    preds = c.predict_batch(feats[:9])
    assert all(p is not None for p in preds)
    # The model should reproduce the training labels on its own data.
    hits = sum(1 for p, lab in zip(preds, labels[:9], strict=True) if p.relation_type == lab)
    assert hits >= 7


def test_classifier_skips_single_class(tmp_path):
    c = ConsolidatedRelationClassifier(tmp_path)
    feats, _ = _toy_ledger(3)
    result = c.train(feats, ["DEPENDS_ON"] * len(feats))
    assert not result.trained and result.reason == "single_class"
    assert not c.is_ready


def test_classifier_persists_and_reloads(tmp_path):
    c = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=0.5)
    feats, labels = _toy_ledger(120)
    c.train(feats, labels)
    c.save()
    # A fresh instance must load from disk (no retrain) — restart persistence.
    c2 = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=0.5)
    assert not c2.is_ready
    assert c2.load() and c2.is_ready
    assert c2.version == c.version
    reloaded = c2.predict_batch(feats[:3])[0].relation_type
    original = c.predict_batch(feats[:3])[0].relation_type
    assert reloaded == original


def test_classifier_low_confidence_defers(tmp_path):
    # Threshold above 1.0 is unreachable (proba ≤ 1.0) → everything defers.
    c = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=1.0001)
    feats, labels = _toy_ledger(120)
    c.train(feats, labels)
    preds = c.predict_batch(feats[:5])
    assert all(p is not None and not p.decided for p in preds)


# ---------------------------------------------------------------------------
# _verify_candidates 3-branch wiring (cold → LLM; warm → consolidated)
# ---------------------------------------------------------------------------


class _BatchVLLM:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def chat_json(self, messages, system=None, **kw):
        self.calls += 1
        return self.response


_TEXT = "PostgreSQL depends on the filesystem for durable storage."
_MENTIONS = [("PostgreSQL", 0, 10), ("filesystem", 25, 35)]
_NAMES = {"PostgreSQL": "id-a", "filesystem": "id-b"}


def _make_svc(verifier, classifier=None, metrics=None):
    svc = SemanticIngestService.__new__(SemanticIngestService)
    svc.relation_extractor = RelationExtractor()
    svc.relation_verifier = verifier
    svc.relation_classifier = classifier
    svc.metrics = metrics
    settings.relation_llm_enabled = True
    settings.relation_llm_max_pairs = 12
    settings.relation_llm_min_structural = 0.0
    return svc


def test_verify_cold_classifier_defers_to_llm(tmp_path):
    fake = _BatchVLLM({"items": [{
        "id": 1, "supported": True, "family": "entailment", "quote": "depends on",
        "rationale": "explicit", "confidence": 0.9,
    }]})
    cold = ConsolidatedRelationClassifier(tmp_path)  # never trained → is_ready False
    assert not cold.is_ready
    svc = _make_svc(LLMRelationVerifier(fake), classifier=cold)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert fake.calls == 1  # cold tier → the LLM still decides
    assert out[0].proposed_type == "DEPENDS_ON"


def test_verify_warm_classifier_skips_llm(tmp_path):
    # A trained, high-confidence classifier should rule WITHOUT any LLM call.
    c = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=0.3)
    feats, labels = _toy_ledger(150)
    c.train(feats, labels)
    fake = _BatchVLLM({"items": []})  # would error if used
    svc = _make_svc(LLMRelationVerifier(fake), classifier=c)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert fake.calls == 0  # the consolidated tier decided it
    assert len(out) >= 1
    assert out[0].model_version.startswith("consolidated-v")


# ---------------------------------------------------------------------------
# B5 — human override: ground truth, immediate, survives restart
# ---------------------------------------------------------------------------


def test_override_registry_roundtrip(tmp_path):
    c = ConsolidatedRelationClassifier(tmp_path)
    c.set_override("id-a", "id-b", "CONTRADICTS", 1.0)
    assert c.lookup_override("id-a", "id-b") == ("CONTRADICTS", 1.0)
    assert c.lookup_override("id-a", "id-z") is None  # different pair untouched
    assert c.override_count() == 1
    assert c.overrides() == [("id-a", "id-b", "CONTRADICTS", 1.0)]
    assert c.clear_override("id-a", "id-b") is True
    assert c.clear_override("id-a", "id-b") is False  # already gone
    assert c.override_count() == 0


def test_override_canonicalizes_uuid_case(tmp_path):
    """The same UUID in different cases must hit the same registry slot — the
    API receives path strings while semantic keys on str(entity.id)."""
    c = ConsolidatedRelationClassifier(tmp_path)
    upper = "11111111-2222-3333-4444-555555555555".upper()
    lower = upper.lower()
    c.set_override(upper, lower, "IS_A", 0.99)
    assert c.lookup_override(lower, upper) == ("IS_A", 0.99)
    assert c.override_count() == 1  # one slot, not two


def test_override_load_bulk_newest_wins(tmp_path):
    class Row:
        def __init__(self, s, t, label, conf):
            self.source_id, self.target_id, self.label, self.confidence = s, t, label, conf

    c = ConsolidatedRelationClassifier(tmp_path)
    # Rows arrive oldest→newest; the later ruling for a pair must win.
    rows = [
        Row("id-a", "id-b", "IS_A", 1.0),
        Row("id-a", "id-b", "PART_OF", 1.0),
        Row("id-c", "id-d", "SUPPORTS", 0.9),
    ]
    assert c.load_overrides(rows) == 3
    assert c.lookup_override("id-a", "id-b") == ("PART_OF", 1.0)
    assert c.lookup_override("id-c", "id-d") == ("SUPPORTS", 0.9)


def test_verify_override_beats_cold_llm(tmp_path):
    """A human ruling wins even when the classifier is cold — no LLM call.

    This is the acceptance criterion: the override takes effect on the very
    next decision, without waiting for tonight's consolidation."""
    fake = _BatchVLLM({"items": []})  # would error if consulted
    cold = ConsolidatedRelationClassifier(tmp_path)
    assert not cold.is_ready
    cold.set_override("id-a", "id-b", "CONTRADICTS", 1.0)
    svc = _make_svc(LLMRelationVerifier(fake), classifier=cold)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert fake.calls == 0  # neither tier ran — the human did
    assert out[0].proposed_type == "CONTRADICTS"
    assert out[0].model_version == "human-override"
    assert out[0].signals.get("semantic") == 1.0


def test_verify_override_reject_drops_pair(tmp_path):
    """A REJECT override suppresses the pair outright (no edge persisted)."""
    fake = _BatchVLLM({"items": []})
    cold = ConsolidatedRelationClassifier(tmp_path)
    cold.set_override("id-a", "id-b", REJECT_LABEL, 1.0)
    svc = _make_svc(LLMRelationVerifier(fake), classifier=cold)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert fake.calls == 0
    assert out == []  # suppressed — the human said there is no relation here


def test_verify_override_beats_warm_classifier(tmp_path):
    """The human outranks even a confident distilled model on the same pair."""
    c = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=0.3)
    feats, labels = _toy_ledger(150)  # would rule DEPENDS_ON
    c.train(feats, labels)
    c.set_override("id-a", "id-b", "SUPPORTS", 1.0)  # human disagrees
    fake = _BatchVLLM({"items": []})
    svc = _make_svc(LLMRelationVerifier(fake), classifier=c)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert fake.calls == 0
    assert out[0].proposed_type == "SUPPORTS"  # human wins over the model
    assert out[0].model_version == "human-override"


# ---------------------------------------------------------------------------
# Step 7 — metrics: the dashboard's data source (relation.decided_by.*)
# ---------------------------------------------------------------------------


def test_metrics_consolidated_counter(tmp_path):
    """A warm-classifier decision bumps relation.decided_by.consolidated — the
    counter whose rising share is the Phase-5 acceptance metric."""
    from services.observability.metrics import MetricsRegistry

    reg = MetricsRegistry()
    c = ConsolidatedRelationClassifier(tmp_path, confidence_threshold=0.3)
    feats, labels = _toy_ledger(150)
    c.train(feats, labels)
    fake = _BatchVLLM({"items": []})
    svc = _make_svc(LLMRelationVerifier(fake), classifier=c, metrics=reg)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    snap = reg.snapshot()["counters"]
    assert snap.get("relation.decided_by.consolidated", 0) >= 1
    assert snap.get("relation.decided_by.llm", 0) == 0


def test_metrics_llm_and_rejected_counters(tmp_path):
    """A cold tier routes to the LLM; a reject bumps the rejected counter too."""
    from services.observability.metrics import MetricsRegistry

    reg = MetricsRegistry()
    cold = ConsolidatedRelationClassifier(tmp_path)  # never trained
    fake = _BatchVLLM({"items": [{
        "id": 1, "supported": False, "family": "none", "quote": "",
        "rationale": "unrelated", "confidence": 0.1,
    }]})
    svc = _make_svc(LLMRelationVerifier(fake), classifier=cold, metrics=reg)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    snap = reg.snapshot()["counters"]
    assert snap.get("relation.decided_by.llm", 0) >= 1
    assert snap.get("relation.decided_by.rejected", 0) >= 1
    assert snap.get("relation.decided_by.consolidated", 0) == 0


def test_metrics_override_counter(tmp_path):
    """A human override bumps its own counter so the dashboard can show it."""
    from services.observability.metrics import MetricsRegistry

    reg = MetricsRegistry()
    cold = ConsolidatedRelationClassifier(tmp_path)
    cold.set_override("id-a", "id-b", "CONTRADICTS", 1.0)
    fake = _BatchVLLM({"items": []})
    svc = _make_svc(LLMRelationVerifier(fake), classifier=cold, metrics=reg)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    snap = reg.snapshot()["counters"]
    assert snap.get("relation.decided_by.override", 0) >= 1


def test_consolidated_ratio_rises_after_training(tmp_path):
    """The headline acceptance criterion, end to end at the unit level.

    Run 1: cold classifier → every decision is the LLM's (consolidated share 0).
    Train on the decisions that run produced (simulating the nightly job).
    Run 2: the warm classifier now settles them itself → consolidated share > 0.
    This is the trend the dashboard must show climb over time.
    """
    from services.observability.metrics import MetricsRegistry

    # --- Run 1: cold. The LLM decides everything. ---
    reg1 = MetricsRegistry()
    cold = ConsolidatedRelationClassifier(tmp_path)
    llm = _BatchVLLM({"items": [{
        "id": 1, "supported": True, "family": "entailment", "quote": "depends on",
        "rationale": "explicit runtime dependency", "confidence": 0.95,
    }]})
    svc1 = _make_svc(LLMRelationVerifier(llm), classifier=cold, metrics=reg1)
    drafts1 = svc1.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    asyncio.run(svc1._verify_candidates(drafts1, name_to_id=_NAMES))
    c1 = reg1.snapshot()["counters"]
    assert c1.get("relation.decided_by.consolidated", 0) == 0
    assert c1.get("relation.decided_by.llm", 0) >= 1

    # --- Simulate tonight's retrain on accumulated decisions. ---
    feats, labels = _toy_ledger(150)
    cold.train(feats, labels)
    cold.confidence_threshold = 0.3  # warm enough to settle the familiar pair
    assert cold.is_ready

    # --- Run 2: warm. The consolidated tier now decides. ---
    reg2 = MetricsRegistry()
    llm2 = _BatchVLLM({"items": []})  # would error if consulted
    svc2 = _make_svc(LLMRelationVerifier(llm2), classifier=cold, metrics=reg2)
    drafts2 = svc2.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    asyncio.run(svc2._verify_candidates(drafts2, name_to_id=_NAMES))
    c2 = reg2.snapshot()["counters"]
    assert llm2.calls == 0  # no LLM call — the fast tier handled it
    assert c2.get("relation.decided_by.consolidated", 0) >= 1
    assert c2.get("relation.decided_by.llm", 0) == 0
    # The ratio strictly rose from run 1 to run 2.
    assert c2.get("relation.decided_by.consolidated", 0) > c1.get(
        "relation.decided_by.consolidated", 0
    )
