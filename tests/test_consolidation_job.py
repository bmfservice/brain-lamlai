"""Consolidation job tests (Giai đoạn 5, B6). Logic-level: dataset SQL is faked,
the classifier and weight law are real. No network, no live PostgreSQL."""

from __future__ import annotations

import asyncio

import services.consolidation.job as job_mod
from services.consolidation.classifier import ConsolidatedRelationClassifier
from services.consolidation.dataset import TrainingSet
from services.consolidation.features import REJECT_LABEL, featurize
from services.consolidation.job import ConsolidationReport, ConsolidationService


class _FakeSession:
    """Records commits; the dataset queries are monkeypatched around it."""

    def __init__(self):
        self.committed = False
        self.rolled_back = False

    async def execute(self, *a, **k):
        return self

    async def commit(self):
        self.committed = True

    async def rollback(self):
        self.rolled_back = True

    def scalars(self):
        return self

    def all(self):
        return []


def _fake_factory():
    sessions: list[_FakeSession] = []

    def factory():
        class _Ctx:
            async def __aenter__(self_inner):
                s = _FakeSession()
                sessions.append(s)
                return s

            async def __aexit__(self_inner, *exc):
                return False

        return _Ctx()

    return factory, sessions


def _toy_training_set(n: int = 60) -> TrainingSet:
    recipes = [
        ("DEPENDS_ON", {"structural": 0.5, "dependency": 0.9}, "DEPENDS_ON", "A depends on B"),
        ("REFERENCES", {"structural": 0.4, "explicit_reference": 0.9}, "REFERENCES", "A refs B"),
        ("POSSIBLE_RELATION", {"structural": 0.1, "entity": 0.5}, REJECT_LABEL, "A and B"),
    ]
    feats, labels, weights = [], [], []
    for i in range(n):
        pt, sig, label, sent = recipes[i % len(recipes)]
        feats.append(featurize(proposed_type=pt, signals=sig, sentence=sent))
        labels.append(label)
        weights.append(1.0)
    return TrainingSet(
        features=feats, labels=labels, weights=weights, n_rows=n, n_overrides=0
    )


# ---------------------------------------------------------------------------
# Step 1 — retrain gate
# ---------------------------------------------------------------------------


def test_retrain_skipped_below_min_samples(tmp_path, monkeypatch):
    factory, sessions = _fake_factory()
    classifier = ConsolidatedRelationClassifier(tmp_path)

    async def fake_count(session):
        return 5  # below the 20-sample gate

    monkeypatch.setattr(job_mod, "count_untrained", fake_count)
    svc = ConsolidationService(factory, classifier, min_new_samples=20)
    report = asyncio.run(svc.run())
    assert report.retrained is False
    assert report.new_samples == 5
    assert not classifier.is_ready  # never trained


def test_retrain_fires_at_threshold_and_persists(tmp_path, monkeypatch):
    factory, sessions = _fake_factory()
    classifier = ConsolidatedRelationClassifier(tmp_path)
    ts = _toy_training_set(60)

    async def fake_count(session):
        return 30  # ≥ gate

    async def fake_load(session):
        return ts

    stamped = {}

    async def fake_stamp(session, when):
        stamped["at"] = when
        return ts.n_rows

    monkeypatch.setattr(job_mod, "count_untrained", fake_count)
    monkeypatch.setattr(job_mod, "load_training_set", fake_load)
    monkeypatch.setattr(job_mod, "stamp_trained", fake_stamp)

    svc = ConsolidationService(factory, classifier, min_new_samples=20)
    report = asyncio.run(svc.run())

    assert report.retrained is True
    assert report.model_version.startswith("consolidated-v")
    assert classifier.is_ready
    assert "at" in stamped  # consumed rows were marked trained
    # The model was saved to disk so a restart loads it without retraining.
    assert (tmp_path / "consolidated_model.joblib").exists()
    # A committed session was used for the stamp.
    assert any(s.committed for s in sessions)


def test_retrain_no_classifier_is_noop(tmp_path):
    factory, _ = _fake_factory()
    svc = ConsolidationService(factory, None, min_new_samples=1)
    report = asyncio.run(svc.run())
    assert report.retrained is False
    assert report.model_version == ""


def test_retrain_single_class_marks_failed(tmp_path, monkeypatch):
    """A ledger with only one label cannot train a multiclass model; the step
    records a failure but the job still completes the other steps."""
    factory, _ = _fake_factory()
    classifier = ConsolidatedRelationClassifier(tmp_path)
    single = _toy_training_set(30)
    single = TrainingSet(
        features=single.features,
        labels=["DEPENDS_ON"] * single.n_rows,  # collapse to one class
        weights=single.weights,
        n_rows=single.n_rows,
        n_overrides=0,
    )

    async def fake_count(session):
        return 30

    async def fake_load(session):
        return single

    async def fake_stamp(session, when):
        return 0

    monkeypatch.setattr(job_mod, "count_untrained", fake_count)
    monkeypatch.setattr(job_mod, "load_training_set", fake_load)
    monkeypatch.setattr(job_mod, "stamp_trained", fake_stamp)

    svc = ConsolidationService(factory, classifier, min_new_samples=20)
    report = asyncio.run(svc.run())
    assert report.retrained is False
    assert any("retrain" in s for s in report.steps_failed)
    assert not classifier.is_ready


# ---------------------------------------------------------------------------
# Step 3 — contradiction quarantine gate
# ---------------------------------------------------------------------------


def test_quarantine_skipped_without_maintenance(tmp_path):
    factory, _ = _fake_factory()
    svc = ConsolidationService(factory, None, maintenance=None)
    report = asyncio.run(svc.run())
    assert report.quarantined == 0


def test_report_defaults():
    r = ConsolidationReport()
    assert r.retrained is False
    assert r.edges_reweighted == 0
    assert r.quarantined == 0
    assert r.steps_failed == []
