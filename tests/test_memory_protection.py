"""Memory & Continuous Learning Protection Tests (Phase 4).

Covers:
- Scenario #6: Memory contradiction routes to quarantine safely without permanent data loss
  (enforcing HR-NO-PERMANENT-DELETE).
- Quarantine restoration: quarantined items can be retrieved and restored with full payload.
- Edge weight decay integration: decay_factor and reuse_factor correctly flag degraded knowledge.
- Listing and filtering of quarantined items by table and reason.
"""

from __future__ import annotations

import asyncio
import uuid

from services.maintenance.quarantine import (
    REASON_AMBIGUOUS,
    REASON_CONTRADICTION,
    REASON_DECAY,
    REASON_DUPLICATE,
    QuarantineService,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test Scenario #6: Memory Contradiction Routes to Quarantine Without Data Loss
# ---------------------------------------------------------------------------


def test_scenario_6_memory_contradiction_quarantines_without_data_loss():
    """Scenario #6: When two memory/knowledge items contradict each other,
    they are safely routed to quarantine with full payloads preserved.
    Enforces HR-NO-PERMANENT-DELETE: data loss is strictly False.
    """
    svc = QuarantineService()

    item_a = {
        "id": str(uuid.uuid4()),
        "table_name": "semantic_units",
        "content": "User prefers dark mode in all applications",
        "polarity": 1,
        "entity_ids": ["pref_theme_dark"],
    }
    item_b = {
        "id": str(uuid.uuid4()),
        "table_name": "semantic_units",
        "content": "User specifically requested light mode for reading",
        "polarity": -1,
        "entity_ids": ["pref_theme_dark"],
    }

    report = _run(
        svc.handle_data_conflict(
            item_a,
            item_b,
            conflict_type=REASON_CONTRADICTION,
            detail="Opposite polarities for user theme preference",
        )
    )

    assert report["action"] == "quarantine"
    assert report["conflict_type"] == REASON_CONTRADICTION
    assert report["data_loss"] is False  # HR-NO-PERMANENT-DELETE
    assert len(report["quarantined_items"]) == 2

    # Verify both items exist in quarantine with their original payloads
    items = _run(svc.list_quarantined(table_name="semantic_units"))
    assert len(items) == 2
    row_ids = {item["row_id"] for item in items}
    assert item_a["id"] in row_ids
    assert item_b["id"] in row_ids

    # Payloads are completely intact
    for item in items:
        assert item["payload"] is not None
        assert "content" in item["payload"]
        assert item["reason"] == REASON_CONTRADICTION


def test_quarantine_single_item_preserves_payload():
    """Direct quarantine_item call preserves full payload and reason."""
    svc = QuarantineService()
    item_id = str(uuid.uuid4())
    session_id = uuid.uuid4()
    payload = {"fact": "Project deadline is Friday", "confidence": 0.95}

    qid = _run(
        svc.quarantine_item(
            table_name="memory",
            row_id=item_id,
            reason=REASON_AMBIGUOUS,
            detail="Deadline ambiguous: multiple dates mentioned in thread",
            payload=payload,
            session_id=session_id,
        )
    )

    assert isinstance(qid, uuid.UUID)

    # Retrieve and verify
    record = _run(svc.get_quarantined(qid))
    assert record is not None
    assert record["row_id"] == item_id
    assert record["table_name"] == "memory"
    assert record["reason"] == REASON_AMBIGUOUS
    assert record["payload"] == payload
    assert record["session_id"] == str(session_id)


# ---------------------------------------------------------------------------
# Test Quarantine Restoration
# ---------------------------------------------------------------------------


def test_quarantine_restoration_restores_data():
    """Quarantined item can be restored cleanly, returning its preserved payload
    and removing the item from quarantine.
    """
    svc = QuarantineService()
    item_id = str(uuid.uuid4())
    payload = {"key": "api_endpoint", "value": "https://api.v2.test"}

    qid = _run(
        svc.quarantine_item(
            table_name="settings_override",
            row_id=item_id,
            reason=REASON_DUPLICATE,
            detail="Duplicate entry detected",
            payload=payload,
        )
    )

    # Ensure it's in quarantine
    assert len(_run(svc.list_quarantined())) == 1

    # Restore
    restored = _run(svc.restore_quarantined(qid))
    assert restored is not None
    assert restored["restored"] is True
    assert restored["row_id"] == item_id
    assert restored["table_name"] == "settings_override"
    assert restored["payload"] == payload

    # Quarantine should now be empty
    assert len(_run(svc.list_quarantined())) == 0
    assert _run(svc.get_quarantined(qid)) is None


def test_restore_nonexistent_item_returns_none():
    svc = QuarantineService()
    assert _run(svc.restore_quarantined(uuid.uuid4())) is None


# ---------------------------------------------------------------------------
# Test Listing and Filtering
# ---------------------------------------------------------------------------


def test_list_quarantined_filters_by_table_and_reason():
    svc = QuarantineService()

    _run(
        svc.quarantine_item(
            table_name="semantic_units",
            row_id="1",
            reason=REASON_CONTRADICTION,
            payload={"text": "fact 1"},
        )
    )
    _run(
        svc.quarantine_item(
            table_name="semantic_units",
            row_id="2",
            reason=REASON_DUPLICATE,
            payload={"text": "fact 2"},
        )
    )
    _run(
        svc.quarantine_item(
            table_name="knowledge_links",
            row_id="3",
            reason=REASON_DECAY,
            payload={"source": "A", "target": "B"},
        )
    )

    # Total items
    assert len(_run(svc.list_quarantined())) == 3

    # Filter by table
    units = _run(svc.list_quarantined(table_name="semantic_units"))
    assert len(units) == 2

    links = _run(svc.list_quarantined(table_name="knowledge_links"))
    assert len(links) == 1
    assert links[0]["row_id"] == "3"

    # Filter by reason
    contradictions = _run(svc.list_quarantined(reason=REASON_CONTRADICTION))
    assert len(contradictions) == 1
    assert contradictions[0]["row_id"] == "1"


# ---------------------------------------------------------------------------
# Test Edge Weight Decay Integration
# ---------------------------------------------------------------------------


def test_edge_decay_evaluation():
    """Edge decay logic properly calculates decay, reuse, and flags stale edges."""
    svc = QuarantineService()

    # Fresh edge heavily reused
    fresh = svc.evaluate_edge_decay(
        confidence=0.8,
        days_unused=0.0,
        use_count=10,
        half_life_days=30.0,
    )
    assert fresh["decay_factor"] == 1.0
    assert fresh["reuse_factor"] > 3.0
    assert fresh["weight"] == 1.0  # Clamped to 1.0
    assert fresh["needs_review"] is False
    assert fresh["should_quarantine"] is False

    # Edge abandoned for 120 days (4 half-lives) with zero use
    stale = svc.evaluate_edge_decay(
        confidence=0.5,
        days_unused=120.0,
        use_count=0,
        half_life_days=30.0,
        review_threshold=0.1,
    )
    assert stale["decay_factor"] < 0.05
    assert stale["reuse_factor"] == 1.0
    assert stale["weight"] < 0.1
    assert stale["needs_review"] is True
    assert stale["should_quarantine"] is True


def test_quarantine_with_db_mock():
    """Test quarantine with a mock session factory mimicking SQLAlchemy."""

    class FakeSession:
        def __init__(self):
            self.added = []
            self.committed = False
            self.executed = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def add(self, entity):
            self.added.append(entity)

        async def execute(self, stmt):
            self.executed.append(stmt)
            # Minimal mock result
            class MockResult:
                def scalars(self):
                    class MockScalars:
                        def all(self):
                            return []
                    return MockScalars()
                def scalar_one_or_none(self):
                    return None
            return MockResult()

        async def commit(self):
            self.committed = True

    fake_session = FakeSession()

    def session_factory():
        return fake_session

    svc = QuarantineService(session_factory=session_factory)
    qid = _run(
        svc.quarantine_item(
            table_name="semantic_units",
            row_id="row-123",
            reason=REASON_CONTRADICTION,
            detail="test contradiction",
            payload={"content": "test"},
        )
    )

    assert isinstance(qid, uuid.UUID)
    assert len(fake_session.added) == 1
    assert fake_session.committed is True
    assert fake_session.added[0].row_id == "row-123"
    assert fake_session.added[0].reason == REASON_CONTRADICTION
