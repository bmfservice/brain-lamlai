"""Evidence-based relation linking tests (Giai đoạn 3). No network, no LLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.config import settings
from core.quality import assess
from core.readability import extract_article
from services.extract.relation_llm import LLMRelationVerifier
from services.extract.relations import RelationExtractor
from services.ingest.semantic import SemanticIngestService
from services.relation.validator import RelationValidator

# ---------------------------------------------------------------------------
# Lexical extractor → candidate carries the quoted sentence
# ---------------------------------------------------------------------------


def test_draft_sentence_travels_to_candidate():
    ex = RelationExtractor()
    drafts = ex.candidates(
        text="PostgreSQL depends on the filesystem for durable storage.",
        mentions=[("PostgreSQL", 0, 10), ("filesystem", 25, 35)],
        source_ref="doc1",
    )
    candidates = ex.to_candidates(
        drafts, name_to_id={"PostgreSQL": "a", "filesystem": "b"}
    )
    assert candidates[0].evidence_text  # quoted span present
    assert "PostgreSQL" in candidates[0].evidence_text


# ---------------------------------------------------------------------------
# Validator: rationale override + evidence_text propagation
# ---------------------------------------------------------------------------


def test_validator_prefers_candidate_rationale():
    v = RelationValidator()
    candidate = RelationValidator.__module__  # noqa: F841 - placeholder
    from services.relation.validator import RelationCandidate

    cand = RelationCandidate(
        source_id="a",
        target_id="b",
        proposed_type="DEPENDS_ON",
        signals={"entity": 1.0, "structural": 0.9, "lexical": 0.6, "dependency": 0.9},
        evidence_sources=["doc:cooccur:X->Y"],
        evidence_text="X depends on Y for storage.",
        rationale="the sentence states a hard runtime dependency",
    )
    verdict = v.validate(cand)
    assert verdict.accepted
    assert verdict.reason == "the sentence states a hard runtime dependency"
    assert verdict.evidence_text == "X depends on Y for storage."


# ---------------------------------------------------------------------------
# LLM verifier: JSON coercion + family mapping + outage degrade
# ---------------------------------------------------------------------------


@dataclass
class FakeVLLM:
    payload: dict | None = None
    fail: bool = False

    async def chat_json(self, messages, system=None, **kw):
        if self.fail:
            raise RuntimeError("vllm down")
        if self.payload is None:
            return {"raw": "not json"}
        return self.payload


def test_judge_maps_entailment_family():
    v = LLMRelationVerifier(FakeVLLM(payload={
        "supported": True, "family": "entailment", "quote": "depends on",
        "rationale": "hard dependency stated", "confidence": 0.9,
    }))
    j = asyncio.run(
        v.judge(sentence="A depends on B", source_name="A", target_name="B", proposed="DEPENDS_ON")
    )
    assert j is not None and j.supported and j.relation_type == "DEPENDS_ON"


def test_judge_maps_contradiction_family():
    v = LLMRelationVerifier(FakeVLLM(payload={
        "supported": True, "family": "contradiction", "quote": "unlike B",
        "rationale": "direct opposition", "confidence": 0.8,
    }))
    j = asyncio.run(
        v.judge(sentence="A is unlike B", source_name="A", target_name="B", proposed="CONTRADICTS")
    )
    assert j.relation_type == "CONTRADICTS"


def test_judge_rejects_unsupported():
    v = LLMRelationVerifier(FakeVLLM(payload={
        "supported": False,
        "family": "none",
        "quote": "",
        "rationale": "unrelated",
        "confidence": 0.1,
    }))
    j = asyncio.run(
        v.judge(sentence="A and B are listed.", source_name="A", target_name="B", proposed="IS_A")
    )
    assert j is not None and not j.supported


def test_judge_outage_returns_none():
    v = LLMRelationVerifier(FakeVLLM(fail=True))
    judgment = asyncio.run(
        v.judge(sentence="s", source_name="A", target_name="B", proposed="IS_A")
    )
    assert judgment is None


def test_judge_unparseable_returns_none():
    v = LLMRelationVerifier(FakeVLLM(payload=None))
    judgment = asyncio.run(
        v.judge(sentence="s", source_name="A", target_name="B", proposed="IS_A")
    )
    assert judgment is None


# ---------------------------------------------------------------------------
# judge_batch (Giai đoạn 5, B3): one call, id-aligned, per-item degrade
# ---------------------------------------------------------------------------


class _BatchVLLM:
    """Records every chat_json call so tests can assert call counts."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def chat_json(self, messages, system=None, **kw):
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        resp = self._responses[idx]
        if resp == "fail":
            raise RuntimeError("vllm down")
        return resp


def test_judge_batch_one_call_aligned_by_id():
    fake = _BatchVLLM([{"items": [
        {"id": 2, "supported": True, "family": "ontology", "quote": "is a",
         "rationale": "kind of", "confidence": 0.8},
        {"id": 1, "supported": True, "family": "entailment", "quote": "causes",
         "rationale": "A causes B", "confidence": 0.9},
    ]}])
    v = LLMRelationVerifier(fake)
    out = asyncio.run(v.judge_batch([
        ("A causes B", "A", "B", "DEPENDS_ON"),
        ("C is a D", "C", "D", "IS_A"),
    ]))
    assert fake.calls == 1  # ONE call for both pairs
    assert out[0].relation_type == "DEPENDS_ON"
    assert out[1].relation_type == "IS_A"


def test_judge_batch_missing_item_falls_back_to_single():
    # Batch answers only item 1; item 2 must be re-judged individually.
    fake = _BatchVLLM([
        {"items": [{"id": 1, "supported": True, "family": "entailment",
                    "quote": "leads to", "rationale": "r1", "confidence": 0.7}]},
        {"supported": True, "family": "contradiction", "quote": "unlike",
         "rationale": "r2", "confidence": 0.6},
    ])
    v = LLMRelationVerifier(fake)
    out = asyncio.run(v.judge_batch([
        ("A leads to B", "A", "B", "DEPENDS_ON"),
        ("C unlike D", "C", "D", "CONTRADICTS"),
    ]))
    assert fake.calls == 2  # 1 batch + 1 single fallback
    assert out[0].relation_type == "DEPENDS_ON"
    assert out[1].relation_type == "CONTRADICTS"


def test_judge_batch_unparseable_returns_all_none():
    v = LLMRelationVerifier(_BatchVLLM([{"raw": "not json"}]))
    out = asyncio.run(v.judge_batch([("s", "A", "B", "IS_A")]))
    assert out == [None]


def test_judge_batch_outage_returns_all_none():
    v = LLMRelationVerifier(_BatchVLLM(["fail"]))
    out = asyncio.run(v.judge_batch([("s", "A", "B", "IS_A")]))
    assert out == [None]


def test_judge_batch_empty_returns_empty():
    v = LLMRelationVerifier(FakeVLLM(payload={"items": []}))
    assert asyncio.run(v.judge_batch([])) == []


# ---------------------------------------------------------------------------
# _verify_candidates: refine / reject / budget / outage paths
# ---------------------------------------------------------------------------


def _make_svc(verifier, enabled=True, monkeypatch=None, classifier=None):
    svc = SemanticIngestService.__new__(SemanticIngestService)
    svc.relation_extractor = RelationExtractor()
    svc.relation_verifier = verifier
    svc.relation_classifier = classifier
    svc.metrics = None
    if monkeypatch is not None:
        monkeypatch.setattr(settings, "relation_llm_enabled", enabled)
        monkeypatch.setattr(settings, "relation_llm_max_pairs", 12)
        monkeypatch.setattr(settings, "relation_llm_min_structural", 0.0)
    else:
        settings.relation_llm_enabled = enabled
        settings.relation_llm_max_pairs = 12
        settings.relation_llm_min_structural = 0.0
    return svc


_TEXT = "PostgreSQL depends on the filesystem for durable storage."
_MENTIONS = [("PostgreSQL", 0, 10), ("filesystem", 25, 35)]
_NAMES = {"PostgreSQL": "id-a", "filesystem": "id-b"}


def test_verify_refines_type_and_rationale(monkeypatch):
    svc = _make_svc(LLMRelationVerifier(FakeVLLM(payload={
        "items": [{
            "id": 1, "supported": True, "family": "entailment", "quote": "depends on",
            "rationale": "explicit runtime dependency", "confidence": 0.95,
        }],
    })))
    drafts = svc.relation_extractor.candidates(
        text=_TEXT, mentions=_MENTIONS, source_ref="doc"
    )
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert svc._last_llm_verified == 1
    assert out[0].proposed_type == "DEPENDS_ON"
    assert out[0].rationale == "explicit runtime dependency"
    assert out[0].signals.get("semantic") == 0.95


def test_verify_drops_model_rejected_pair(monkeypatch):
    svc = _make_svc(LLMRelationVerifier(FakeVLLM(payload={
        "items": [{
            "id": 1, "supported": False, "family": "none", "quote": "",
            "rationale": "", "confidence": 0.0,
        }],
    })))
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert out == []  # rejected pair is not persisted as noise


def test_verify_outage_keeps_lexical_verdict(monkeypatch):
    svc = _make_svc(LLMRelationVerifier(FakeVLLM(fail=True)), monkeypatch=monkeypatch)
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert len(out) == 1 and out[0].evidence_text  # lexical candidate untouched


def test_verify_disabled_keeps_all(monkeypatch):
    svc = _make_svc(
        LLMRelationVerifier(FakeVLLM(fail=True)), enabled=False, monkeypatch=monkeypatch
    )
    drafts = svc.relation_extractor.candidates(text=_TEXT, mentions=_MENTIONS, source_ref="doc")
    out = asyncio.run(svc._verify_candidates(drafts, name_to_id=_NAMES))
    assert len(out) == 1


def test_verify_budget_caps_llm_calls(monkeypatch):
    calls = {"n": 0}

    class Counting(FakeVLLM):
        async def chat_json(self, messages, system=None, **kw):
            calls["n"] += 1
            return {"supported": True, "family": "ontology", "quote": "is a",
                    "rationale": "r", "confidence": 0.8}

    svc = _make_svc(LLMRelationVerifier(Counting()), monkeypatch=monkeypatch)
    monkeypatch.setattr(settings, "relation_llm_max_pairs", 1)
    text = "Alpha is a database. Beta is a cache. Gamma is a broker. Delta is a store."
    drafts = svc.relation_extractor.candidates(
        text=text,
        mentions=[("Alpha", 0, 5), ("Beta", 20, 24), ("Gamma", 40, 45), ("Delta", 60, 65)],
        source_ref="doc",
    )
    out = asyncio.run(
        svc._verify_candidates(
            drafts, name_to_id={n: f"id-{n}" for n in ("Alpha", "Beta", "Gamma", "Delta")}
        )
    )
    assert calls["n"] <= 1  # budget enforced
    assert len(out) >= 5  # the 5 unjudged candidates still pass through


# ---------------------------------------------------------------------------
# Chunking: prune=False is the gate contract (no content loss)
# ---------------------------------------------------------------------------


def test_semantic_pass_no_longer_prunes():
    import inspect

    from services.ingest import semantic

    src = inspect.getsource(semantic.SemanticIngestService.semantic_pass)
    assert "prune=False" in src


def test_adaptive_fallback_only_when_no_prose():
    """Structure-first chunking: prose documents never hit the fixed-size path."""
    from services.ingest.service import IngestService

    svc = IngestService.__new__(IngestService)
    from core.chunking import HierarchicalChunker
    from services.ingest.segment import TopicShiftSegmenter

    svc.hchunker = HierarchicalChunker(chunk_size=800)
    svc.chunker = HierarchicalChunker(chunk_size=800)
    svc.segmenter = TopicShiftSegmenter(None)  # LLM off → sentence_pack fallback
    svc.semantic = None
    text = "# Title\n\n" + " ".join(
        f"Sentence number {i} explains an important concept in detail." for i in range(60)
    )
    chunks, used_fallback = asyncio.run(svc._retrieval_chunks(text, title="t"))
    assert not used_fallback
    kinds = {c.chunk_kind for c in chunks}
    assert kinds <= {"structure", "topic_shift", "sentence_pack"}  # no "fallback"
    joined = "\n".join(c.text for c in chunks)
    report = assess(extract_article(text) or joined)
    assert report.prose_ratio > 0  # gate accepts the prose
