"""Tests for Phase 3: Verification Engine (tests/test_verification.py).

Verifies Layer 1 Deterministic Validator & Layer 2 LLM Second-Pass Reviewer.
Includes Test Scenario #3: Verification Engine catches hallucinated citations or non-existent files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from services.verification.engine import (
    VerificationService,
    VerificationStatus,
)

# ----------------------------------------------------------------------
# Scenario #3: Verification Engine catches hallucinated citations or non-existent files
# ----------------------------------------------------------------------


def test_scenario_3_catches_hallucinated_citations():
    """Scenario #3 (Part A): Deliverable references non-existent RAG citation."""
    verifier = VerificationService()

    # Legitimate RAG chunks retrieved by system
    real_rag_chunks = [
        {"id": "chunk_doc1_0", "text": "Market analysis shows 15% growth in Q3."},
        {"id": "chunk_doc1_1", "text": "Competitor X launched a new product line."},
    ]

    # Case A: Hallucinated citation chunk_fake_99
    deliverable_with_hallucination = {
        "summary": "Growth is expected to continue.",
        "citations": ["chunk_doc1_0", "chunk_fake_99"],
    }

    result = verifier.verify(
        deliverable=deliverable_with_hallucination,
        rag_chunks=real_rag_chunks,
    )

    assert not result.is_valid
    assert result.status == VerificationStatus.UNVERIFIED
    assert len(result.critical_errors) >= 1
    hallucinated = [e for e in result.critical_errors if e.type == "hallucinated_citation"]
    assert len(hallucinated) == 1
    assert "chunk_fake_99" in hallucinated[0].message


def test_scenario_3_catches_non_existent_files():
    """Scenario #3 (Part B): Deliverable references a file path that does not exist on disk."""
    verifier = VerificationService()

    non_existent_path = "D:/AI/braind/non_existent_directory_xyz/ghost_file.json"

    deliverable = {
        "summary": "Processed output saved to local disk.",
        "files": [non_existent_path],
    }

    result = verifier.verify(
        deliverable=deliverable,
        referenced_files=[non_existent_path],
    )

    assert not result.is_valid
    assert result.status == VerificationStatus.UNVERIFIED
    missing_file_errors = [e for e in result.critical_errors if e.type == "missing_file"]
    assert len(missing_file_errors) >= 1
    assert "ghost_file.json" in missing_file_errors[0].message


def test_scenario_3_passes_when_citations_and_files_are_genuine():
    """Scenario #3 (Part C): Deliverable with valid citations and real files passes Layer 1."""
    verifier = VerificationService()

    # Create real temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        real_file = tmp.name

    try:
        real_chunks = [{"id": "chunk_001", "text": "Valid RAG context"}]
        deliverable = {
            "summary": "Valid deliverable with authentic citations and existing file.",
            "citations": ["chunk_001"],
            "files": [real_file],
        }

        result = verifier.verify(
            deliverable=deliverable,
            rag_chunks=real_chunks,
            referenced_files=[real_file],
        )

        assert result.is_valid
        assert result.status in (
            VerificationStatus.VERIFIED,
            VerificationStatus.VERIFIED_WITH_HIGH_CONFIDENCE,
        )
        assert len(result.critical_errors) == 0
        assert result.layer1.passed
    finally:
        if Path(real_file).exists():
            Path(real_file).unlink()


# ----------------------------------------------------------------------
# Additional Layer 1: Schema Validation
# ----------------------------------------------------------------------


def test_json_schema_validation_catches_missing_fields():
    """Verify that JSON schema validation detects missing required fields."""
    verifier = VerificationService()

    schema = {
        "required": ["report_title", "findings", "score"],
        "properties": {
            "report_title": {"type": "string"},
            "findings": {"type": "array"},
            "score": {"type": "number"},
        },
    }

    # Missing "score" and "findings" is wrong type
    bad_payload = {
        "report_title": "Q3 Audit",
        "findings": "Not an array, a string",
    }

    valid, errors = verifier.verify_json_schema(bad_payload, schema)
    assert not valid
    error_fields = {e.field_name for e in errors}
    assert "score" in error_fields
    assert "findings" in error_fields


# ----------------------------------------------------------------------
# Layer 2: Completeness & Reasoning Coherence
# ----------------------------------------------------------------------


def test_layer2_evaluates_completeness_and_coherence():
    """Verify Layer 2 evaluator checks empty/sparse output and assigns appropriate score."""
    verifier = VerificationService()

    # Empty deliverable
    res_empty = verifier.verify_layer2("", objective="Write a comprehensive guide")
    assert not res_empty.passed
    assert res_empty.completeness_score == 0.0

    # Substantial deliverable
    res_good = verifier.verify_layer2(
        "A detailed deliverable analyzing market trends and strategic recommendations.",
        objective="Write a comprehensive guide",
    )
    assert res_good.passed
    assert res_good.completeness_score >= 0.8
    assert res_good.coherence_score >= 0.8


async def test_layer2_custom_llm_reviewer():
    """Verify Layer 2 integrates custom LLM reviewer callable when supplied."""
    async def mock_reviewer(prompt: str) -> str:
        return (
            '{"completeness_score": 0.95, "coherence_score": 0.90, '
            '"feedback": "Solid work", "issues": []}'
        )

    verifier = VerificationService(reviewer_llm=mock_reviewer)
    res = await verifier.verify_layer2_async(
        "Some detailed content",
        objective="Analyze quarterly metrics",
    )

    assert res.passed
    assert res.completeness_score == 0.95
    assert res.coherence_score == 0.90
    assert res.feedback == "Solid work"
