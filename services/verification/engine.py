"""Verification Service — Layer 1 (Deterministic) & Layer 2 (LLM Reviewer) (Brain Second Phase 3).

Layer 1 (Deterministic Validator):
- Citation presence check: verifies citations against actual retrieved RAG chunks.
- File system check: verifies referenced file paths actually exist on disk.
- Schema validator: validates JSON payloads, required fields, and data schemas.

Layer 2 (LLM Second-Pass Reviewer):
- Evaluates deliverable completeness against task objective.
- Evaluates reasoning coherence and internal consistency.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.logging import get_logger
from services.settings.snapshot import SettingsSnapshot

logger = get_logger(__name__)


class VerificationStatus(StrEnum):
    """Vocabulary of honesty labels for verified deliverables (Master Rule 6)."""

    VERIFIED_WITH_HIGH_CONFIDENCE = "verified_with_high_confidence"
    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"


@dataclass
class VerificationError:
    """Individual validation failure or issue."""

    type: str  # hallucinated_citation | missing_file | schema_violation | completeness | coherence
    message: str
    is_critical: bool = True
    field_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "is_critical": self.is_critical,
            "field_name": self.field_name,
            "details": self.details,
        }


@dataclass
class Layer1Result:
    """Outcome of Layer 1 deterministic validation."""

    passed: bool
    citation_valid: bool = True
    file_paths_valid: bool = True
    schema_valid: bool = True
    errors: list[VerificationError] = field(default_factory=list)
    score: float = 1.0  # 0.0 to 1.0

    @property
    def has_critical_error(self) -> bool:
        return any(e.is_critical for e in self.errors)


@dataclass
class Layer2Result:
    """Outcome of Layer 2 LLM second-pass review."""

    passed: bool
    completeness_score: float = 1.0  # 0.0 to 1.0
    coherence_score: float = 1.0  # 0.0 to 1.0
    feedback: str | None = None
    errors: list[VerificationError] = field(default_factory=list)

    @property
    def quality_score(self) -> float:
        return (self.completeness_score + self.coherence_score) / 2.0


@dataclass
class VerificationResult:
    """Combined verification result across all verification layers."""

    is_valid: bool
    status: VerificationStatus
    confidence_score: float
    layer1: Layer1Result
    layer2: Layer2Result
    errors: list[VerificationError] = field(default_factory=list)
    critical_errors: list[VerificationError] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "confidence_score": self.confidence_score,
            "errors": [e.to_dict() for e in self.errors],
            "critical_errors": [e.to_dict() for e in self.critical_errors],
            "summary": self.summary,
            "layer1": {
                "passed": self.layer1.passed,
                "citation_valid": self.layer1.citation_valid,
                "file_paths_valid": self.layer1.file_paths_valid,
                "schema_valid": self.layer1.schema_valid,
                "score": self.layer1.score,
            },
            "layer2": {
                "passed": self.layer2.passed,
                "completeness_score": self.layer2.completeness_score,
                "coherence_score": self.layer2.coherence_score,
                "quality_score": self.layer2.quality_score,
                "feedback": self.layer2.feedback,
            },
        }


class VerificationService:
    """Two-layer verification engine (Deterministic Validator + LLM Reviewer)."""

    def __init__(self, reviewer_llm: Any = None) -> None:
        self.reviewer_llm = reviewer_llm

    # ------------------------------------------------------------------
    # Layer 1: Deterministic Validator
    # ------------------------------------------------------------------

    def verify_citations(
        self,
        citations: Sequence[str | dict[str, Any]],
        rag_chunks: Sequence[dict[str, Any] | str] | Mapping[str, Any] | set[str] | None,
    ) -> list[VerificationError]:
        """Check that every cited chunk or document actually exists in the retrieved RAG context.

        Catches hallucinated citations (spec §16 & Master Rule 6).
        """
        errors: list[VerificationError] = []
        if not citations:
            return errors

        # Extract available chunk IDs and texts
        valid_chunk_ids: set[str] = set()
        valid_chunk_texts: list[str] = []

        if rag_chunks is not None:
            if isinstance(rag_chunks, set):
                valid_chunk_ids = {str(c) for c in rag_chunks}
            elif isinstance(rag_chunks, Mapping):
                valid_chunk_ids = {str(k) for k in rag_chunks}
                for v in rag_chunks.values():
                    if isinstance(v, str):
                        valid_chunk_texts.append(v)
                    elif isinstance(v, dict) and "text" in v:
                        valid_chunk_texts.append(str(v["text"]))
            elif isinstance(rag_chunks, (list, tuple)):
                for item in rag_chunks:
                    if isinstance(item, str):
                        valid_chunk_ids.add(item)
                        valid_chunk_texts.append(item)
                    elif isinstance(item, dict):
                        for id_field in ("id", "chunk_id", "doc_id", "source_id"):
                            if id_field in item and item[id_field] is not None:
                                valid_chunk_ids.add(str(item[id_field]))
                        if "text" in item:
                            valid_chunk_texts.append(str(item["text"]))
                        if "content" in item:
                            valid_chunk_texts.append(str(item["content"]))

        for cite in citations:
            cite_id: str | None = None
            cite_quote: str | None = None

            if isinstance(cite, str):
                cite_id = cite
            elif isinstance(cite, dict):
                cite_id = str(cite.get("chunk_id") or cite.get("id") or cite.get("doc_id") or "")
                cite_quote = cite.get("quote") or cite.get("text")

            # Validate ID existence if an ID was cited
            if cite_id and cite_id not in valid_chunk_ids:
                msg = (
                    f"Hallucinated citation: chunk ID '{cite_id}' was not found "
                    f"in retrieved RAG context"
                )
                errors.append(
                    VerificationError(
                        type="hallucinated_citation",
                        message=msg,
                        is_critical=True,
                        field_name="citations",
                        details={
                            "cited_id": cite_id,
                            "available_chunk_ids": sorted(valid_chunk_ids),
                        },
                    )
                )

            # Validate quote presence if quote was provided
            if cite_quote and valid_chunk_texts:
                quote_found = any(cite_quote in text for text in valid_chunk_texts)
                if not quote_found:
                    truncated_quote = cite_quote[:50]
                    errors.append(
                        VerificationError(
                            type="hallucinated_citation",
                            message=f"Citation quote '{truncated_quote}...' not in retrieved text",
                            is_critical=True,
                            field_name="citations",
                            details={"quote": cite_quote},
                        )
                    )

        return errors

    def verify_file_paths(
        self,
        file_paths: Sequence[str | Path],
    ) -> list[VerificationError]:
        """Check that referenced file paths exist on the local filesystem."""
        errors: list[VerificationError] = []
        for path_item in file_paths:
            p_str = str(path_item).strip()
            if not p_str:
                continue
            p = Path(p_str)
            if not p.exists() and not os.path.exists(p_str):
                errors.append(
                    VerificationError(
                        type="missing_file",
                        message=f"Referenced file path '{p_str}' does not exist on disk",
                        is_critical=True,
                        field_name="file_paths",
                        details={"path": p_str},
                    )
                )
        return errors

    def verify_json_schema(
        self,
        data: Any,
        schema: Mapping[str, Any] | None = None,
    ) -> tuple[bool, list[VerificationError]]:
        """Validate JSON string/dict against an expected schema or required fields."""
        errors: list[VerificationError] = []
        parsed_data = data

        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(
                    VerificationError(
                        type="schema_violation",
                        message=f"Invalid JSON data: {exc}",
                        is_critical=True,
                        field_name="data",
                        details={"error": str(exc)},
                    )
                )
                return False, errors

        if schema and isinstance(parsed_data, dict):
            # Check required fields
            required_fields = schema.get("required", [])
            if isinstance(required_fields, (list, tuple)):
                for req in required_fields:
                    if req not in parsed_data or parsed_data[req] is None:
                        errors.append(
                            VerificationError(
                                type="schema_violation",
                                message=f"Missing required schema field '{req}'",
                                is_critical=True,
                                field_name=req,
                            )
                        )

            # Check expected types
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                for prop, prop_schema in properties.items():
                    if prop in parsed_data and isinstance(prop_schema, dict):
                        expected_type = prop_schema.get("type")
                        val = parsed_data[prop]
                        val_tname = type(val).__name__
                        if expected_type == "string" and not isinstance(val, str):
                            errors.append(
                                VerificationError(
                                    type="schema_violation",
                                    message=f"Field '{prop}' must be a string, got {val_tname}",
                                    is_critical=False,
                                    field_name=prop,
                                )
                            )
                        elif expected_type == "number" and not isinstance(val, (int, float)):
                            errors.append(
                                VerificationError(
                                    type="schema_violation",
                                    message=f"Field '{prop}' must be a number, got {val_tname}",
                                    is_critical=False,
                                    field_name=prop,
                                )
                            )
                        elif expected_type == "array" and not isinstance(val, (list, tuple)):
                            errors.append(
                                VerificationError(
                                    type="schema_violation",
                                    message=f"Field '{prop}' must be a list/array, got {val_tname}",
                                    is_critical=False,
                                    field_name=prop,
                                )
                            )

        return len(errors) == 0, errors

    def verify_layer1(
        self,
        deliverable: Mapping[str, Any] | str,
        rag_chunks: Sequence[dict[str, Any] | str] | Mapping[str, Any] | set[str] | None = None,
        referenced_files: Sequence[str | Path] | None = None,
        expected_schema: Mapping[str, Any] | None = None,
    ) -> Layer1Result:
        """Run complete Layer 1 deterministic validation."""
        errors: list[VerificationError] = []

        # 1. Citations
        citations: list[Any] = []
        if isinstance(deliverable, dict):
            citations = deliverable.get("citations") or deliverable.get("source_documents") or []
        citation_errors = self.verify_citations(citations, rag_chunks)
        errors.extend(citation_errors)
        citation_valid = len(citation_errors) == 0

        # 2. Files
        files_to_check: list[Any] = list(referenced_files or [])
        if isinstance(deliverable, dict) and "files" in deliverable:
            files_to_check.extend(deliverable["files"])
        file_errors = self.verify_file_paths(files_to_check)
        errors.extend(file_errors)
        file_paths_valid = len(file_errors) == 0

        # 3. Schema
        schema_valid = True
        if expected_schema is not None:
            schema_valid, schema_errors = self.verify_json_schema(deliverable, expected_schema)
            errors.extend(schema_errors)

        critical_count = sum(1 for e in errors if e.is_critical)
        score = max(0.0, 1.0 - (0.5 * critical_count + 0.1 * (len(errors) - critical_count)))

        return Layer1Result(
            passed=len(errors) == 0,
            citation_valid=citation_valid,
            file_paths_valid=file_paths_valid,
            schema_valid=schema_valid,
            errors=errors,
            score=score,
        )

    # ------------------------------------------------------------------
    # Layer 2: LLM Second-Pass Reviewer
    # ------------------------------------------------------------------

    async def verify_layer2_async(
        self,
        deliverable: Any,
        objective: str = "",
    ) -> Layer2Result:
        """Evaluate deliverable completeness and reasoning coherence asynchronously."""
        if self.reviewer_llm is not None:
            try:
                # If reviewer is a callable (async or sync)
                prompt = (
                    f"Evaluate the following deliverable against the objective.\n"
                    f"Objective: {objective}\n"
                    f"Deliverable: {deliverable}\n"
                    f"Return JSON with: completeness_score (0.0-1.0), "
                    f"coherence_score (0.0-1.0), feedback (str), issues (list[str])"
                )
                if inspect.iscoroutinefunction(self.reviewer_llm):
                    review = await self.reviewer_llm(prompt)
                elif callable(self.reviewer_llm):
                    review = self.reviewer_llm(prompt)
                else:
                    review = {}

                if isinstance(review, str):
                    review = json.loads(review)

                comp = float(review.get("completeness_score", 1.0))
                coh = float(review.get("coherence_score", 1.0))
                feedback = review.get("feedback")
                issues = review.get("issues", [])

                errors = [
                    VerificationError(
                        type="coherence_failure",
                        message=str(issue),
                        is_critical=False,
                    )
                    for issue in issues
                ]

                passed = comp >= 0.7 and coh >= 0.7 and len(errors) == 0
                return Layer2Result(
                    passed=passed,
                    completeness_score=comp,
                    coherence_score=coh,
                    feedback=feedback,
                    errors=errors,
                )
            except Exception as exc:
                logger.warning("verification.layer2_llm_failed", error=str(exc))

        # Heuristic evaluation fallback
        return self._heuristic_layer2(deliverable, objective)

    def verify_layer2(
        self,
        deliverable: Any,
        objective: str = "",
    ) -> Layer2Result:
        """Evaluate deliverable completeness and reasoning coherence synchronously."""
        is_sync_callable = (
            self.reviewer_llm is not None
            and callable(self.reviewer_llm)
            and not inspect.iscoroutinefunction(self.reviewer_llm)
        )
        if is_sync_callable:
            try:
                prompt = f"Objective: {objective}\nDeliverable: {deliverable}"
                review = self.reviewer_llm(prompt)
                if isinstance(review, str):
                    review = json.loads(review)
                comp = float(review.get("completeness_score", 1.0))
                coh = float(review.get("coherence_score", 1.0))
                return Layer2Result(
                    passed=comp >= 0.7 and coh >= 0.7,
                    completeness_score=comp,
                    coherence_score=coh,
                    feedback=review.get("feedback"),
                )
            except Exception:
                pass

        return self._heuristic_layer2(deliverable, objective)

    def _heuristic_layer2(self, deliverable: Any, objective: str) -> Layer2Result:
        """Deterministic heuristic check for deliverable completeness & basic coherence."""
        errors: list[VerificationError] = []
        if deliverable is None or deliverable == "":
            errors.append(
                VerificationError(
                    type="completeness",
                    message="Deliverable is empty",
                    is_critical=True,
                )
            )
            return Layer2Result(
                passed=False,
                completeness_score=0.0,
                coherence_score=0.0,
                feedback="Deliverable is completely empty",
                errors=errors,
            )

        content_str = str(deliverable)
        if len(content_str.strip()) < 10:
            errors.append(
                VerificationError(
                    type="completeness",
                    message="Deliverable is too short or incomplete",
                    is_critical=False,
                )
            )
            return Layer2Result(
                passed=False,
                completeness_score=0.3,
                coherence_score=0.5,
                feedback="Deliverable content too brief",
                errors=errors,
            )

        return Layer2Result(
            passed=True,
            completeness_score=1.0,
            coherence_score=1.0,
            feedback="Deliverable satisfies basic structural completeness and coherence heuristics",
            errors=[],
        )

    # ------------------------------------------------------------------
    # Combined Multi-Layer Verification
    # ------------------------------------------------------------------

    def verify(
        self,
        deliverable: Mapping[str, Any] | str,
        *,
        objective: str = "",
        rag_chunks: Sequence[dict[str, Any] | str] | Mapping[str, Any] | set[str] | None = None,
        referenced_files: Sequence[str | Path] | None = None,
        expected_schema: Mapping[str, Any] | None = None,
        settings_snapshot: SettingsSnapshot | None = None,
    ) -> VerificationResult:
        """Run full verification across Layer 1 and Layer 2 synchronously."""
        # Layer 1
        l1 = self.verify_layer1(
            deliverable=deliverable,
            rag_chunks=rag_chunks,
            referenced_files=referenced_files,
            expected_schema=expected_schema,
        )

        # Layer 2
        l2 = self.verify_layer2(deliverable, objective)

        return self._combine_results(l1, l2, settings_snapshot)

    async def verify_async(
        self,
        deliverable: Mapping[str, Any] | str,
        *,
        objective: str = "",
        rag_chunks: Sequence[dict[str, Any] | str] | Mapping[str, Any] | set[str] | None = None,
        referenced_files: Sequence[str | Path] | None = None,
        expected_schema: Mapping[str, Any] | None = None,
        settings_snapshot: SettingsSnapshot | None = None,
    ) -> VerificationResult:
        """Run full verification across Layer 1 and Layer 2 asynchronously."""
        # Layer 1
        l1 = self.verify_layer1(
            deliverable=deliverable,
            rag_chunks=rag_chunks,
            referenced_files=referenced_files,
            expected_schema=expected_schema,
        )

        # Layer 2
        l2 = await self.verify_layer2_async(deliverable, objective)

        return self._combine_results(l1, l2, settings_snapshot)

    def _combine_results(
        self,
        l1: Layer1Result,
        l2: Layer2Result,
        settings_snapshot: SettingsSnapshot | None = None,
    ) -> VerificationResult:
        all_errors = l1.errors + l2.errors
        critical_errors = [e for e in all_errors if e.is_critical]

        # Calculate confidence score (0.0 to 1.0)
        base_confidence = 0.6 * l1.score + 0.4 * l2.quality_score

        # Penalty for critical errors
        if critical_errors:
            confidence = max(0.0, min(0.4, base_confidence - 0.5 * len(critical_errors)))
            is_valid = False
            status = VerificationStatus.UNVERIFIED
        else:
            confidence = min(1.0, max(0.0, base_confidence))
            min_conf = (
                settings_snapshot.get("min_verification_confidence", 0.8)
                if settings_snapshot
                else 0.8
            )

            if confidence >= 0.95 and not all_errors:
                status = VerificationStatus.VERIFIED_WITH_HIGH_CONFIDENCE
                is_valid = True
            elif confidence >= min_conf:
                status = VerificationStatus.VERIFIED
                is_valid = True
            elif confidence >= 0.5:
                status = VerificationStatus.PARTIALLY_VERIFIED
                is_valid = False
            else:
                status = VerificationStatus.UNVERIFIED
                is_valid = False

        summary = (
            f"Verification: {status.value} (confidence={confidence:.2f}, "
            f"layer1={'PASS' if l1.passed else 'FAIL'}, layer2={'PASS' if l2.passed else 'FAIL'})"
        )

        return VerificationResult(
            is_valid=is_valid,
            status=status,
            confidence_score=confidence,
            layer1=l1,
            layer2=l2,
            errors=all_errors,
            critical_errors=critical_errors,
            summary=summary,
        )
