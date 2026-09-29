"""Relation candidate generation (Brain Second P4-09, golden CASE D).

Turns co-occurring resolved entities within a chunk into *candidate* relations
carrying multi-signal evidence — NOT final edges. Every candidate then flows
through ``services.relation.validator.RelationValidator`` which rejects
cosine-only or evidence-less candidates (§9) and downgrades unknown types to
``POSSIBLE_RELATION`` (§10).

Signal honesty rules:

- The signals here are structural, lexical, and syntactic — never "the
  embeddings are close" as the sole signal (§9/§63, CASE D).
- A verb cue counts as *dependency* evidence (weight 0.15) only when it sits
  in the same sentence span between the two mentions.  A cue found across a
  sentence boundary (e.g. "PostgreSQL is a database. PostgreSQL logo is a
  brand asset.") does **not** relate the pair — CASE D — and downgrades to
  ``POSSIBLE_RELATION`` (capped confidence, never a strong default).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.relation.validator import RelationCandidate

# Lexical cues → (proposed edge type, signal key). Ordered: first match wins.
# Syntactic cues carry `dependency` evidence (§11 weight 0.15); reference-type
# cues carry `explicit_reference` evidence (§11 weight 0.10).
_VERB_CUES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b(depends? on|requires?|needs?)\b", re.IGNORECASE), "DEPENDS_ON", "dependency"),
    (re.compile(r"\b(is a|is an|are)\b", re.IGNORECASE), "IS_A", "dependency"),
    (re.compile(r"\b(part of|component of|belongs to)\b", re.IGNORECASE), "PART_OF", "dependency"),
    (re.compile(r"\b(causes?|leads? to|results? in)\b", re.IGNORECASE), "CAUSES", "dependency"),
    (re.compile(r"\b(uses?|utilizes?|via|through)\b", re.IGNORECASE), "USES", "dependency"),
    (re.compile(r"\b(implements?|provides?)\b", re.IGNORECASE), "IMPLEMENTS", "dependency"),
    (re.compile(r"\b(succeeds?|supersedes?|replaces?)\b", re.IGNORECASE),
     "SUPERSEDES", "explicit_reference"),
    (re.compile(r"\b(references?|mentions?|cites?)\b", re.IGNORECASE),
     "REFERENCES", "explicit_reference"),
    (re.compile(r"\b(contradicts?|refutes?|disagrees? with)\b", re.IGNORECASE),
     "CONTRADICTS", "explicit_reference"),
]

# Max character gap between two mentions to count as a co-occurrence signal.
_COOCCUR_WINDOW = 160

# A sentence boundary inside the between-text means the "cue" does not relate
# the pair (CASE D) — the two facts are separate statements.
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")


@dataclass(frozen=True)
class RelationCandidateDraft:
    """A pre-validation relation candidate with its provenance."""

    source_name: str
    target_name: str
    proposed_type: str
    signals: dict[str, float]
    evidence_sources: list[str]
    sentence: str


class RelationExtractor:
    """Generate typed relation candidates from co-occurring entity mentions."""

    def __init__(self, model_version: str = "braind-relation-v1") -> None:
        self.model_version = model_version

    def candidates(
        self,
        *,
        text: str,
        mentions: list[tuple[str, int, int]],
        source_ref: str = "",
    ) -> list[RelationCandidateDraft]:
        """Return relation-candidate drafts for ordered entity ``mentions``.

        ``mentions`` = list of (name, start, end) spans in ``text``.  Two
        mentions within ``_COOCCUR_WINDOW`` chars generate a candidate; the
        text between them chooses the proposed type via verb cues.  A pair with
        NO cue (or a cue separated by a sentence boundary, CASE D) is emitted
        as ``POSSIBLE_RELATION`` (never a strong default, §10).
        """
        drafts: list[RelationCandidateDraft] = []
        ordered = sorted(mentions, key=lambda m: m[1])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a_name, a_start, a_end = ordered[i]
                b_name, b_start, b_end = ordered[j]
                if a_name == b_name:
                    continue
                gap = b_start - a_end
                if gap < 0 or gap > _COOCCUR_WINDOW:
                    continue
                between = text[a_end:b_start]
                sentence = text[a_start:b_end].strip()
                proposed, signal_key, cue_strength = self._classify(between)
                signals: dict[str, float] = {
                    # Co-occurrence proximity: closer = stronger (structural).
                    "structural": round(max(0.0, 1.0 - gap / _COOCCUR_WINDOW) * 0.6, 3),
                    # Both endpoints are resolved entities.
                    "entity": 0.5,
                }
                if cue_strength > 0:
                    signals[signal_key] = cue_strength
                    signals["lexical"] = 0.4
                evidence = [
                    f"{source_ref}:cooccur:{a_name}->{b_name}"
                ] if source_ref else [f"cooccur:{a_name}->{b_name}"]
                drafts.append(
                    RelationCandidateDraft(
                        source_name=a_name,
                        target_name=b_name,
                        proposed_type=proposed,
                        signals=signals,
                        evidence_sources=evidence,
                        sentence=sentence,
                    )
                )
        return drafts

    def to_candidates(
        self,
        drafts: list[RelationCandidateDraft],
        *,
        name_to_id: dict[str, str],
    ) -> list[RelationCandidate]:
        """Bind drafts to entity ids for the validator (drop unbound pairs).

        The draft's supporting ``sentence`` travels with the candidate as
        ``evidence_text`` — Giai đoạn 3: an edge must quote the span that
        justifies it, not just name a provenance label.
        """
        out: list[RelationCandidate] = []
        for d in drafts:
            src = name_to_id.get(d.source_name)
            tgt = name_to_id.get(d.target_name)
            if not src or not tgt or src == tgt:
                continue
            out.append(
                RelationCandidate(
                    source_id=src,
                    target_id=tgt,
                    proposed_type=d.proposed_type,
                    signals=dict(d.signals),
                    evidence_sources=list(d.evidence_sources),
                    model_version=self.model_version,
                    evidence_text=d.sentence,
                )
            )
        return out

    @staticmethod
    def _classify(between: str) -> tuple[str, str, float]:
        """Pick a proposed edge type from the linking text; POSSIBLE if none.

        A verb cue only counts when the two mentions share the sentence: a
        sentence boundary between them means the cue makes no claim about the
        pair (CASE D) → ``POSSIBLE_RELATION`` with zero cue strength.
        """
        if _SENT_BOUNDARY_RE.search(between):
            return "POSSIBLE_RELATION", "dependency", 0.0
        for pattern, edge_type, signal_key in _VERB_CUES:
            if pattern.search(between):
                return edge_type, signal_key, 0.9
        return "POSSIBLE_RELATION", "dependency", 0.0