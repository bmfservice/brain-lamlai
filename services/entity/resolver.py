"""Entity resolver — canonicalization + merge decisions (Phase 6.2/6.3).

Governs when two mentions name the same entity (canonical identity) without
ever merging on raw name/embedding similarity alone (§13, golden CASE E/F).

Rules are deterministic and evidence-based:

0. **No identity key, no merge.** A name whose normalized form is empty
   ("platform", "BINARY" — every token is a generic suffix) carries no identity
   at all. It must never match another empty-normalizing name, or "system" and
   "platform" auto-merge onto each other with a confidence of 0.98.
1. **Exact identity**: mention equals a canonical name or one of its aliases
   → ``AUTO_MERGE`` onto that entity.
2. **Generic-suffix collapse**: "Postgres database" / "PostgreSQL DB" collapse
   to the base token, which is then matched against canonical/aliases
   (``AUTO_MERGE`` — this is a grammatical transform, allowed by §13).
3. **Subsumption guard**: a mention that is a strict prefix of another with
   *additional meaningful tokens* must NOT be merged. "PostgreSQL" vs
   "PostgreSQL logo", "Docker" vs "Docker Compose" stay distinct (CASE E/F);
   a *relation* hint (`RELATED_TO` / `PART_OF`) is returned instead.
4. **Ambiguity guard**: plausibly-equal names with conflicting context/domain
   signals → ``REVIEW_REQUIRED`` (never silent).
5. **Everything else** → ``DO_NOT_MERGE`` (create new entity).

This is layer 2 of the three-layer contract (§5): ``services.entity.blocking``
narrows the comparison set before scoring, and clusters what still needs a human
after it.

**Graph-structural evidence** (``graph_neighbors`` / ``context_ids``) is a second,
independent signal: a candidate that shares knowledge-graph neighbours with the
entities already resolved for this document is corroborated by the graph's shape,
not by any similarity score. It can raise confidence and it can promote a weak
name overlap to ``REVIEW_REQUIRED`` — it can never, alone or in combination,
produce a merge that orthographic evidence did not already justify.

The resolver is pure (no DB, no network) so quality can be unit-tested
deterministically; persistence lives in the ingest wiring layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DO_NOT_MERGE = "do_not_merge"
REVIEW_REQUIRED = "review_required"
AUTO_MERGE = "auto_merge"

# Trailing words that denote the *same* entity ("Postgres database" = Postgres).
_GENERIC_SUFFIXES: frozenset[str] = frozenset(
    {"database", "db", "server", "system", "platform", "engine", "software", "binary"}
)

# Leading articles/prepositions stripped before comparing.
_STRIP_LEADING = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)

# These are *meaningful* qualifiers that keep entities distinct (CASE E/F).
_RELATION_SUFFIXES: dict[str, str] = {
    "logo": "related_to",
    "compose": "part_of",
    "swarm": "part_of",
    "desktop": "related_to",
    "cli": "tool_for",
    "sdk": "related_to",
}


@dataclass(frozen=True)
class ResolutionEvidence:
    """Facts used to reach a decision (kept for audit/explainability)."""

    matched_by: str  # exact_alias | suffix_collapse | none
    normalized: str
    canonical_of_candidate: str | None = None
    notes: list[str] = field(default_factory=list)
    # How many knowledge-graph neighbours the candidate shares with the entities
    # already resolved for this document. Corroboration, never identity on its own.
    structural_support: int = 0


@dataclass(frozen=True)
class CandidateEntity:
    """A pre-existing entity offered for resolution (thin DTO)."""

    entity_id: str
    canonical_name: str
    aliases: list[str]
    entity_type: str = "concept"
    domain: str | None = None


@dataclass(frozen=True)
class EntityResolution:
    """Outcome of resolving one mention against known entities."""

    mention: str
    entity_type: str
    decision: str  # AUTO_MERGE | REVIEW_REQUIRED | DO_NOT_MERGE
    target_id: str | None  # entity to merge onto, or None (new entity)
    confidence: float
    reason: str
    evidence: ResolutionEvidence
    relation_hint: tuple[str, str] | None = None  # (relation_type, target_name)


def normalize_name(name: str) -> str:
    """Lowercase, strip leading articles and trailing generic suffixes."""
    token = re.sub(r"[^0-9a-z\s]", " ", name.lower())
    token = re.sub(r"\s{2,}", " ", token).strip()
    token = _STRIP_LEADING.sub("", token).strip()
    words = token.split()
    while words and words[-1] in _GENERIC_SUFFIXES:
        words.pop()
    return " ".join(words)


class EntityResolver:
    """Decide merge policy for a mention given known entities (§13)."""

    def __init__(
        self,
        canonical_alias: dict[str, list[str]] | None = None,
    ) -> None:
        # Additional cross-name aliases: {"postgresql": ["postgres", "pg"]}.
        self.canonical_alias = canonical_alias or {}

    def resolve(
        self,
        mention: str,
        *,
        entity_type: str = "concept",
        known: list[CandidateEntity] = (),
        context: str | None = None,
        domain: str | None = None,
        graph_neighbors: dict[str, set[str]] | None = None,
        context_ids: set[str] | None = None,
    ) -> EntityResolution:
        """Resolve ``mention`` against ``known`` entities deterministically.

        ``known`` should already be narrowed by ``services.entity.blocking`` —
        the resolver rebuilds its lookup tables per call, so handing it the whole
        corpus is both quadratic and a correctness hazard (every extra candidate
        is another chance for the subsumption guard to fire on an unrelated name).

        ``graph_neighbors`` maps entity id → ids of its knowledge-graph
        neighbours, and ``context_ids`` are the entities already resolved for this
        document. Their intersection is *structural* evidence: independent of
        spelling and of any embedding. It adjusts confidence and can force a
        review, never a merge.
        """
        norm = normalize_name(mention)
        if not norm:
            # Every token was a generic suffix ("platform", "system", "BINARY"),
            # so there is no identity key left to compare. Matching here would
            # merge "system" onto "platform" at confidence 0.98.
            return EntityResolution(
                mention=mention,
                entity_type=entity_type,
                decision=DO_NOT_MERGE,
                target_id=None,
                confidence=0.2,
                reason="name has no identity tokens after normalization",
                evidence=ResolutionEvidence(matched_by="none", normalized=""),
            )

        by_norm = {
            n: e for e in known if (n := normalize_name(e.canonical_name))
        }
        alias_lookup: dict[str, CandidateEntity] = {}
        for e in known:
            for alias in e.aliases:
                if alias_norm := normalize_name(alias):
                    alias_lookup[alias_norm] = e

        target: CandidateEntity | None = None
        matched_by = "none"
        reason = ""
        notes: list[str] = []

        # 1. Exact identity on canonical name.
        if norm in by_norm:
            target, matched_by, reason = (
                by_norm[norm],
                "canonical",
                "name matches canonical",
            )
        # 2. Exact identity on a registered alias.
        elif norm in alias_lookup:
            target, matched_by, reason = (
                alias_lookup[norm],
                "alias",
                "name matches a registered alias",
            )
        # 3. Cross-language / nickname alias table (postgres → postgresql).
        elif self.canonical_alias:
            for canonical, aliases in self.canonical_alias.items():
                if norm in {normalize_name(a) for a in aliases}:
                    key = normalize_name(canonical)
                    if key in by_norm:
                        target, matched_by, reason = (
                            by_norm[key],
                            "alias",
                            f"nickname alias of {canonical}",
                        )
                        notes = [f"{mention} == {canonical} via alias table"]
                    break

        # Evidence-based merge candidate found → apply the §13 governance gate:
        # an identical name in a *different domain* is never auto-merged.
        if target is not None:
            support = self._structural_support(target, graph_neighbors, context_ids)
            if domain and target.domain and domain != target.domain:
                # A shared-neighbour graph does NOT overrule a domain conflict.
                # Two entities can be structurally adjacent and still be
                # different things; that is precisely what review is for.
                return EntityResolution(
                    mention=mention,
                    entity_type=entity_type,
                    decision=REVIEW_REQUIRED,
                    target_id=None,
                    confidence=0.4,
                    reason=(
                        f"same name as '{target.canonical_name}' but conflicting "
                        f"domain {target.domain!r} vs {domain!r}"
                    ),
                    evidence=ResolutionEvidence(
                        matched_by=matched_by,
                        normalized=norm,
                        canonical_of_candidate=target.canonical_name,
                        notes=["domain conflict → human review"],
                        structural_support=support,
                    ),
                )
            return self._merge_with(
                mention,
                entity_type,
                target,
                matched_by,
                reason,
                notes,
                structural_support=support,
            )

        # 4. Subsumption guard: one is a strict prefix of the other with a
        #    meaningful trailing token → distinct entities + relation hint.
        for norm_other, other in {**by_norm, **alias_lookup}.items():
            relation = self._subsumes(norm, norm_other)
            if relation is not None:
                support = self._structural_support(other, graph_neighbors, context_ids)
                return EntityResolution(
                    mention=mention,
                    entity_type=entity_type,
                    decision=DO_NOT_MERGE,
                    target_id=None,
                    # Shared graph neighbours make the *relation* more credible,
                    # which is what this branch emits — still two entities.
                    confidence=min(0.95, 0.85 + 0.02 * support),
                    reason=(
                        f"'{mention}' is related to but distinct from "
                        f"'{other.canonical_name}' (subsumption guard, CASE E/F)"
                    ),
                    evidence=ResolutionEvidence(
                        matched_by="subsumption",
                        normalized=norm,
                        canonical_of_candidate=other.canonical_name,
                        notes=[f"{norm} vs {norm_other}"],
                        structural_support=support,
                    ),
                    relation_hint=(relation, other.canonical_name),
                )

        # 5. No identity evidence → new entity.  Neighbour-name overlap with
        #    conflicting domain still deserves a human review, never a merge.
        if norm and domain:
            overlapping = [
                e
                for e in known
                if e.domain and e.domain != domain and normalize_name(e.canonical_name) == norm
            ]
            if overlapping:
                return EntityResolution(
                    mention=mention,
                    entity_type=entity_type,
                    decision=REVIEW_REQUIRED,
                    target_id=None,
                    confidence=0.4,
                    reason=(
                        f"'{mention}' collides with '{overlapping[0].canonical_name}' "
                        f"in a different domain"
                    ),
                    evidence=ResolutionEvidence(
                        matched_by="none",
                        normalized=norm,
                        canonical_of_candidate=overlapping[0].canonical_name,
                        notes=["cross-domain collision → human review"],
                    ),
                )

        # 6. Structural corroboration without identity evidence. A name that
        #    partially overlaps a candidate AND shares graph neighbours with the
        #    entities already resolved for this document is not proof — but it is
        #    the case a human should look at, so it is queued rather than
        #    silently forked into a second entity.
        structural = self._structural_review(
            norm, known, graph_neighbors, context_ids
        )
        if structural is not None:
            other, support = structural
            return EntityResolution(
                mention=mention,
                entity_type=entity_type,
                decision=REVIEW_REQUIRED,
                target_id=None,
                confidence=0.45,
                reason=(
                    f"'{mention}' overlaps '{other.canonical_name}' and shares "
                    f"{support} graph neighbour(s) with this document"
                ),
                evidence=ResolutionEvidence(
                    matched_by="graph_structural",
                    normalized=norm,
                    canonical_of_candidate=other.canonical_name,
                    notes=["shared-neighbour evidence → human review"],
                    structural_support=support,
                ),
            )

        return EntityResolution(
            mention=mention,
            entity_type=entity_type,
            decision=DO_NOT_MERGE,
            target_id=None,
            confidence=0.3,
            reason="no identity evidence → create new entity",
            evidence=ResolutionEvidence(matched_by="none", normalized=norm),
        )

    # ------------------------------------------------------------------

    def _merge_with(
        self,
        mention: str,
        entity_type: str,
        target: CandidateEntity,
        matched_by: str,
        reason: str,
        notes: list[str],
        *,
        structural_support: int = 0,
    ) -> EntityResolution:
        return EntityResolution(
            mention=mention,
            entity_type=entity_type,
            decision=AUTO_MERGE,
            target_id=target.entity_id,
            # Orthographic identity is already decisive; shared graph neighbours
            # only close the remaining gap. They can never create a merge, which
            # is why this is a ceiling and not a weighted sum.
            confidence=min(0.99, 0.98 + 0.005 * min(structural_support, 2)),
            reason=(
                f"{reason} (+{structural_support} shared graph neighbour(s))"
                if structural_support
                else reason
            ),
            evidence=ResolutionEvidence(
                matched_by=matched_by,
                normalized=normalize_name(mention),
                canonical_of_candidate=target.canonical_name,
                notes=notes,
                structural_support=structural_support,
            ),
        )

    @staticmethod
    def _structural_support(
        candidate: CandidateEntity,
        graph_neighbors: dict[str, set[str]] | None,
        context_ids: set[str] | None,
    ) -> int:
        """Count graph neighbours ``candidate`` shares with the document context.

        This is the §5 "structural" signal: it reads the knowledge graph's edges,
        which exist only because ``RelationValidator`` already accepted evidence
        for them. No embedding is consulted, and a count of 0 is the norm — most
        candidates are simply not adjacent to what this document talks about.
        """
        if not graph_neighbors or not context_ids:
            return 0
        neighbours = graph_neighbors.get(candidate.entity_id)
        if not neighbours:
            return 0
        return len(neighbours & context_ids)

    def _structural_review(
        self,
        norm: str,
        known: list[CandidateEntity],
        graph_neighbors: dict[str, set[str]] | None,
        context_ids: set[str] | None,
    ) -> tuple[CandidateEntity, int] | None:
        """Find a token-overlapping candidate corroborated by the graph.

        Requires *both* a partial name overlap and shared neighbours. Either one
        alone is too weak: name overlap without structure is the ordinary case
        for unrelated entities that share a common word, and structure without
        name overlap describes every pair of related-but-distinct entities.
        """
        if not graph_neighbors or not context_ids:
            return None
        tokens = set(norm.split())
        if not tokens:
            return None
        best: tuple[CandidateEntity, int] | None = None
        for candidate in known:
            other_norm = normalize_name(candidate.canonical_name)
            if not other_norm or other_norm == norm:
                continue
            if not tokens & set(other_norm.split()):
                continue
            support = self._structural_support(candidate, graph_neighbors, context_ids)
            if support > 0 and (best is None or support > best[1]):
                best = (candidate, support)
        return best

    @staticmethod
    def _subsumes(a: str, b: str) -> str | None:
        """Return relation hint if ``a`` subsumes ``b`` (or vice-versa).

        Subsumption = one token-set is a strict prefix of the other and the
        extra trailing token is semantically meaningful (logo, compose, …).
        Returns ``None`` when they are not related-but-distinct.
        """
        wa, wb = a.split(), b.split()
        if wa and wa == wb:
            return None
        prefix_len = 0
        if len(wa) < len(wb) and wb[: len(wa)] == wa:
            prefix_len = len(wa)
            extra = wb[prefix_len:]
        elif len(wb) < len(wa) and wa[: len(wb)] == wb:
            prefix_len = len(wb)
            extra = wa[prefix_len:]
        else:
            return None
        if len(extra) != 1:
            return None
        qualifier = extra[0]
        return _RELATION_SUFFIXES.get(qualifier)