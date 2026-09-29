"""Candidate blocking + review clustering for entity resolution (§5, layer 1/3).

Resolution has three layers in the ingestion contract: **blocking** narrows the
comparison set, **multi-signal scoring** decides identity, **clustering** groups
what still needs a human. Only the middle layer existed; this module supplies
the outer two, and neither of them ever consults an embedding.

Why blocking is not an optimization detail
------------------------------------------
``resolve_entities`` used to hand the resolver *every* entity in the database and
the resolver rebuilt its lookup tables per mention. Measured on the live corpus
(2470 entities, 60 mentions — one ordinary document):

    resolve() 60 mentions x 2470 known : 519.5 ms
    resolve() 60 mentions x  155 known :  32.3 ms   (16.1x)

The cost is quadratic in corpus size, so it degrades with every ingest. Worse, a
comparison set that wide is a *correctness* hazard, not just a slow one: every
extra candidate is another chance for the subsumption guard to fire against an
unrelated name that happens to share a prefix.

Blocking keys
-------------
A mention blocks with a candidate when they share at least one **token key** or
an **acronym key**. Both are derived from the resolver's own normalization, so a
name can never block differently from the way it is later compared:

- token keys: normalized tokens of the name and of every registered alias.
- acronym key: initials of a multi-word name (``Legal Integrity Analytics`` →
  ``lia``), which is how the live corpus already stores that entity's alias.

Blocking is recall-first: it may admit a candidate that scoring then rejects, but
it must never hide a candidate that scoring would have merged. Anything the
blocker drops is a candidate the resolver never sees, so the rule is deliberately
generous and the *scoring* layer stays strict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from services.entity.resolver import CandidateEntity, normalize_name

# Tokens this short are not selective enough to be a blocking key on their own —
# they would pull in a large slice of the corpus for no gain. Known technical
# acronyms are exempt because for them the short token IS the identity.
_MIN_TOKEN_LEN = 3
_KNOWN_SHORT_TOKENS: frozenset[str] = frozenset(
    {"ai", "ml", "db", "os", "id", "ui", "ux", "js", "go", "r", "c", "sql", "api",
     "rag", "llm", "gpu", "cpu", "ssl", "tls", "dns", "css", "seo", "pdf", "csv",
     "xml", "vin", "dmv", "epa", "dot"}
)


def blocking_keys(name: str, *, aliases: tuple[str, ...] | list[str] = ()) -> frozenset[str]:
    """Return the blocking keys for ``name`` and its ``aliases``.

    Keys are normalized tokens plus the acronym of each multi-word form. An
    empty result means the name carries no usable key (it normalized away
    entirely) — the caller must treat that as "no block", never as "matches
    everything".
    """
    keys: set[str] = set()
    for raw in (name, *aliases):
        norm = normalize_name(str(raw or ""))
        if not norm:
            continue
        tokens = norm.split()
        for token in tokens:
            if len(token) >= _MIN_TOKEN_LEN or token in _KNOWN_SHORT_TOKENS:
                keys.add(token)
        if len(tokens) > 1:
            acronym = "".join(t[0] for t in tokens if t)
            if len(acronym) >= 2:
                keys.add(acronym)
    return frozenset(keys)


@dataclass(frozen=True)
class BlockedCandidates:
    """The narrowed comparison set for one document, with its own audit trail."""

    candidates: list[CandidateEntity]
    considered: int  # how many entities were offered to the blocker
    keys: frozenset[str] = field(default_factory=frozenset)
    truncated: bool = False

    @property
    def kept(self) -> int:
        return len(self.candidates)


def block_candidates(
    mentions: list[str],
    known: list[CandidateEntity],
    *,
    max_candidates: int = 400,
) -> BlockedCandidates:
    """Narrow ``known`` to the candidates that share a blocking key with a mention.

    The result is the *only* set the resolver sees, so a candidate that shares no
    token and no acronym with anything the document mentions is not merely
    deprioritized — it is out of scope for this document, which is the point.

    ``max_candidates`` caps a pathological block (a document mentioning a token
    that half the corpus contains). Truncation keeps the shortest names first:
    a short canonical name is the one an alias or suffix variant is most likely
    to be a form of, so it is the candidate worth keeping when we must choose.
    """
    mention_keys: set[str] = set()
    for mention in mentions:
        mention_keys |= blocking_keys(mention)

    if not mention_keys:
        return BlockedCandidates(candidates=[], considered=len(known), keys=frozenset())

    matched: list[CandidateEntity] = []
    for candidate in known:
        if blocking_keys(candidate.canonical_name, aliases=candidate.aliases) & mention_keys:
            matched.append(candidate)

    truncated = len(matched) > max_candidates
    if truncated:
        matched = sorted(matched, key=lambda c: len(c.canonical_name))[:max_candidates]
    return BlockedCandidates(
        candidates=matched,
        considered=len(known),
        keys=frozenset(mention_keys),
        truncated=truncated,
    )


@dataclass(frozen=True)
class ReviewCluster:
    """A group of entities that resolve to the same identity key.

    The third resolution layer. Two rows in this cluster are *probably* the same
    entity, but nothing in layers 1-2 produced identity evidence strong enough to
    merge them, so they are grouped for one human decision instead of being
    silently collapsed or silently forgotten.
    """

    key: str  # the shared resolver-normalized name
    entity_ids: list[str]
    names: list[str]


def review_clusters(known: list[CandidateEntity]) -> list[ReviewCluster]:
    """Group ``known`` entities whose normalized names collide.

    Measured on the live corpus this found 13 collision groups over 2470 rows —
    ``Machine learning`` / ``machine-learning``, ``Risk Analysis`` / ``risk
    analysis system``, ``The Mozilla Blog`` / ``mozilla blog``. They exist because
    the keyword path and the resolver path deduplicated with *different*
    normalizers, so each pass believed the name was new.

    A cluster is a review queue entry, never an automatic merge: names that
    collide under normalization can still be different things (``Review`` the
    action vs ``Review System`` the product), which is exactly the judgement a
    human has to make.
    """
    groups: dict[str, list[CandidateEntity]] = {}
    for candidate in known:
        key = normalize_name(candidate.canonical_name)
        if not key:
            # A name that normalizes to nothing ("platform", "BINARY") has no
            # identity key at all. Clustering it would group unrelated rows
            # under the empty string.
            continue
        groups.setdefault(key, []).append(candidate)
    return [
        ReviewCluster(
            key=key,
            entity_ids=[c.entity_id for c in members],
            names=[c.canonical_name for c in members],
        )
        for key, members in sorted(groups.items())
        if len(members) > 1
    ]
