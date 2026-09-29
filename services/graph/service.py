"""Graph service — bounded traversal over the typed knowledge graph (§7.2).

The graph is stored as first-class rows in ``knowledge_links`` (typed, weighted,
confidence- and evidence-aware, lifecycle-versioned). This service exposes:

- **CRUD**: upsert/deactivate a typed edge, always carrying evidence.
- **Bounded traversal**: neighborhood walk that hard-caps on
  ``graph_max_hops`` / ``graph_max_nodes`` / ``graph_max_edges_per_node`` from
  ``core/config.py`` — never an unbounded walk (§master graph-safety).
- **Integrity check**: dangling-edge detection so maintenance can flag rows
  whose endpoint no longer exists in the node tables.

The service is async and takes the session factory; tests mock the session
(never live PostgreSQL).  No edge is ever created on cosine similarity alone —
``upsert_edge`` requires a non-empty ``evidence`` list (§9/§63).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core import database as db
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class GraphTraversalError(RuntimeError):
    """Raised when a traversal is rejected (e.g. bad bounds, non-existent root)."""


@dataclass(frozen=True)
class Neighbor:
    """One hop result in a neighborhood walk."""

    node_id: str
    node_kind: str
    relation_type: str
    confidence: float
    depth: int
    direction: str  # "out" | "in"
    evidence_count: int = 0


@dataclass(frozen=True)
class GraphEdge:
    """A typed knowledge-graph edge."""

    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    weight: float
    source_kind: str
    target_kind: str
    evidence_count: int
    lifecycle: str
    reason: str | None = None


@dataclass
class GraphNeighborhood:
    """Result of a bounded neighborhood walk."""

    root_id: str
    hops: int
    nodes: list[Neighbor] = field(default_factory=list)
    truncated: bool = False  # hit a bound (graph_max_nodes / edges)

    @property
    def node_ids(self) -> list[str]:
        seen: list[str] = []
        for n in self.nodes:
            if n.node_id not in seen:
                seen.append(n.node_id)
        return seen


class GraphService:
    """CRUD + bounded traversal over ``knowledge_links``."""

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.max_hops = int(settings.graph_max_hops)
        self.max_nodes = int(settings.graph_max_nodes)
        self.max_edges_per_node = int(settings.graph_max_edges_per_node)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def upsert_edge(
        self,
        session,
        *,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float,
        evidence: list[str],
        source_kind: str = "entity",
        target_kind: str = "entity",
        reason: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Upsert a typed evidence-backed edge.

        ``evidence`` MUST be non-empty — never create an edge on cosine
        similarity alone (§9/§63).
        """
        if not evidence:
            raise GraphTraversalError("edge requires non-empty evidence (§9)")
        if confidence < 0 or confidence > 1:
            raise GraphTraversalError(f"confidence out of range: {confidence}")
        await session.execute(
            pg_insert(db.KnowledgeLink)
            .values(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=round(float(confidence), 4),
                weight=round(float(confidence), 4),
                source_kind=source_kind,
                target_kind=target_kind,
                evidence=evidence,
                evidence_count=len(evidence),
                reason=reason,
                model_version=model_version,
                lifecycle="active",
            )
            .on_conflict_do_update(
                index_elements=["source_id", "target_id", "relation_type"],
                set_={
                    "confidence": round(float(confidence), 4),
                    "weight": round(float(confidence), 4),
                    "evidence": evidence,
                    "evidence_count": len(evidence),
                    "reason": reason,
                    "model_version": model_version,
                    "lifecycle": "active",
                },
            )
        )

    async def deactivate_edge(
        self,
        session,
        *,
        source_id: str,
        target_id: str,
        relation_type: str | None,
    ) -> int:
        """Mark an edge superseded (non-destructive: never DELETE the row).

        ``relation_type=None`` suppresses *every* type on the pair — used by a
        human override that asserts "no relation here" without naming one."""
        clauses = [
            db.KnowledgeLink.source_id == source_id,
            db.KnowledgeLink.target_id == target_id,
        ]
        if relation_type is not None:
            clauses.append(db.KnowledgeLink.relation_type == relation_type)
        result = await session.execute(
            db.KnowledgeLink.__table__.update()
            .where(and_(*clauses))
            .values(lifecycle="superseded")
        )
        return int(result.rowcount or 0)

    async def touch_edges(self, session, *, node_ids: list[str]) -> int:
        """Record a retrieval hit on every active edge incident to ``node_ids``.

        Giai đoạn 5, B4 (reuse signal): an edge that retrieval actually walked is
        evidence it is *useful*, so its ``use_count`` rises and ``last_used_at``
        is stamped. The nightly consolidation folds both into ``weight`` —
        reused edges strengthen, unused ones decay (B4 formula). One batched
        UPDATE per retrieval call, never one per edge, so per-query cost stays
        flat as the graph grows. Best-effort: a bump failure must not fail
        retrieval (the caller swallows exceptions).
        """
        ids = [n for n in dict.fromkeys(node_ids) if n]
        if not ids:
            return 0
        result = await session.execute(
            db.KnowledgeLink.__table__.update()
            .where(
                and_(
                    db.KnowledgeLink.lifecycle == "active",
                    or_(
                        db.KnowledgeLink.source_id.in_(ids),
                        db.KnowledgeLink.target_id.in_(ids),
                    ),
                )
            )
            .values(
                use_count=db.KnowledgeLink.use_count + 1,
                last_used_at=func.now(),
            )
        )
        return int(result.rowcount or 0)

    # ------------------------------------------------------------------
    # Bounded traversal
    # ------------------------------------------------------------------

    async def neighborhood(
        self,
        session,
        *,
        root_id: str,
        hops: int | None = None,
        relation_types: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> GraphNeighborhood:
        """BFS neighborhood walk from ``root_id`` bounded by config caps.

        Direction is symmetric (both out- and in-edges), because the typed
        graph carries semantic direction in ``relation_type``.
        """
        if hops is None:
            hops = self.max_hops
        if hops <= 0:
            raise GraphTraversalError(f"hops must be ≥1, got {hops}")
        if hops > self.max_hops:
            hops = self.max_hops  # hard cap from config, never unbounded

        result = GraphNeighborhood(root_id=root_id, hops=hops)
        visited: set[str] = {root_id}
        frontier: list[str] = [root_id]

        for depth in range(1, hops + 1):
            if not frontier:
                break
            next_frontier: list[str] = []
            for node_id in frontier:
                if len(result.nodes) >= self.max_nodes:
                    result.truncated = True
                    return result
                neighbors = await self._direct_neighbors(
                    session,
                    node_id=node_id,
                    relation_types=relation_types,
                    min_confidence=min_confidence,
                    limit=self.max_edges_per_node,
                )
                for nb in neighbors:
                    if len(result.nodes) >= self.max_nodes:
                        result.truncated = True
                        return result
                    nb_depth = Neighbor(
                        node_id=nb.node_id,
                        node_kind=nb.node_kind,
                        relation_type=nb.relation_type,
                        confidence=nb.confidence,
                        depth=depth,
                        direction=nb.direction,
                        evidence_count=nb.evidence_count,
                    )
                    result.nodes.append(nb_depth)
                    if nb.node_id not in visited:
                        visited.add(nb.node_id)
                        next_frontier.append(nb.node_id)
            frontier = next_frontier
        return result

    async def _direct_neighbors(
        self,
        session,
        *,
        node_id: str,
        relation_types: list[str] | None,
        min_confidence: float,
        limit: int,
    ) -> list[Neighbor]:
        """Fetch direct out + in edges for one node, capped at ``limit``."""
        clauses = [
            db.KnowledgeLink.lifecycle == "active",
            db.KnowledgeLink.confidence >= min_confidence,
            or_clause(node_id),
        ]
        if relation_types:
            clauses.append(db.KnowledgeLink.relation_type.in_(relation_types))
        rows = (
            (
                await session.execute(
                    select(
                        db.KnowledgeLink.target_id.label("out_node"),
                        db.KnowledgeLink.target_kind.label("out_kind"),
                        db.KnowledgeLink.source_id.label("in_node"),
                        db.KnowledgeLink.source_kind.label("in_kind"),
                        db.KnowledgeLink.relation_type,
                        db.KnowledgeLink.confidence,
                        db.KnowledgeLink.evidence_count,
                    )
                    .where(and_(*clauses))
                    .limit(limit)
                )
            )
            .all()
        )
        neighbors: list[Neighbor] = []
        for r in rows:
            out_node = str(r[0]) if r[0] is not None else None
            if out_node is not None and out_node != node_id:
                neighbors.append(
                    Neighbor(
                        node_id=out_node,
                        node_kind=str(r[1] or "source"),
                        relation_type=str(r[4] or "related"),
                        confidence=float(r[5] or 0.0),
                        direction="out",
                        depth=0,
                        evidence_count=int(r[6] or 0),
                    )
                )
            in_node = str(r[2]) if r[2] is not None else None
            if in_node is not None and in_node != node_id:
                neighbors.append(
                    Neighbor(
                        node_id=in_node,
                        node_kind=str(r[3] or "source"),
                        relation_type=str(r[4] or "related"),
                        confidence=float(r[5] or 0.0),
                        direction="in",
                        depth=0,
                        evidence_count=int(r[6] or 0),
                    )
                )
        return neighbors

    # ------------------------------------------------------------------
    # Integrity / maintenance helpers
    # ------------------------------------------------------------------

    async def dangling_edges(self, session, *, limit: int = 200) -> list[GraphEdge]:
        """Edges whose source or target no longer exists in ``sources``.

        Maintenance may quarantine (never delete) these for review (§63).
        """
        kl = db.KnowledgeLink
        src_exists = select(db.Source.id).where(db.Source.id == kl.source_id).exists()
        tgt_exists = select(db.Source.id).where(db.Source.id == kl.target_id).exists()
        rows = (
            (
                await session.execute(
                    select(kl)
                    .where(and_(~src_exists | ~tgt_exists))
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
        return [
            GraphEdge(
                source_id=str(r.source_id),
                target_id=str(r.target_id),
                relation_type=r.relation_type,
                confidence=float(r.confidence or 0.0),
                weight=float(r.weight or 0.0),
                source_kind=r.source_kind or "source",
                target_kind=r.target_kind or "source",
                evidence_count=int(r.evidence_count or 0),
                lifecycle=r.lifecycle or "active",
                reason=r.reason,
            )
            for r in rows
        ]

    async def edge_count(self, session) -> int:
        count = await session.execute(
            select(func.count(db.KnowledgeLink.id)).where(
                db.KnowledgeLink.lifecycle == "active"
            )
        )
        return int(count.scalar() or 0)


def or_clause(node_id: str):
    """``source_id == node OR target_id == node`` for one BFS step."""
    return (db.KnowledgeLink.source_id == node_id) | (
        db.KnowledgeLink.target_id == node_id
    )
