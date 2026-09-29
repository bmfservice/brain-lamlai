"""Braind API - shared dependency providers.

Each service is created once at startup and attached to `app.state`.
Dependencies below just pull them out — keeps router code thin and testable.
"""

from __future__ import annotations

from fastapi import Request

from core.errors import ServiceUnavailableError
from core.vllm import VLLMClient
from services.agent.service import AgentService
from services.ingest.service import IngestService
from services.learn.service import LearningService
from services.observability.jobs import IngestionJobTracker
from services.obsidian import ObsidianExporter
from services.rag.pipeline import UnifiedRetrievalService
from services.registry.service import RegistryService
from services.scheduler.service import BraindScheduler
from services.writer.service import WriterService


def get_vllm(request: Request) -> VLLMClient:
    """Return the shared local vLLM inference client."""
    return request.app.state.vllm


def get_ingest(request: Request) -> IngestService:
    return request.app.state.ingest


def get_obsidian(request: Request) -> ObsidianExporter:
    """Return the Obsidian display-layer exporter (§3)."""
    exporter = getattr(request.app.state, "obsidian", None)
    if exporter is None:
        raise ServiceUnavailableError("obsidian exporter is not initialized")
    return exporter


def get_writer(request: Request) -> WriterService:
    return request.app.state.writer


def get_scheduler(request: Request) -> BraindScheduler:
    return request.app.state.scheduler


def get_learn(request: Request) -> LearningService:
    return request.app.state.learn


def get_agent(request: Request) -> AgentService:
    """Return the Dify-style AgentService (tool-using brain)."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise ServiceUnavailableError("AgentService has not been initialized")
    return agent


def get_unified_retrieval(request: Request) -> UnifiedRetrievalService:
    """Return the unified retrieval service — the only retrieval entry point."""
    retrieval = getattr(request.app.state, "retrieval", None)
    if retrieval is None:
        raise ServiceUnavailableError("UnifiedRetrievalService has not been initialized")
    return retrieval


def get_registry(request: Request) -> RegistryService:
    """Return the dynamic UI/system registry service."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise ServiceUnavailableError("RegistryService has not been initialized")
    return registry


def get_session_factory(request: Request):
    """Return the async SQLAlchemy session_factory attached at startup."""
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise ServiceUnavailableError("session_factory is not initialized")
    return session_factory


def get_session_context(request: Request):
    """Resolve the anonymous visitor session for this request (Giai đoạn 4).

    Every route that touches knowledge data depends on this so per-request
    session scoping flows through the existing call sites unchanged.
    """
    from core.session_context import resolve_session

    return resolve_session(request)


def get_file_policy(request: Request):
    """Return the file-import governance gate (Giai đoạn 5, Layer 6)."""
    file_policy = getattr(request.app.state, "file_policy", None)
    if file_policy is None:
        raise ServiceUnavailableError("file_policy is not initialized")
    return file_policy


def get_graph(request: Request):
    """Return the knowledge-graph CRUD/traversal service."""
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise ServiceUnavailableError("graph service is not initialized")
    return graph


def get_relation_classifier(request: Request):
    """Return the consolidated relation classifier (Giai đoạn 5, Layer 4).

    Optional by design — when consolidation is disabled it may be absent, and
    callers must tolerate ``None`` (an override is then ledger-only until the
    classifier is re-enabled)."""
    return getattr(request.app.state, "relation_classifier", None)


def get_job_tracker(request: Request) -> IngestionJobTracker:
    """Return the ingestion-stage job tracker attached at startup."""
    tracker = getattr(request.app.state, "job_tracker", None)
    if tracker is None:
        raise ServiceUnavailableError("job_tracker is not initialized")
    return tracker


def get_settings_store(request: Request):
    """Return the SettingsStore service attached at startup (Phase 1)."""
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        raise ServiceUnavailableError("settings_store is not initialized")
    return store


def get_settings_history(request: Request):
    """Return the SettingsHistory service attached at startup (Phase 1)."""
    history = getattr(request.app.state, "settings_history", None)
    if history is None:
        raise ServiceUnavailableError("settings_history is not initialized")
    return history


def get_settings_resolver(request: Request):
    """Return the 7-tier SettingsResolver service attached at startup (Phase 1b)."""
    resolver = getattr(request.app.state, "settings_resolver", None)
    if resolver is None:
        raise ServiceUnavailableError("settings_resolver is not initialized")
    return resolver


def get_settings_cache(request: Request):
    """Return the in-memory SettingsCache service attached at startup (Phase 1c)."""
    cache = getattr(request.app.state, "settings_cache", None)
    if cache is None:
        raise ServiceUnavailableError("settings_cache is not initialized")
    return cache


def get_settings_snapshot(request: Request):
    """Return the SettingsSnapshotService attached at startup (Phase 1b)."""
    svc = getattr(request.app.state, "settings_snapshot", None)
    if svc is None:
        raise ServiceUnavailableError("settings_snapshot service is not initialized")
    return svc
