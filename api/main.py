"""Braind Server - FastAPI application entrypoint.

Runs the full stack natively (no Docker):
  PostgreSQL + pgvector  <-  vLLM chat + embedding servers  <-  FastAPI
  plus APScheduler background jobs (RSS fetch, vault sync, digest).

Start:  .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import (
    errors,
    routes_admin,
    routes_agent,
    routes_agents,  # Phase 6: Multi-agent routes
    routes_approvals,  # Phase 7: Approval routes
    routes_chat,
    routes_graph,  # Giai đoạn 5: human-override graph edges
    routes_ingest,
    routes_learn,
    routes_policy,  # Giai đoạn 5: governance policy rules
    routes_search,
    routes_settings,  # Brain Second Phase 1c: Settings REST API
    routes_tools,
    routes_ui,
    routes_workflow,  # Hierarchical agent workflow
    routes_writer,
)
from core.agent.tool import get_registry
from core.config import settings
from core.database import create_engine, create_session, init_db
from core.logging import get_logger
from core.vllm import VLLMClient
from services.agent.service import AgentService

# Phase 6: Multi-Agent System
from services.agents import AgentRegistry, OrchestrationEngine

# Phase 7: Approval System
from services.approval import ApprovalService
from services.consolidation.classifier import ConsolidatedRelationClassifier
from services.consolidation.job import ConsolidationService
from services.governance.engine import PolicyEngine
from services.governance.file_policy import FileImportPolicy
from services.graph import GraphService
from services.ingest.search_validate import SearchValidator
from services.ingest.semantic import SemanticIngestService
from services.ingest.service import IngestService
from services.learn.service import LearningService
from services.maintenance.service import MaintenanceService

# Phase 4: Marketing Agent Tools
from services.marketing.manager import MarketingManager
from services.observability.jobs import IngestionJobTracker
from services.observability.metrics import MetricsRegistry
from services.obsidian import ObsidianExporter
from services.rag.pipeline import UnifiedRetrievalService
from services.rag.service import RAGService
from services.registry.service import RegistryService
from services.scheduler.service import BraindScheduler

# Brain Second Phase 1: settings foundation, resolver, cache, snapshots
from services.settings.cache import SettingsCache
from services.settings.history import SettingsHistory
from services.settings.resolver import SettingsResolver
from services.settings.seed import seed_settings
from services.settings.snapshot import SettingsSnapshotService
from services.settings.store import SettingsStore

# Phase 5: Social Media Tools
from services.social.manager import SocialMediaManager
from services.tools.knowledge import KnowledgeRetrieveTool
from services.tools.marketing import (
    CompetitiveIntelTool,  # Fixed: IntelTool not IntelligenceTool
    CompetitorResearchTool,
    CustomerPersonaTool,  # Fixed: PersonaTool not PersonasTool
    MarketingCampaignTool,
    MarketResearchTool,
    ProductAnalysisTool,
    ProductOnboardingTool,
)
from services.tools.rss import FetchRSSTool, FetchURLTool
from services.tools.social import (
    SocialAnalyticsTool,
    SocialMonitorTool,
    SocialPostTool,
    SocialTimingTool,
)
from services.tools.web import WebSearchTool
from services.writer.service import WriterService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Database ---
    engine = create_engine()
    await init_db(engine)
    session_factory = create_session(engine)

    # --- AI client ---
    vllm = VLLMClient()
    if not await vllm.is_available():
        logger.warning(
            "vllm.unavailable",
            llm_base_url=settings.vllm_llm_base_url,
            embed_base_url=settings.vllm_embed_base_url,
            hint="Start both local vLLM servers before Braind.",
        )
    else:
        logger.info(
            "vllm.connected",
            llm_base_url=settings.vllm_llm_base_url,
            embed_base_url=settings.vllm_embed_base_url,
        )

    # --- Services ---
    metrics = MetricsRegistry()
    # Giai đoạn 5, Layer 4: the distilled fast tier for relation decisions.
    # Loaded once at startup (no retrain on boot — persistence is an acceptance
    # criterion) and shared with the consolidation job, which retrains this same
    # instance in place so the new model is live immediately.
    relation_classifier = ConsolidatedRelationClassifier(
        settings.consolidation_model_dir,
        confidence_threshold=settings.consolidation_confidence_threshold,
    )
    if settings.consolidation_enabled:
        relation_classifier.load()
        # Repopulate the in-memory override registry from the ledger so an
        # exact-pair human correction survives a restart *before* the next
        # retrain folds it into the model. Best-effort: a DB hiccup must not
        # block boot.
        try:
            from services.consolidation.dataset import load_human_overrides

            async with session_factory() as _s:
                _rows = await load_human_overrides(_s)
            _n = relation_classifier.load_overrides(_rows)
            if _n:
                logger.info("consolidated_classifier.overrides_loaded", count=_n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("consolidated_classifier.overrides_load_failed", error=str(exc)[:160])
    semantic = SemanticIngestService(
        vllm,
        settings.chunk_size,
        relation_classifier=relation_classifier,
        metrics=metrics,
    )
    job_tracker = IngestionJobTracker(session_factory)
    # Obsidian display layer (§3). It writes into a subdirectory of the vault
    # that the vault scanner is configured to skip, so generated notes are
    # readable in Obsidian without ever being re-ingested as source material.
    obsidian = ObsidianExporter(settings.vault_dir / settings.obsidian_export_dirname)
    ingest = IngestService(
        session_factory,
        vllm,
        semantic=semantic,
        job_tracker=job_tracker,
        obsidian=obsidian,
    )
    rag = RAGService(session_factory, vllm)
    # Giai đoạn 5, Layer 6: governance gate for every mutating action.
    policy_engine = PolicyEngine(session_factory, enabled=settings.policy_enabled)
    file_policy = FileImportPolicy(policy_engine)
    graph_service = GraphService(session_factory)
    # The unified pipeline is the retrieval entry point for every consumer:
    # it exposes the same retrieve()/recent_knowledge() surface as RAGService
    # but adds the analyzer, entity and graph streams plus reranking.
    retrieval = UnifiedRetrievalService(
        rag=rag, graph=graph_service, session_factory=session_factory, metrics=metrics
    )
    web_tool = WebSearchTool(ingest)
    # Phase 4: search validation reuses the SSRF-safe web tool. Injected after
    # construction to break the web_tool -> ingest -> semantic -> validator
    # cycle; ``semantic`` keeps ``search_validator`` optional (None → no-op).
    semantic.search_validator = SearchValidator(vllm, web_tool)
    writer = WriterService(session_factory, vllm, retrieval, ingest=ingest, web_tool=web_tool)
    maintenance = MaintenanceService(session_factory=session_factory, embed_fn=vllm.embed_batch)
    # Giai đoạn 5, Layer 7: the nightly consolidation replays decisions into the
    # distilled classifier and re-breathes edge weights. Shares the classifier
    # instance so a retrain is live immediately (no restart, no reload).
    consolidation = ConsolidationService(
        session_factory,
        relation_classifier,
        maintenance=maintenance,
    )
    scheduler = BraindScheduler(
        session_factory,
        ingest,
        retrieval,
        writer,
        maintenance=maintenance,
        consolidation=consolidation,
    )
    learn = LearningService(session_factory, vllm, retrieval, ingest, web_tool=web_tool)

    # --- Phase 4: Marketing Manager ---
    marketing = MarketingManager(vllm, retrieval, ingest, web_tool)
    logger.info("marketing.manager_initialized")

    # --- Phase 5: Social Media Manager ---
    # Note: Requires platform credentials - initialize empty for now
    # User must configure credentials via environment variables or settings
    social = SocialMediaManager(platforms={}, vllm=vllm)
    logger.info("social.manager_initialized", platforms=list(social.platforms.keys()))

    # --- Dify-style Agent Tools ---
    tool_registry = get_registry()
    
    # Core tools
    tool_registry.register(KnowledgeRetrieveTool(retrieval))
    tool_registry.register(web_tool)
    tool_registry.register(FetchRSSTool(ingest))
    tool_registry.register(FetchURLTool(ingest))
    
    # Phase 4: Marketing tools (7 tools)
    tool_registry.register(ProductAnalysisTool(marketing))
    tool_registry.register(CompetitorResearchTool(marketing))
    tool_registry.register(MarketResearchTool(marketing))
    tool_registry.register(CustomerPersonaTool(marketing))  # Fixed
    tool_registry.register(MarketingCampaignTool(marketing))
    tool_registry.register(CompetitiveIntelTool(marketing))  # Fixed
    tool_registry.register(ProductOnboardingTool(marketing))
    
    # Phase 5: Social media tools (4 tools)
    tool_registry.register(SocialPostTool(social))
    tool_registry.register(SocialAnalyticsTool(social))
    tool_registry.register(SocialMonitorTool(social))
    tool_registry.register(SocialTimingTool(social))
    
    logger.info("agent.tools_registered", count=len(tool_registry.all()))

    # --- Phase 6: Multi-Agent System ---
    agent_registry = AgentRegistry(session_factory)
    await agent_registry.seed_default_agents()  # Seed 10 default agents
    await agent_registry.backfill_hierarchy()  # Fill level/parent for pre-existing rows
    logger.info("agents.registry_initialized")

    # --- Phase 7: Approval System ---
    approval_service = ApprovalService(session_factory)
    logger.info("approval.service_initialized")

    # Hierarchical orchestration engine (real Braind->Leader->Specialist workflow).
    orchestrator = OrchestrationEngine(
        session_factory, vllm, tool_registry, agent_registry, approval_service
    )
    logger.info("agents.orchestrator_initialized")

    # --- Agent Service ---
    agent = AgentService(session_factory, vllm, retrieval, tool_registry, ingest=ingest)
    scheduler.agent = agent
    logger.info("agent.service_initialized")

    # --- Dynamic UI registry ---
    registry = RegistryService(session_factory)
    try:
        seeded = await registry.ensure_system_entries()
        logger.info("registry.system_seeded", count=seeded)
    except Exception as exc:
        logger.warning("registry.system_seed_failed", error=str(exc))

    # --- Settings foundation (Brain Second Phase 1a) ---
    # The DB-backed source of truth for BILL's behavior. The store reuses the
    # governance PolicyEngine.audit so setting changes land in the same
    # audit_logs trail. Seeding is idempotent and never overwrites a value an
    # operator edited; a second boot reports values=0.
    settings_store = SettingsStore(session_factory, auditor=policy_engine)
    settings_history = SettingsHistory(session_factory, settings_store)
    settings_resolver = SettingsResolver(session_factory, store=settings_store)
    settings_cache = SettingsCache()
    settings_snapshot = SettingsSnapshotService(session_factory, settings_resolver)
    if settings.settings_seed_enabled:
        try:
            seed_report = await seed_settings(settings_store, session_factory)
            logger.info(
                "settings.seeded",
                definitions=seed_report.definitions,
                values=seed_report.values,
                skipped=seed_report.skipped_values,
            )
        except Exception as exc:  # noqa: BLE001 — a seed failure must not block boot
            logger.warning("settings.seed_failed", error=str(exc)[:160])

    # --- Attach to app state ---
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.vllm = vllm
    app.state.ingest = ingest
    app.state.obsidian = obsidian
    app.state.job_tracker = job_tracker
    app.state.retrieval = retrieval
    app.state.metrics = metrics
    app.state.relation_classifier = relation_classifier
    app.state.consolidation = consolidation
    app.state.policy_engine = policy_engine
    app.state.file_policy = file_policy
    app.state.graph = graph_service
    app.state.writer = writer
    app.state.scheduler = scheduler
    app.state.agent = agent
    app.state.tools = tool_registry
    app.state.learn = learn
    app.state.registry = registry
    app.state.marketing = marketing  # Phase 4
    app.state.social = social  # Phase 5
    app.state.agent_registry = agent_registry  # Phase 6
    app.state.orchestrator = orchestrator  # Hierarchical workflow engine
    app.state.approval_service = approval_service  # Phase 7
    app.state.settings_store = settings_store  # Brain Second Phase 1a
    app.state.settings_history = settings_history  # Brain Second Phase 1a
    app.state.settings_resolver = settings_resolver  # Brain Second Phase 1b
    app.state.settings_cache = settings_cache  # Brain Second Phase 1c
    app.state.settings_snapshot = settings_snapshot  # Brain Second Phase 1b

    logger.info("braind.startup_complete")

    # --- Background jobs (thay n8n) ---
    if settings.scheduler_enabled:
        scheduler.start()

    # Initial vault sync on first boot (silently best-effort).
    # Budget accounts for the vault sync now running the semantic collaborator
    # (per-file LLM extraction), not just the legacy flat ingest.
    try:
        await asyncio.wait_for(ingest.sync_vault(), timeout=300)
        logger.info("vault.initial_sync_done", vault=str(settings.vault_dir))
    except Exception as exc:
        logger.warning("vault.initial_sync_skipped", error=str(exc))

    yield

    # --- Shutdown ---
    scheduler.shutdown()
    await vllm.close()
    await engine.dispose()
    logger.info("braind.shutdown_complete")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Self-hosted AI second brain: RAG knowledge base, chat, "
            "blog/marketing writer, RSS + vault sync (no Docker, no n8n)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    errors.register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_chat.router)
    app.include_router(routes_search.router)
    app.include_router(routes_policy.router)  # Giai đoạn 5: governance rules
    app.include_router(routes_graph.router)  # Giai đoạn 5: human-override edges
    app.include_router(routes_ingest.router)
    app.include_router(routes_admin.router)
    app.include_router(routes_tools.router)
    app.include_router(routes_agent.router)
    app.include_router(routes_agents.router)  # Phase 6: Multi-agent routes
    app.include_router(routes_approvals.router)  # Phase 7: Approval routes
    app.include_router(routes_workflow.router)  # Hierarchical agent workflow
    app.include_router(routes_writer.router)
    app.include_router(routes_learn.router)
    app.include_router(routes_ui.router)
    app.include_router(routes_settings.router)  # Brain Second Phase 1c: Settings REST API

    # --- Web UI (single-page test console) ---
    _static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")

    @app.get("/ui", include_in_schema=False)
    async def ui():
        return FileResponse(
            _static / "main" / "index.html",
            headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
        )

    @app.get("/", include_in_schema=True)
    async def root():
        return {
            "name": settings.app_name,
            "version": "0.1.0",
            "endpoints": [
                "POST /api/chat",
                "POST /api/chat/stream",
                "GET  /api/agent/tools",
                "POST /api/search",
                "POST /api/ingest/text",
                "POST /api/ingest/file",
                "POST /api/ingest/vault",
                "GET  /api/ingest/stats",
                "POST /api/writer/generate",
                "POST /api/writer/refine",
                "GET  /api/writer/content",
                "GET  /api/scheduler/jobs",
                "POST /api/scheduler/run",
                "POST /api/learn/deep (SSE)",
                "GET  /api/learn/runs",
                "GET  /api/learn/runs/{run_id}",
                "GET  /api/ui/nav",
                "GET  /api/ui/commands",
                "GET  /api/ui/widgets",
                "GET  /api/ui/writer-options",
            ],
        }

    return app


app = create_app()
