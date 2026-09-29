"""Unit and integration tests for Step 0 shared database foundation.

Validates:
1. All newly mapped Step 0 ORM models and drift-closed agent models exist in Base.metadata.
2. Step 0 settings in catalog.py have non-empty source and rationale, self-validate defaults,
   and meet bounds constraints (such as gpu_reserve_free_mib >= 5120).
3. Integration with PostgreSQL:
   - Tables are created without seeding runtime data (DB empty start).
   - COALESCE expression unique indexes enforce NULL-safe uniqueness correctly.
   - All 15 Step 0 tables can be inserted, queried, and cascaded without errors.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core import database as db
from core.config import settings
from services.settings.catalog import SPECS_BY_KEY
from services.settings.schema import validate


def test_step0_orm_models_registered_in_metadata():
    """All 15 target tables are registered in SQLAlchemy Base.metadata."""
    expected_tables = {
        # Dynamic taxonomy
        "taxonomy_facets",
        "taxonomy_terms",
        "taxonomy_aliases",
        "taxonomy_rules",
        # Capability registry & self-org
        "capabilities",
        "reorg_operations",
        # Resource monitor
        "gpu_resource_samples",
        # Multi-agent drift-closed + durable runs + learned domains
        "agents",
        "agent_tasks",
        "agent_conversations",
        "agent_reports",
        "agent_audit_events",
        "agent_memory",
        "agent_runs",
        "agent_domains",
    }
    registered_tables = set(db.Base.metadata.tables.keys())
    for tbl in expected_tables:
        assert tbl in registered_tables, f"Table {tbl} missing in Base.metadata"


def test_step0_catalog_specs_validity():
    """All Step 0 SettingSpec entries have valid provenance, rationale, and bounds."""
    expected_keys = [
        "selforg_merge_threshold",
        "selforg_split_dispersion_threshold",
        "selforg_enabled",
        "selforg_frequency",
        "gpu_reserve_free_mib",
        "gpu_monitor_interval_seconds",
        "max_parallel_tasks",
        "embed_batch_size",
        "taxonomy_acceptance_confidence",
        "taxonomy_ambiguity_confidence",
        "taxonomy_propose_threshold",
        "taxonomy_promote_min_hits",
        "consolidation_confidence_threshold",
        "agent_workflows_enabled",
        "agent_max_steps",
        "agent_max_model_calls",
        "agent_timeout_seconds",
    ]

    for key in expected_keys:
        assert key in SPECS_BY_KEY, f"Key {key} missing from SPECS_BY_KEY"
        spec = SPECS_BY_KEY[key]
        assert spec.source == "approved_step0_plan"
        assert spec.rationale and len(spec.rationale.strip()) > 10
        # Check self-validation
        assert validate(spec, spec.default_value) is not None

    # Strict constraint from CLAUDE.md §3: >= 5GB VRAM
    gpu_vram_spec = SPECS_BY_KEY["gpu_reserve_free_mib"]
    assert gpu_vram_spec.min_value >= 5120
    assert gpu_vram_spec.default_value >= 5120


# --- PostgreSQL Integration Tests ---


_STEP0_SCHEMA = "braind_step0_test"

_STEP0_TABLES = (
    db.TaxonomyFacet.__table__,
    db.TaxonomyTerm.__table__,
    db.TaxonomyAlias.__table__,
    db.TaxonomyRule.__table__,
    db.Capability.__table__,
    db.ReorgOperation.__table__,
    db.GPUResourceSample.__table__,
    db.Agent.__table__,
    db.AgentTaskRecord.__table__,
    db.AgentConversation.__table__,
    db.AgentReport.__table__,
    db.AgentAuditEvent.__table__,
    db.AgentMemory.__table__,
    db.AgentRun.__table__,
    db.AgentDomain.__table__,
)


@pytest_asyncio.fixture
async def step0_engine():
    """Async engine creating isolated schema with all Step 0 tables.

    Isolation notes:
    - search_path leads with the test schema; ``public`` stays reachable because
      taxonomy_terms has an FK onto the pre-existing ``chunks`` table.
    - ``checkfirst=False`` is mandatory: with the default (True) SQLAlchemy sees the
      already-migrated ``public`` copies through the search path, skips creating the
      isolated copies, and every INSERT then leaks into ``public``.
    - the schema is dropped before creation so a previously interrupted run cannot
      leave rows behind and break DB-empty-start assertions.
    """
    eng = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {"search_path": f"{_STEP0_SCHEMA},public"}
        },
    )

    async with eng.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {_STEP0_SCHEMA} CASCADE"))
        await conn.execute(text(f"CREATE SCHEMA {_STEP0_SCHEMA}"))
        await conn.run_sync(
            lambda sync_conn: db.Base.metadata.create_all(
                sync_conn, tables=list(_STEP0_TABLES), checkfirst=False
            )
        )
        created = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = :schema"
                ),
                {"schema": _STEP0_SCHEMA},
            )
        ).scalar()
        assert created == len(_STEP0_TABLES), (
            f"isolated schema must own all Step 0 tables, found {created}"
        )

    yield eng

    async with eng.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {_STEP0_SCHEMA} CASCADE"))
    await eng.dispose()


@pytest_asyncio.fixture
async def step0_session(step0_engine):
    maker = async_sessionmaker(step0_engine, expire_on_commit=False)
    async with maker() as session:
        yield session


@pytest.mark.asyncio
async def test_db_empty_start_behavior(step0_session):
    """Tables start completely empty (no seeded taxonomy or capabilities)."""
    facets_count = await step0_session.scalar(select(func.count(db.TaxonomyFacet.facet)))
    terms_count = await step0_session.scalar(select(func.count(db.TaxonomyTerm.id)))
    caps_count = await step0_session.scalar(select(func.count(db.Capability.id)))
    reorg_count = await step0_session.scalar(select(func.count(db.ReorgOperation.id)))

    assert facets_count == 0
    assert terms_count == 0
    assert caps_count == 0
    assert reorg_count == 0


@pytest.mark.asyncio
async def test_taxonomy_alias_coalesce_unique_index(step0_session):
    """Test COALESCE(locale, '') unique index on taxonomy_aliases."""
    facet = db.TaxonomyFacet(
        facet="domain", label="Domain", growable=True, authority_editable=True
    )
    step0_session.add(facet)
    await step0_session.flush()

    term_id = uuid.uuid4()
    term = db.TaxonomyTerm(
        id=term_id,
        facet="domain",
        term="quantum_computing",
        origin="model",
        status="active",
    )
    step0_session.add(term)
    await step0_session.flush()

    # Insert alias with locale=None
    alias1 = db.TaxonomyAlias(
        id=uuid.uuid4(),
        term_id=term_id,
        surface_form="QC",
        normalized_form="qc",
        locale=None,
    )
    step0_session.add(alias1)
    await step0_session.commit()

    # Inserting duplicate (normalized_form="qc", locale=None) must violate unique index
    alias2 = db.TaxonomyAlias(
        id=uuid.uuid4(),
        term_id=term_id,
        surface_form="Q.C.",
        normalized_form="qc",
        locale=None,
    )
    step0_session.add(alias2)
    with pytest.raises(IntegrityError):
        await step0_session.commit()
    await step0_session.rollback()

    # Inserting duplicate with explicit locale="en" should succeed (distinct from NULL)
    alias3 = db.TaxonomyAlias(
        id=uuid.uuid4(),
        term_id=term_id,
        surface_form="QC",
        normalized_form="qc",
        locale="en",
    )
    step0_session.add(alias3)
    await step0_session.commit()


@pytest.mark.asyncio
async def test_taxonomy_rule_coalesce_unique_index(step0_session):
    """Test COALESCE(locale, '') unique index on taxonomy_rules."""
    facet = db.TaxonomyFacet(
        facet="entity", label="Entity", growable=True, authority_editable=True
    )
    step0_session.add(facet)
    await step0_session.flush()

    rule1 = db.TaxonomyRule(
        id=uuid.uuid4(),
        facet="entity",
        kind="prefix",
        pattern="test_",
        locale=None,
        origin="model",
        enabled=True,
    )
    step0_session.add(rule1)
    await step0_session.commit()

    # Duplicate (facet, kind, pattern, locale=None) must fail
    rule2 = db.TaxonomyRule(
        id=uuid.uuid4(),
        facet="entity",
        kind="prefix",
        pattern="test_",
        locale=None,
        origin="user",
        enabled=False,
    )
    step0_session.add(rule2)
    with pytest.raises(IntegrityError):
        await step0_session.commit()
    await step0_session.rollback()


@pytest.mark.asyncio
async def test_agent_run_and_domain_lifecycle(step0_session):
    """Test Agent, AgentRun, and AgentDomain ORM persistence."""
    agent = db.Agent(
        agent_id="spec_test_01",
        name="Test Specialist",
        role="SPECIALIST",
        department="Engineering",
        available_tools=["search", "calculate"],
    )
    step0_session.add(agent)

    facet = db.TaxonomyFacet(
        facet="skills", label="Skills", growable=True, authority_editable=True
    )
    step0_session.add(facet)
    await step0_session.flush()

    term = db.TaxonomyTerm(
        id=uuid.uuid4(),
        facet="skills",
        term="python_optimization",
        origin="derived",
        status="active",
    )
    step0_session.add(term)
    await step0_session.flush()

    # Create AgentDomain mapping
    domain = db.AgentDomain(
        id=uuid.uuid4(),
        agent_id=agent.agent_id,
        facet=facet.facet,
        term_id=term.id,
        confidence=0.95,
        origin="model",
    )
    step0_session.add(domain)

    # Create AgentRun
    run_id = uuid.uuid4()
    agent_run = db.AgentRun(
        run_id=run_id,
        root_task_id="task_root_100",
        state="RUNNING",
        checkpoint_json={"step": 2, "last_action": "search"},
        budget_json={"max_steps": 32, "steps_taken": 2},
    )
    step0_session.add(agent_run)
    await step0_session.commit()

    # Query back
    fetched_run = await step0_session.get(db.AgentRun, run_id)
    assert fetched_run is not None
    assert fetched_run.state == "RUNNING"
    assert fetched_run.checkpoint_json["step"] == 2

    fetched_domains = (
        (
            await step0_session.execute(
                select(db.AgentDomain).where(db.AgentDomain.agent_id == agent.agent_id)
            )
        )
        .scalars()
        .all()
    )
    assert len(fetched_domains) == 1
    assert fetched_domains[0].term_id == term.id


@pytest.mark.asyncio
async def test_gpu_resource_sample_bigint(step0_session):
    """Test GPUResourceSample with 24GB VRAM byte counts (> 2^31-1)."""
    vram_24gb = 24 * 1024 * 1024 * 1024  # 25,769,803,776 bytes
    sample = db.GPUResourceSample(
        id=uuid.uuid4(),
        gpu_index=0,
        total_vram_bytes=vram_24gb,
        used_vram_bytes=vram_24gb // 2,
        free_vram_bytes=vram_24gb // 2,
        utilization_percent=50.0,
        temperature_celsius=62.5,
    )
    step0_session.add(sample)
    await step0_session.commit()

    fetched = await step0_session.get(db.GPUResourceSample, sample.id)
    assert fetched is not None
    assert fetched.total_vram_bytes == vram_24gb
