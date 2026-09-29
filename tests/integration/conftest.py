"""Integration test fixtures — PostgreSQL thật, cách ly schema braind_test.

Brain Second Phase 1a. Conventions:
- DB URL từ core.config.settings (đọc .env như app thật)
- Cách ly bằng `search_path=braind_test`: ORM's unqualified `setting_values` → schema test
- Chỉ `create_all` 4 bảng settings (né pgvector extension — R14)
- Skip module-level khi không connect được → 120+ unit test vẫn chạy
- Mọi fixture function-scoped: engine/schema/seed tạo mới cho từng test, tránh
  event-loop mismatch của pytest-asyncio 1.4 (asyncpg pool bound vào loop)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core import database as db
from core.config import settings
from services.settings.history import SettingsHistory
from services.settings.seed import seed_settings
from services.settings.store import SettingsStore


def _probe_database() -> tuple[bool, str]:
    """True nếu PostgreSQL thật reachable (asyncpg, driver đã có sẵn)."""
    try:
        eng = create_async_engine(settings.database_url)

        async def _ping() -> None:
            async with eng.connect() as conn:
                await conn.execute(text("SELECT 1"))

        asyncio.run(_ping())
        asyncio.run(eng.dispose())
        return True, ""
    except Exception as exc:  # noqa: BLE001 — probe mọi lỗi connect
        return False, str(exc)


_db_available, _skip_reason = _probe_database()
if not _db_available:
    pytest.skip(f"PostgreSQL not reachable: {_skip_reason}", allow_module_level=True)


@pytest_asyncio.fixture
async def engine():
    """Function-scoped async engine trỏ schema braind_test (tạo + drop mỗi test)."""
    eng = create_async_engine(
        settings.database_url,
        echo=False,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": "braind_test"}},
    )

    async with eng.begin() as conn:
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS braind_test"))
        # Chỉ 4 bảng settings, né pgvector (R14)
        await conn.run_sync(
            db.Base.metadata.create_all,
            tables=[
                db.SettingDefinition.__table__,
                db.SettingValue.__table__,
                db.SettingHistory.__table__,
                db.SettingsSnapshot.__table__,
            ],
        )

    yield eng

    async with eng.begin() as conn:
        await conn.execute(text("DROP SCHEMA IF EXISTS braind_test CASCADE"))
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine):
    """Session factory cho test — reuse engine, mỗi test riêng session."""
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def store(session_factory):
    """SettingsStore (auditor=None — skip audit, chỉ cần 4 bảng)."""
    return SettingsStore(session_factory, auditor=None)


@pytest_asyncio.fixture
async def history(session_factory, store):
    """SettingsHistory delegate tới store."""
    return SettingsHistory(session_factory, store)


@pytest_asyncio.fixture
async def seeded(store, session_factory):
    """Seed catalog 1 lần cho test — idempotent, giống app boot."""
    report = await seed_settings(store, session_factory)
    return report