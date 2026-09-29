"""Tests for Settings REST API (Brain Second Phase 1c).

Tests all endpoints:
- GET /api/settings/definitions (all, filtered by category)
- GET /api/settings/values (all, filtered by key/scope)
- PUT /api/settings/values/{key} (valid update, invalid value 400, unknown key 404, admin token 403)
- DELETE /api/settings/values/{key} (delete override, 404)
- GET /api/settings/history/{key} (history trail, 404)
- POST /api/settings/history/{key}/rollback/{version} (rollback version, 404/400)
- GET /api/settings/effective (effective view + caching)
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from api.deps import (
    get_settings_cache,
    get_settings_history,
    get_settings_resolver,
    get_settings_store,
)
from api.routes_settings import router
from core.config import settings
from services.settings.cache import SettingsCache
from services.settings.resolver import SettingsResolver
from services.settings.schema import SettingScope

pytest_plugins = ["tests.integration.conftest"]


@pytest_asyncio.fixture
async def resolver(session_factory, store):
    return SettingsResolver(session_factory, store=store)


@pytest_asyncio.fixture
async def cache():
    return SettingsCache(default_ttl_seconds=60.0)


@pytest_asyncio.fixture
async def test_app(store, history, resolver, cache):
    """Minimal FastAPI test app with routes_settings and injected dependencies."""
    app = FastAPI()
    app.include_router(router)

    app.dependency_overrides[get_settings_store] = lambda: store
    app.dependency_overrides[get_settings_history] = lambda: history
    app.dependency_overrides[get_settings_resolver] = lambda: resolver
    app.dependency_overrides[get_settings_cache] = lambda: cache

    return app


@pytest_asyncio.fixture
async def client(test_app):
    transport = httpx.ASGITransport(app=test_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Definitions Endpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_definitions(client, seeded):
    """GET /api/settings/definitions returns all definitions."""
    resp = await client.get("/api/settings/definitions")
    assert resp.status_code == 200
    data = resp.json()
    assert "definitions" in data
    assert len(data["definitions"]) >= 116

    # Verify a known definition is returned with all typed metadata
    keys = {d["key"] for d in data["definitions"]}
    assert "reasoning_depth" in keys


@pytest.mark.asyncio
async def test_get_definitions_filtered_by_category(client, seeded):
    """GET /api/settings/definitions?category=... returns filtered definitions."""
    resp = await client.get("/api/settings/definitions?category=core_behavior")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["definitions"]) > 0
    for d in data["definitions"]:
        assert d["category"] == "core_behavior"


# ---------------------------------------------------------------------------
# Values Endpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_values_all_and_filtered(client, seeded):
    """GET /api/settings/values returns current values, with key and scope filtering."""
    # All values
    resp = await client.get("/api/settings/values")
    assert resp.status_code == 200
    data = resp.json()
    assert "values" in data
    assert len(data["values"]) >= 116

    # Filtered by key
    resp_key = await client.get("/api/settings/values?key=reasoning_depth")
    assert resp_key.status_code == 200
    items = resp_key.json()["values"]
    assert len(items) == 1
    assert items[0]["key"] == "reasoning_depth"
    assert items[0]["value"] == "deep"

    # Filtered by scope
    resp_scope = await client.get("/api/settings/values?scope=project")
    assert resp_scope.status_code == 200
    assert len(resp_scope.json()["values"]) >= 116


@pytest.mark.asyncio
async def test_put_value_success_and_cache_invalidation(client, cache, seeded):
    """PUT /api/settings/values/{key} sets a value and invalidates cache."""
    key = "reasoning_depth"
    cache.set(f"{key}:test", "stale_cached_value")

    payload = {
        "value": "normal",
        "scope": "project",
        "changed_by": "tester",
        "reason": "testing PUT endpoint",
    }
    resp = await client.put(f"/api/settings/values/{key}", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["key"] == key
    assert data["status"] == "ok"
    assert data["version"] >= 2

    # Verify cache was invalidated
    assert cache.get(f"{key}:test") is None

    # Verify value changed in GET
    resp_get = await client.get(f"/api/settings/values?key={key}")
    assert resp_get.json()["values"][0]["value"] == "normal"


@pytest.mark.asyncio
async def test_put_value_invalid_returns_400(client, seeded):
    """PUT /api/settings/values/{key} with invalid value returns 400 Bad Request."""
    key = "max_retry_count"
    # max_retry_count has max_value 5, so 99 is invalid
    payload = {
        "value": 99,
        "scope": "project",
    }
    resp = await client.put(f"/api/settings/values/{key}", json=payload)
    assert resp.status_code == 400
    assert "> max" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_put_value_unknown_key_returns_404(client, seeded):
    """PUT /api/settings/values/{key} for an unknown key returns 404."""
    resp = await client.put(
        "/api/settings/values/non_existent_key_xyz",
        json={"value": "test", "scope": "project"},
    )
    assert resp.status_code == 404
    assert "unknown setting key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_put_value_admin_token_guard(client, seeded, monkeypatch):
    """PUT /api/settings/values/{key} for critical scope requires valid X-Admin-Token."""
    monkeypatch.setattr(settings, "settings_admin_token", "super_secret_token_123")
    key = "reasoning_depth"

    # 1. Privileged scope 'system_hard' without token -> 403
    payload = {
        "value": "deep",
        "scope": "system_hard",
        "changed_by": "intruder",
    }
    resp = await client.put(f"/api/settings/values/{key}", json=payload)
    assert resp.status_code == 403

    # 2. With invalid token -> 403
    resp_bad = await client.put(
        f"/api/settings/values/{key}",
        json=payload,
        headers={"X-Admin-Token": "wrong_token"},
    )
    assert resp_bad.status_code == 403

    # 3. With valid token -> 200
    resp_good = await client.put(
        f"/api/settings/values/{key}",
        json=payload,
        headers={"X-Admin-Token": "super_secret_token_123"},
    )
    assert resp_good.status_code == 200
    assert resp_good.json()["scope"] == "system_hard"


# ---------------------------------------------------------------------------
# Delete Endpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_value_override(client, store, seeded):
    """DELETE /api/settings/values/{key} removes scoped override."""
    key = "reasoning_depth"
    task_id = "task-delete-test"

    # Add a task override
    await store.set_value(
        key, "low", scope=SettingScope.TASK, scope_ref=task_id, changed_by="test"
    )

    # Delete the task override
    resp = await client.delete(
        f"/api/settings/values/{key}?scope=task&scope_ref={task_id}"
    )
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # Deleting again returns 404 Not Found
    resp_again = await client.delete(
        f"/api/settings/values/{key}?scope=task&scope_ref={task_id}"
    )
    assert resp_again.status_code == 404


# ---------------------------------------------------------------------------
# History and Rollback Endpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_and_rollback(client, seeded):
    """GET /api/settings/history/{key} and POST rollback."""
    key = "reasoning_depth"

    # 1. Update twice to produce history
    await client.put(
        f"/api/settings/values/{key}",
        json={"value": "normal", "scope": "project", "reason": "step 1"},
    )
    await client.put(
        f"/api/settings/values/{key}",
        json={"value": "low", "scope": "project", "reason": "step 2"},
    )

    # 2. Inspect history
    resp_hist = await client.get(f"/api/settings/history/{key}")
    assert resp_hist.status_code == 200
    entries = resp_hist.json()["history"]
    assert len(entries) >= 3  # seed(v1), normal(v2), low(v3)
    assert entries[0]["new_value"] == "low"

    # 3. Rollback to version 1 (initial 'deep')
    resp_rb = await client.post(
        f"/api/settings/history/{key}/rollback/1",
        json={"scope": "project", "reason": "restore default"},
    )
    assert resp_rb.status_code == 200
    rb_data = resp_rb.json()
    assert rb_data["rolled_back_to"] == 1
    assert rb_data["new_version"] == entries[0]["version"] + 1

    # Verify current value is back to 'deep'
    resp_curr = await client.get(f"/api/settings/values?key={key}")
    assert resp_curr.json()["values"][0]["value"] == "deep"


@pytest.mark.asyncio
async def test_rollback_nonexistent_version_returns_404(client, seeded):
    """Rolling back to a non-existent version returns 404."""
    resp = await client.post(
        "/api/settings/history/reasoning_depth/rollback/9999",
        json={"scope": "project"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Effective View Endpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_effective_settings_and_cache(client, store, cache, seeded):
    """GET /api/settings/effective returns resolved settings and winning scopes."""
    task_id = "task-effective-test"
    key = "reasoning_depth"

    # Set task override to 'low'
    await store.set_value(
        key, "low", scope=SettingScope.TASK, scope_ref=task_id, changed_by="test"
    )

    # Note: PROJECT (deep) beats TASK (low). To verify TASK wins, delete PROJECT scope
    await store.delete_value(key, scope=SettingScope.PROJECT, changed_by="test")

    resp = await client.get(f"/api/settings/effective?task_id={task_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "values" in data
    assert "resolved_from" in data
    assert data["values"][key] == "low"
    assert data["resolved_from"][key] == "task"

    # Verify response was cached
    cache_key = f"effective:{task_id}:None:None"
    cached = cache.get(cache_key)
    assert cached is not None
    assert cached["values"][key] == "low"
