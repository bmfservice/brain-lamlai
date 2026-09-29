"""Settings REST API (Brain Second Phase 1c).

Provides endpoints for:
- Listing typed setting definitions with category filtering
- Reading and writing scoped values
- Audit history inspection and safe version rollback
- Effective view resolution (winning value + winning scope)
- Guarding sensitive and critical settings with X-Admin-Token
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from api.deps import (
    get_settings_cache,
    get_settings_history,
    get_settings_resolver,
    get_settings_store,
)
from core.config import settings
from services.settings.cache import SettingsCache
from services.settings.history import SettingsHistory
from services.settings.resolver import SettingsResolver
from services.settings.schema import (
    RiskLevel,
    SettingScope,
    SettingValueError,
)
from services.settings.store import SettingsStore

router = APIRouter(prefix="/api/settings", tags=["settings"])


# ---------------------------------------------------------------------------
# Pydantic Request Models
# ---------------------------------------------------------------------------


class SettingValueIn(BaseModel):
    """Payload to set a setting value at a scope."""

    value: Any = Field(..., description="Value matching the setting's declared data_type")
    scope: str = Field(default="project", description="Target scope in priority hierarchy")
    scope_ref: str | None = Field(
        default=None, description="Scope identifier (task_id, user_id, etc.)"
    )
    changed_by: str = Field(default="api", description="Actor performing the change")
    reason: str | None = Field(default=None, description="Optional explanation for change audit")


class RollbackIn(BaseModel):
    """Payload to roll back a setting to a previous version."""

    scope: str = Field(default="project", description="Target scope to rollback")
    scope_ref: str | None = Field(default=None, description="Scope identifier")
    changed_by: str = Field(default="api", description="Actor performing the rollback")
    reason: str | None = Field(default=None, description="Optional explanation for audit")


# ---------------------------------------------------------------------------
# Helper: Admin Token Security Gate
# ---------------------------------------------------------------------------


def _check_admin_token(
    spec: Any,
    scope: str,
    x_admin_token: str | None,
) -> None:
    """Validate X-Admin-Token when modifying sensitive, critical, or system-level settings."""
    is_sensitive = getattr(spec, "is_sensitive", False)
    is_critical = getattr(spec, "is_critical", False)
    risk_level = getattr(spec, "risk_level", None)
    is_critical_risk = (
        risk_level == RiskLevel.CRITICAL or str(risk_level).lower() == "critical"
    )
    is_privileged_scope = scope in (
        SettingScope.SYSTEM_HARD,
        SettingScope.SECURITY,
        "system_hard",
        "security",
    )

    if is_sensitive or is_critical or is_critical_risk or is_privileged_scope:
        expected_token = settings.settings_admin_token
        if expected_token:
            if not x_admin_token or x_admin_token != expected_token:
                raise HTTPException(
                    status_code=403,
                    detail="Valid X-Admin-Token required for sensitive or critical settings",
                )
        else:
            if not x_admin_token:
                raise HTTPException(
                    status_code=403,
                    detail="X-Admin-Token header required for sensitive or critical settings",
                )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/definitions")
async def list_definitions(
    category: str | None = Query(default=None, description="Filter by setting category"),
    store: Annotated[SettingsStore, Depends(get_settings_store)] = None,
) -> dict[str, Any]:
    """List setting definitions, optionally filtered by category."""
    specs = await store.list_definitions(category=category)
    return {"definitions": [spec.to_row() for spec in specs]}


@router.get("/values")
async def list_values(
    scope: str | None = Query(default=None, description="Filter by scope"),
    key: str | None = Query(default=None, description="Filter by setting key"),
    store: Annotated[SettingsStore, Depends(get_settings_store)] = None,
) -> dict[str, Any]:
    """List current setting values (unwrapped), optionally filtered."""
    values = await store.list_values(key=key, scope=scope)
    return {"values": values}


@router.put("/values/{key}")
async def set_value(
    key: str,
    body: SettingValueIn,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    store: Annotated[SettingsStore, Depends(get_settings_store)] = None,
    cache: Annotated[SettingsCache, Depends(get_settings_cache)] = None,
) -> dict[str, Any]:
    """Set setting value at the specified scope (checked against hard rules & typed schema)."""
    spec = await store.get_definition(key)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown setting key: {key!r} (not in catalog)",
        )

    _check_admin_token(spec, body.scope, x_admin_token)

    try:
        version = await store.set_value(
            key,
            body.value,
            scope=body.scope,
            scope_ref=body.scope_ref,
            changed_by=body.changed_by,
            reason=body.reason,
        )
    except SettingValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if cache is not None:
        cache.invalidate(key)

    return {
        "key": key,
        "scope": body.scope,
        "scope_ref": body.scope_ref,
        "version": version,
        "status": "ok",
    }


@router.delete("/values/{key}")
async def delete_value(
    key: str,
    scope: str = Query(default="project", description="Scope to delete override from"),
    scope_ref: str | None = Query(default=None, description="Scope identifier"),
    changed_by: str = Query(default="api", description="Actor deleting the override"),
    reason: str | None = Query(default=None, description="Reason for deletion"),
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    store: Annotated[SettingsStore, Depends(get_settings_store)] = None,
    cache: Annotated[SettingsCache, Depends(get_settings_cache)] = None,
) -> dict[str, Any]:
    """Delete a scoped setting override (records tombstone in history)."""
    spec = await store.get_definition(key)
    if spec is not None:
        _check_admin_token(spec, scope, x_admin_token)

    deleted = await store.delete_value(
        key,
        scope=scope,
        scope_ref=scope_ref,
        changed_by=changed_by,
        reason=reason,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="setting value not found")

    if cache is not None:
        cache.invalidate(key)

    return {
        "key": key,
        "scope": scope,
        "scope_ref": scope_ref,
        "deleted": True,
    }


@router.get("/history/{key}")
async def get_history(
    key: str,
    scope: str | None = Query(default=None, description="Filter by scope"),
    scope_ref: str | None = Query(default=None, description="Filter by scope_ref"),
    limit: int = Query(default=50, ge=1, le=500, description="Max history entries to return"),
    store: Annotated[SettingsStore, Depends(get_settings_store)] = None,
    history: Annotated[SettingsHistory, Depends(get_settings_history)] = None,
) -> dict[str, Any]:
    """Get version change history for a setting key."""
    spec = await store.get_definition(key)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown setting key: {key!r} (not in catalog)",
        )

    entries = await history.list_history(
        key,
        scope=scope,
        scope_ref=scope_ref,
        limit=limit,
    )
    return {
        "key": key,
        "history": [
            {
                "version": e.version,
                "scope": e.scope,
                "scope_ref": e.scope_ref,
                "previous_value": e.previous_value,
                "new_value": e.new_value,
                "changed_at": (
                    e.changed_at.isoformat()
                    if hasattr(e.changed_at, "isoformat")
                    else str(e.changed_at)
                ),
                "changed_by": e.changed_by,
                "reason": e.reason,
            }
            for e in entries
        ],
    }


@router.post("/history/{key}/rollback/{version}")
async def rollback_version(
    key: str,
    version: int,
    body: RollbackIn | None = None,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    store: Annotated[SettingsStore, Depends(get_settings_store)] = None,
    history: Annotated[SettingsHistory, Depends(get_settings_history)] = None,
    cache: Annotated[SettingsCache, Depends(get_settings_cache)] = None,
) -> dict[str, Any]:
    """Roll back a setting value to a historical version (creates a new version)."""
    payload = body or RollbackIn()
    spec = await store.get_definition(key)
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown setting key: {key!r} (not in catalog)",
        )

    _check_admin_token(spec, payload.scope, x_admin_token)

    try:
        new_version = await history.rollback(
            key,
            to_version=version,
            scope=payload.scope,
            scope_ref=payload.scope_ref,
            changed_by=payload.changed_by,
            reason=payload.reason,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, SettingValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if cache is not None:
        cache.invalidate(key)

    return {
        "key": key,
        "rolled_back_to": version,
        "new_version": new_version,
        "status": "ok",
    }


@router.get("/effective")
async def get_effective_settings(
    task_id: str | None = Query(default=None, description="Optional task ID for resolution"),
    session_id: str | None = Query(default=None, description="Optional session ID for resolution"),
    user_id: str | None = Query(default=None, description="Optional user ID for resolution"),
    resolver: Annotated[SettingsResolver, Depends(get_settings_resolver)] = None,
    cache: Annotated[SettingsCache, Depends(get_settings_cache)] = None,
) -> dict[str, Any]:
    """Return effective resolved settings + winning scope for UI Effective View."""
    cache_key = f"effective:{task_id}:{session_id}:{user_id}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    values, resolved_from = await resolver.resolve_all(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
    )
    result = {
        "values": values,
        "resolved_from": resolved_from,
    }

    if cache is not None:
        cache.set(cache_key, result, ttl_seconds=30.0)

    return result
