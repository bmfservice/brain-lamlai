"""Settings resolver — 7-level Priority Hierarchy (Brain Second Phase 1b).

Resolves the effective setting value for any key by traversing the 7-tier
priority hierarchy in descending order:
    SYSTEM_HARD > SECURITY > PROJECT > TASK > USER_PREFERENCE > TEMPORARY > LEARNED

If no scope has an override, falls back to the definition's default value.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select

from core import database as db
from core.logging import get_logger
from services.settings.schema import (
    SettingScope,
    SettingValueError,
    _unwrap,
)
from services.settings.store import SettingsStore

logger = get_logger(__name__)

# Scopes in descending priority order
_SCOPE_PRECEDENCE: tuple[SettingScope, ...] = tuple(SettingScope)


def _matches_scope_ref(
    scope: str,
    scope_ref: str | None,
    *,
    task_id: str | None,
    session_id: str | None,
    user_id: str | None,
) -> bool:
    """Return True if row's scope_ref matches the given resolution context."""
    if scope_ref is None:
        return True

    if scope == SettingScope.TASK.value:
        return task_id is not None and str(task_id) == scope_ref
    if scope == SettingScope.USER_PREFERENCE.value:
        return user_id is not None and str(user_id) == scope_ref
    if scope == SettingScope.TEMPORARY.value:
        return session_id is not None and str(session_id) == scope_ref
    if scope == SettingScope.LEARNED.value:
        context_ids = {str(x) for x in (task_id, session_id, user_id) if x is not None}
        return scope_ref in context_ids

    return True


class SettingsResolver:
    """7-tier priority hierarchy resolver for settings."""

    def __init__(self, session_factory, store: SettingsStore | None = None) -> None:
        self.session_factory = session_factory
        self.store = store or SettingsStore(session_factory)

    async def resolve(
        self,
        key: str,
        *,
        task_id: str | uuid.UUID | None = None,
        session_id: str | uuid.UUID | None = None,
        user_id: str | uuid.UUID | None = None,
    ) -> tuple[Any, SettingScope | str]:
        """Resolve the winning value and winning scope for one key.

        Returns (winning_value, winning_scope), falling back to definition default.
        Raises SettingValueError if key is not defined in catalog.
        """
        spec = await self.store.get_definition(key)
        if spec is None:
            raise SettingValueError(f"unknown setting key: {key!r} (not in catalog)")

        task_id_str = str(task_id) if task_id is not None else None
        session_id_str = str(session_id) if session_id is not None else None
        user_id_str = str(user_id) if user_id is not None else None

        async with self.session_factory() as session:
            stmt = select(db.SettingValue).where(db.SettingValue.key == key)
            rows = (await session.execute(stmt)).scalars().all()

        # Group rows by scope
        rows_by_scope: dict[str, list[db.SettingValue]] = {}
        for r in rows:
            rows_by_scope.setdefault(r.scope, []).append(r)

        # Walk through scopes in descending precedence
        for scope in _SCOPE_PRECEDENCE:
            scope_rows = rows_by_scope.get(scope.value, [])
            if not scope_rows:
                continue

            # Prefer exact scope_ref match over global (scope_ref=None)
            exact_match: db.SettingValue | None = None
            fallback_match: db.SettingValue | None = None

            for r in scope_rows:
                if not _matches_scope_ref(
                    r.scope,
                    r.scope_ref,
                    task_id=task_id_str,
                    session_id=session_id_str,
                    user_id=user_id_str,
                ):
                    continue

                if r.scope_ref is not None:
                    exact_match = r
                    break
                fallback_match = r

            matched = exact_match or fallback_match
            if matched is not None:
                return _unwrap(matched.value), scope

        return spec.default_value, "default"

    async def resolve_all(
        self,
        *,
        task_id: str | uuid.UUID | None = None,
        session_id: str | uuid.UUID | None = None,
        user_id: str | uuid.UUID | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Resolve all setting definitions in the catalog.

        Returns (resolved_values, resolved_from) maps.
        """
        specs = await self.store.list_definitions()
        if not specs:
            from services.settings.catalog import CATALOG

            specs = list(CATALOG)

        task_id_str = str(task_id) if task_id is not None else None
        session_id_str = str(session_id) if session_id is not None else None
        user_id_str = str(user_id) if user_id is not None else None

        async with self.session_factory() as session:
            stmt = select(db.SettingValue)
            rows = (await session.execute(stmt)).scalars().all()

        # Map (key, scope) -> list of rows
        values_by_key_scope: dict[tuple[str, str], list[db.SettingValue]] = {}
        for r in rows:
            values_by_key_scope.setdefault((r.key, r.scope), []).append(r)

        resolved_values: dict[str, Any] = {}
        resolved_from: dict[str, str] = {}

        for spec in specs:
            winner_value = spec.default_value
            winner_scope = "default"

            for scope in _SCOPE_PRECEDENCE:
                scope_rows = values_by_key_scope.get((spec.key, scope.value), [])
                if not scope_rows:
                    continue

                exact_match: db.SettingValue | None = None
                fallback_match: db.SettingValue | None = None

                for r in scope_rows:
                    if not _matches_scope_ref(
                        r.scope,
                        r.scope_ref,
                        task_id=task_id_str,
                        session_id=session_id_str,
                        user_id=user_id_str,
                    ):
                        continue

                    if r.scope_ref is not None:
                        exact_match = r
                        break
                    fallback_match = r

                matched = exact_match or fallback_match
                if matched is not None:
                    winner_value = _unwrap(matched.value)
                    winner_scope = str(scope)
                    break

            resolved_values[spec.key] = winner_value
            resolved_from[spec.key] = winner_scope

        return resolved_values, resolved_from
