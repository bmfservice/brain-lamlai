"""Seed the settings catalog into the DB (Brain Second Phase 1a).

Idempotent by construction, following the existing ``RegistryService.
ensure_system_entries`` convention (pg_insert + on_conflict_do_update):

- every ``SettingSpec`` in the catalog is upserted into ``setting_definitions``;
- a PROJECT-scope ``setting_values`` row is created **only if absent**, so a value
  an operator already edited is never overwritten by a re-seed.

Run from ``api/main.py::lifespan`` at startup (guarded by
``settings.settings_seed_enabled``). Returns a report so the caller can log how
much was created vs skipped — a second boot must report ``values=0``.

Seeding uses the store's own ``set_value`` for the initial PROJECT value so the
first change is recorded in history with ``changed_by='seed'``: the audit trail
starts at the seed, with no unexplained gap.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from core import database as db
from services.settings.catalog import CATALOG
from services.settings.schema import SettingScope


@dataclass(frozen=True)
class SeedReport:
    """How much a seed run created vs left alone."""

    definitions: int
    values: int
    skipped_values: int


async def seed_settings(store, session_factory) -> SeedReport:
    """Upsert all definitions and create missing PROJECT-scope default values."""
    definitions = 0
    values = 0
    skipped = 0

    for spec in CATALOG:
        await store.upsert_definition(spec)
        definitions += 1

    # Create the PROJECT-scope current value only where none exists yet.
    async with session_factory() as session:
        existing = set(
            (
                await session.execute(
                    select(db.SettingValue.key).where(
                        db.SettingValue.scope == str(SettingScope.PROJECT)
                    )
                )
            ).scalars().all()
        )

    for spec in CATALOG:
        if spec.key in existing:
            skipped += 1
            continue
        # Goes through set_value so validation + history + audit all apply to the
        # seed write exactly as they do to a later human edit.
        await store.set_value(
            spec.key,
            spec.default_value,
            scope=SettingScope.PROJECT,
            changed_by="seed",
            reason="initial catalog seed",
        )
        values += 1

    report = SeedReport(definitions=definitions, values=values, skipped_values=skipped)
    # Caller logs the report (api/main.py), matching the registry.system_seeded
    # convention — so there is exactly one settings.seeded line per boot.
    return report
