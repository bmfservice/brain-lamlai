"""Robinson UI registry routes.

Expose navigation, command-palette items, dashboard widgets, and writer option
lists from the dynamic registry.  The frontend must load these lists from the
API rather than hardcoding them.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from api.deps import get_registry
from services.registry.service import RegistryService

router = APIRouter(tags=["ui", "registry"])


class RegistryEntryOut(BaseModel):
    """Serialized registry entry returned to the UI."""

    id: str
    kind: str
    key: str
    label: str
    description: str | None = None
    icon: str | None = None
    href: str | None = None
    action: str | None = None
    group_name: str | None = None
    sort_order: int = 0
    enabled: bool = True
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class RegistryListResponse(BaseModel):
    """A list of registry entries for a given kind."""

    kind: str
    items: list[RegistryEntryOut]
    count: int


class WriterOptionsResponse(BaseModel):
    """Grouped writer option lists for the content-generation form."""

    content_type: list[RegistryEntryOut] = Field(default_factory=list)
    tone: list[RegistryEntryOut] = Field(default_factory=list)
    audience: list[RegistryEntryOut] = Field(default_factory=list)
    output_format: list[RegistryEntryOut] = Field(default_factory=list)
    language: list[RegistryEntryOut] = Field(default_factory=list)


@router.get("/api/ui/nav", response_model=RegistryListResponse)
async def get_nav(
    registry: Annotated[RegistryService, Depends(get_registry)],
) -> RegistryListResponse:
    """Return enabled navigation items for the Robinson shell."""
    items = await registry.list_entries(kind="nav")
    return RegistryListResponse(
        kind="nav",
        items=[RegistryEntryOut(**item) for item in items],
        count=len(items),
    )


@router.get("/api/ui/commands", response_model=RegistryListResponse)
async def get_commands(
    registry: Annotated[RegistryService, Depends(get_registry)],
) -> RegistryListResponse:
    """Return enabled command-palette entries."""
    items = await registry.list_entries(kind="command")
    return RegistryListResponse(
        kind="command",
        items=[RegistryEntryOut(**item) for item in items],
        count=len(items),
    )


@router.get("/api/ui/widgets", response_model=RegistryListResponse)
async def get_widgets(
    registry: Annotated[RegistryService, Depends(get_registry)],
) -> RegistryListResponse:
    """Return enabled dashboard widgets."""
    items = await registry.list_entries(kind="widget")
    return RegistryListResponse(
        kind="widget",
        items=[RegistryEntryOut(**item) for item in items],
        count=len(items),
    )


@router.get("/api/ui/writer-options", response_model=WriterOptionsResponse)
async def get_writer_options(
    registry: Annotated[RegistryService, Depends(get_registry)],
) -> WriterOptionsResponse:
    """Return writer option lists grouped by field."""
    items = await registry.list_entries(kind="writer_option")
    grouped: dict[str, list[RegistryEntryOut]] = {
        "content_type": [],
        "tone": [],
        "audience": [],
        "output_format": [],
        "language": [],
    }
    for item in items:
        group = item.get("group_name") or ""
        if group in grouped:
            grouped[group].append(RegistryEntryOut(**item))
    return WriterOptionsResponse(**grouped)


@router.get("/api/ui/registry", response_model=RegistryListResponse)
async def list_registry(
    registry: Annotated[RegistryService, Depends(get_registry)],
    kind: str | None = Query(default=None),
    group: str | None = Query(default=None),
) -> RegistryListResponse:
    """Generic registry list endpoint with optional filters."""
    items = await registry.list_entries(kind=kind, group_name=group)
    return RegistryListResponse(
        kind=kind or "all",
        items=[RegistryEntryOut(**item) for item in items],
        count=len(items),
    )
