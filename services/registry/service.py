"""Robinson dynamic registry service.

The registry owns all user-facing UI/system configuration that is currently
hardcoded in the frontend (navigation, commands, writer option lists, model
profiles, etc.).  Every business list is stored as a ``RegistryEntry`` row and
exposed through a small, versioned API surface.

The registry uses :class:`core.database.RegistryEntry` as its single source of
truth.  On first call the service seeds a set of built-in system entries so
that a fresh database still renders a usable UI.
"""

from __future__ import annotations

from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from core.database import RegistryEntry
from core.logging import get_logger

logger = get_logger(__name__)

RegistryKind = Literal["nav", "command", "widget", "model", "status", "writer_option"]


# Built-in system entries.  Kept here as code so the database remains the
# source of truth at runtime, but the system still works on a fresh install.
# These entries are upserted idempotently using (kind, key) as the stable
# natural key.  User-defined entries can override system entries by sharing
# the same key and a higher ``sort_order`` / ``enabled`` flag.
_SYSTEM_REGISTRY: list[dict[str, Any]] = [
    # Navigation
    {
        "kind": "nav",
        "key": "chat",
        "label": "Chat",
        "icon": "fa-comments",
        "action": "switchView('chat')",
        "group_name": "main",
        "sort_order": 10,
    },
    {
        "kind": "nav",
        "key": "memory",
        "label": "Bộ nhớ",
        "icon": "fa-project-diagram",
        "action": "switchView('memory')",
        "group_name": "main",
        "sort_order": 20,
    },
    {
        "kind": "nav",
        "key": "ingest",
        "label": "Nạp kiến thức",
        "icon": "fa-upload",
        "action": "switchView('ingest')",
        "group_name": "main",
        "sort_order": 30,
    },
    {
        "kind": "nav",
        "key": "writer",
        "label": "Viết nội dung",
        "icon": "fa-pen-nib",
        "action": "switchView('writer')",
        "group_name": "main",
        "sort_order": 40,
    },
    {
        "kind": "nav",
        "key": "learn",
        "label": "Tự học",
        "icon": "fa-globe",
        "action": "switchView('learn')",
        "group_name": "main",
        "sort_order": 50,
    },
    # Commands (command palette)
    {
        "kind": "command",
        "key": "new-conversation",
        "label": "Cuộc trò chuyện mới",
        "icon": "fa-plus",
        "action": "newConversation()",
        "group_name": "chat",
        "sort_order": 10,
    },
    {
        "kind": "command",
        "key": "load-conversations",
        "label": "Tải lịch sử trò chuyện",
        "icon": "fa-clock-rotate-left",
        "action": "loadConversations()",
        "group_name": "chat",
        "sort_order": 20,
    },
    {
        "kind": "command",
        "key": "sync-vault",
        "label": "Đồng bộ vault Obsidian",
        "icon": "fa-rotate",
        "action": "syncVault()",
        "group_name": "ingest",
        "sort_order": 30,
    },
    {
        "kind": "command",
        "key": "open-writer",
        "label": "Mở trang viết nội dung",
        "icon": "fa-pen-nib",
        "action": "switchView('writer')",
        "group_name": "writer",
        "sort_order": 40,
    },
    {
        "kind": "command",
        "key": "open-learn",
        "label": "Mở máy tự học",
        "icon": "fa-brain",
        "action": "switchView('learn')",
        "group_name": "learn",
        "sort_order": 50,
    },
    {
        "kind": "command",
        "key": "reload-memory",
        "label": "Tải lại đồ thị bộ nhớ",
        "icon": "fa-sync",
        "action": "loadMemoryGraph()",
        "group_name": "memory",
        "sort_order": 60,
    },
    # Widgets (dashboard cards / HUD)
    {
        "kind": "widget",
        "key": "status-pill",
        "label": "Trạng thái hệ thống",
        "icon": "fa-circle",
        "group_name": "status",
        "sort_order": 10,
        "meta": {"component": "StatusPill", "poll": "/api/health"},
    },
    {
        "kind": "widget",
        "key": "quick-stats",
        "label": "Thống kê nhanh",
        "icon": "fa-chart-bar",
        "group_name": "dashboard",
        "sort_order": 20,
        "meta": {"component": "QuickStats", "poll": "/api/metrics"},
    },
    # Writer option lists (formerly hardcoded in api/schemas.py)
    {
        "kind": "writer_option",
        "key": "content_type::blog",
        "label": "Bài viết blog",
        "group_name": "content_type",
        "sort_order": 10,
        "meta": {"value": "blog"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::social",
        "label": "Bài đăng mạng xã hội",
        "group_name": "content_type",
        "sort_order": 20,
        "meta": {"value": "social"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::email",
        "label": "Email marketing / newsletter",
        "group_name": "content_type",
        "sort_order": 30,
        "meta": {"value": "email"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::ad",
        "label": "Quảng cáo (ad copy)",
        "group_name": "content_type",
        "sort_order": 40,
        "meta": {"value": "ad"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::seo",
        "label": "Nội dung SEO",
        "group_name": "content_type",
        "sort_order": 50,
        "meta": {"value": "seo"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::summary",
        "label": "Tóm tắt nội dung",
        "group_name": "content_type",
        "sort_order": 60,
        "meta": {"value": "summary"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::newsletter",
        "label": "Bản tin định kỳ",
        "group_name": "content_type",
        "sort_order": 70,
        "meta": {"value": "newsletter"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::case_study",
        "label": "Case study",
        "group_name": "content_type",
        "sort_order": 80,
        "meta": {"value": "case_study"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::product_desc",
        "label": "Mô tả sản phẩm",
        "group_name": "content_type",
        "sort_order": 90,
        "meta": {"value": "product_desc"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::press_release",
        "label": "Thông cáo báo chí",
        "group_name": "content_type",
        "sort_order": 100,
        "meta": {"value": "press_release"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::thread",
        "label": "Thread",
        "group_name": "content_type",
        "sort_order": 110,
        "meta": {"value": "thread"},
    },
    {
        "kind": "writer_option",
        "key": "content_type::landing_page",
        "label": "Landing page",
        "group_name": "content_type",
        "sort_order": 120,
        "meta": {"value": "landing_page"},
    },
    {
        "kind": "writer_option",
        "key": "tone::chuyen_nghiep",
        "label": "Chuyên nghiệp",
        "group_name": "tone",
        "sort_order": 10,
        "meta": {"value": "chuyên nghiệp"},
    },
    {
        "kind": "writer_option",
        "key": "tone::than_thien",
        "label": "Thân thiện",
        "group_name": "tone",
        "sort_order": 20,
        "meta": {"value": "thân thiện"},
    },
    {
        "kind": "writer_option",
        "key": "tone::trang_trong",
        "label": "Trang trọng",
        "group_name": "tone",
        "sort_order": 30,
        "meta": {"value": "trang trọng"},
    },
    {
        "kind": "writer_option",
        "key": "tone::hai_huoc",
        "label": "Hài hước",
        "group_name": "tone",
        "sort_order": 40,
        "meta": {"value": "hài hước"},
    },
    {
        "kind": "writer_option",
        "key": "tone::thuyet_phuc",
        "label": "Thuyết phục",
        "group_name": "tone",
        "sort_order": 50,
        "meta": {"value": "thuyết phục"},
    },
    {
        "kind": "writer_option",
        "key": "tone::truyen_cam_hung",
        "label": "Truyền cảm hứng",
        "group_name": "tone",
        "sort_order": 60,
        "meta": {"value": "truyền cảm hứng"},
    },
    {
        "kind": "writer_option",
        "key": "tone::binh_dan",
        "label": "Bình dân",
        "group_name": "tone",
        "sort_order": 70,
        "meta": {"value": "bình dân"},
    },
    {
        "kind": "writer_option",
        "key": "audience::nguoi_moi",
        "label": "Người mới",
        "group_name": "audience",
        "sort_order": 10,
        "meta": {"value": "người mới"},
    },
    {
        "kind": "writer_option",
        "key": "audience::chuyen_gia",
        "label": "Chuyên gia",
        "group_name": "audience",
        "sort_order": 20,
        "meta": {"value": "chuyên gia"},
    },
    {
        "kind": "writer_option",
        "key": "audience::lanh_dao",
        "label": "Lãnh đạo",
        "group_name": "audience",
        "sort_order": 30,
        "meta": {"value": "lãnh đạo"},
    },
    {
        "kind": "writer_option",
        "key": "audience::sinh_vien",
        "label": "Sinh viên",
        "group_name": "audience",
        "sort_order": 40,
        "meta": {"value": "sinh viên"},
    },
    {
        "kind": "writer_option",
        "key": "audience::b2b",
        "label": "Khách hàng B2B",
        "group_name": "audience",
        "sort_order": 50,
        "meta": {"value": "khách hàng B2B"},
    },
    {
        "kind": "writer_option",
        "key": "audience::b2c",
        "label": "Khách hàng B2C",
        "group_name": "audience",
        "sort_order": 60,
        "meta": {"value": "khách hàng B2C"},
    },
    {
        "kind": "writer_option",
        "key": "output_format::markdown",
        "label": "Markdown",
        "group_name": "output_format",
        "sort_order": 10,
        "meta": {"value": "markdown"},
    },
    {
        "kind": "writer_option",
        "key": "output_format::plain",
        "label": "Plain text",
        "group_name": "output_format",
        "sort_order": 20,
        "meta": {"value": "plain"},
    },
    {
        "kind": "writer_option",
        "key": "output_format::html",
        "label": "HTML",
        "group_name": "output_format",
        "sort_order": 30,
        "meta": {"value": "html"},
    },
    {
        "kind": "writer_option",
        "key": "language::vi",
        "label": "Tiếng Việt",
        "group_name": "language",
        "sort_order": 10,
        "meta": {"value": "vi"},
    },
    {
        "kind": "writer_option",
        "key": "language::en",
        "label": "English",
        "group_name": "language",
        "sort_order": 20,
        "meta": {"value": "en"},
    },
    {
        "kind": "writer_option",
        "key": "language::vi_en",
        "label": "Song ngữ Việt-Anh",
        "group_name": "language",
        "sort_order": 30,
        "meta": {"value": "vi-en"},
    },
]


class RegistryService:
    """Dynamic registry for UI configuration and system option lists."""

    def __init__(self, session_factory: Any) -> None:
        """Initialize the registry service.

        Args:
            session_factory: An async callable yielding a SQLAlchemy session.
        """
        self.session_factory = session_factory

    async def ensure_system_entries(self) -> int:
        """Upsert built-in system registry entries.

        Returns:
            Number of entries upserted/ensured.
        """
        async with self.session_factory() as session:
            for entry in _SYSTEM_REGISTRY:
                values = {
                    **entry,
                    "meta": entry.get("meta") or {},
                }
                stmt = (
                    pg_insert(RegistryEntry)
                    .values(values)
                    .on_conflict_do_update(
                        index_elements=["kind", "key"],
                        set_={
                            "label": values["label"],
                            "description": values.get("description"),
                            "icon": values.get("icon"),
                            "href": values.get("href"),
                            "action": values.get("action"),
                            "group_name": values.get("group_name"),
                            "sort_order": values["sort_order"],
                            "enabled": values.get("enabled", True),
                            "meta": values["meta"],
                        },
                    )
                )
                await session.execute(stmt)
            await session.commit()
        return len(_SYSTEM_REGISTRY)

    async def list_entries(
        self,
        kind: str | None = None,
        group_name: str | None = None,
        enabled_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Return registry entries ordered by ``kind``, ``group_name``, ``sort_order``.

        Args:
            kind: Optional kind filter.
            group_name: Optional group_name filter.
            enabled_only: If ``True`` (default), only enabled entries are returned.

        Returns:
            A list of serialized registry entries.
        """
        async with self.session_factory() as session:
            stmt = select(RegistryEntry)
            if enabled_only:
                stmt = stmt.where(RegistryEntry.enabled.is_(True))
            if kind:
                stmt = stmt.where(RegistryEntry.kind == kind)
            if group_name:
                stmt = stmt.where(RegistryEntry.group_name == group_name)
            stmt = stmt.order_by(
                RegistryEntry.kind,
                RegistryEntry.group_name,
                RegistryEntry.sort_order,
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_serialize(entry) for entry in rows]

    async def get_entry(self, kind: str, key: str) -> dict[str, Any] | None:
        """Return a single registry entry by natural key.

        Args:
            kind: Entry kind.
            key: Entry key.

        Returns:
            Serialized entry or ``None``.
        """
        async with self.session_factory() as session:
            stmt = select(RegistryEntry).where(
                RegistryEntry.kind == kind, RegistryEntry.key == key
            )
            row = (await session.execute(stmt)).scalars().first()
            return _serialize(row) if row else None

    async def upsert_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Upsert a registry entry.

        Args:
            payload: Must contain ``kind`` and ``key``; other keys are optional.

        Returns:
            The upserted entry as a dictionary.
        """
        async with self.session_factory() as session:
            values = {
                "kind": payload["kind"],
                "key": payload["key"],
                "label": payload.get("label", ""),
                "description": payload.get("description"),
                "icon": payload.get("icon"),
                "href": payload.get("href"),
                "action": payload.get("action"),
                "group_name": payload.get("group_name"),
                "sort_order": payload.get("sort_order", 0),
                "enabled": payload.get("enabled", True),
                "meta": payload.get("meta") or {},
            }
            stmt = (
                pg_insert(RegistryEntry)
                .values(values)
                .on_conflict_do_update(
                    index_elements=["kind", "key"],
                    set_={
                        "label": values["label"],
                        "description": values["description"],
                        "icon": values["icon"],
                        "href": values["href"],
                        "action": values["action"],
                        "group_name": values["group_name"],
                        "sort_order": values["sort_order"],
                        "enabled": values["enabled"],
                        "meta": values["meta"],
                    },
                )
            )
            await session.execute(stmt)
            await session.commit()
        result = await self.get_entry(values["kind"], values["key"])
        assert result is not None
        return result


def _serialize(entry: RegistryEntry | None) -> dict[str, Any]:
    """Convert a RegistryEntry model instance into a JSON-serializable dict."""
    if entry is None:
        return {}
    return {
        "id": str(entry.id),
        "kind": entry.kind,
        "key": entry.key,
        "label": entry.label,
        "description": entry.description,
        "icon": entry.icon,
        "href": entry.href,
        "action": entry.action,
        "group_name": entry.group_name,
        "sort_order": entry.sort_order,
        "enabled": entry.enabled,
        "meta": entry.meta or {},
        "created_at": entry.created_at.isoformat() if entry.created_at else None,
        "updated_at": entry.updated_at.isoformat() if entry.updated_at else None,
    }
