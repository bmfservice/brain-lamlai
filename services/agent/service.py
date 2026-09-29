"""Braind Agent Service.

This is the new "brain" of Braind — an agent that can use tools, reason step-by-step,
retrieve knowledge, search the web, ingest new data, and write content.

It replaces the old simple ChatService with a Dify-style agent runtime.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.agent.loop import AgentLoop
from core.agent.tool import ToolRegistry, get_registry
from core.config import settings
from core.database import Conversation, Message
from core.logging import get_logger
from core.vllm import VLLMClient
from services.rag.pipeline import UnifiedRetrievalService

logger = get_logger(__name__)

# How many prior turns (user+assistant pairs ≈ 2×) to feed the agent loop.
_HISTORY_LIMIT = 12

_WEEKDAYS_VI = (
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
)


class AgentService:
    """High-level agent that powers intelligent conversations."""

    def __init__(
        self,
        session_factory,
        vllm: VLLMClient,
        retrieval: UnifiedRetrievalService,
        registry: ToolRegistry | None = None,
        ingest=None,
    ) -> None:
        self.session_factory = session_factory
        self.vllm = vllm
        self.retrieval = retrieval
        self.ingest = ingest
        self.registry = registry or get_registry()
        self.loop = AgentLoop(vllm, self.registry, max_steps=6)
        # Strong refs so fire-and-forget learning tasks are not garbage collected.
        self._learn_tasks: set[asyncio.Task] = set()

    async def chat(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        system: str | None = None,
        session_id: str | None = None,
        fingerprint_hash: str | None = None,
    ) -> dict[str, Any]:
        """Run the agent loop on a user message and persist the turn.

        Loads the last ``_HISTORY_LIMIT`` messages when ``conversation_id`` is
        known, pre-retrieves knowledge so brains is always consulted (scoped to
        the visitor's session plus the global corpus), runs the tool loop,
        collects citations from tool observations, then inserts user + assistant
        ``Message`` rows under a real ``Conversation`` UUID owned by the session.
        """
        history, memory = await asyncio.gather(
            self._load_history(conversation_id, session_id=session_id),
            self._pre_retrieve(message, session_id=session_id),
        )

        result = await self.loop.run(
            message,
            system=system or self._default_system(memory=memory),
            conversation_history=history,
        )

        tool_calls = result.get("tool_calls", [])
        if isinstance(tool_calls, int):
            tool_calls = []

        steps = result.get("steps") or []
        answer = result.get("answer") or ""
        citations = self._collect_citations(steps)
        if memory:
            citations = self._merge_memory_citations(citations, memory)

        conv_id = await self._persist_turn(
            conversation_id=conversation_id,
            user_message=message,
            assistant_message=answer,
            citations=citations,
            steps=steps,
            session_id=session_id,
            fingerprint_hash=fingerprint_hash,
        )

        self._schedule_learning(
            conversation_id=conv_id,
            user_message=message,
            assistant_message=answer,
            citations=citations,
            session_id=session_id,
        )

        return {
            "conversation_id": conv_id,
            "content": answer,
            "steps": steps,
            "tool_calls": tool_calls,
            "citations": citations,
        }

    async def list_conversations(
        self, *, limit: int = 30, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return recent conversations with a short last-message preview.

        Giai đoạn 4: scoped to the caller's session plus system conversations.
        """
        limit = max(1, min(limit, 100))
        async with self.session_factory() as session:
            stmt = select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)
            if session_id:
                stmt = stmt.where(
                    or_(Conversation.session_id.is_(None), Conversation.session_id == session_id)
                )
            rows = (
                (
                    await session.execute(stmt)
                )
                .scalars()
                .all()
            )
            out: list[dict[str, Any]] = []
            for conv in rows:
                last = (
                    await session.execute(
                        select(Message)
                        .where(Message.conversation_id == conv.id)
                        .order_by(Message.created_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                preview = (last.content if last else "")[:120]
                out.append(
                    {
                        "id": str(conv.id),
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat() if conv.created_at else None,
                        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                        "preview": preview,
                    }
                )
            return out

    async def get_messages(
        self, conversation_id: str, *, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return messages for one conversation, oldest first (session-scoped)."""
        try:
            cid = uuid.UUID(conversation_id)
        except (ValueError, TypeError):
            return []
        async with self.session_factory() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == cid)
                .order_by(Message.created_at.asc())
            )
            if session_id:
                stmt = stmt.where(
                    or_(Message.session_id.is_(None), Message.session_id == session_id)
                )
            rows = (
                (
                    await session.execute(stmt)
                )
                .scalars()
                .all()
            )
            return [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "citations": m.citations or [],
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in rows
            ]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    async def _load_history(
        self, conversation_id: str | None, *, session_id: str | None = None
    ) -> list[dict[str, str]]:
        """Load the last N user/assistant messages for context resume."""
        if not conversation_id or conversation_id in ("new", "null", "undefined"):
            return []
        try:
            cid = uuid.UUID(conversation_id)
        except (ValueError, TypeError):
            return []
        async with self.session_factory() as session:
            stmt = (
                select(Message)
                .where(
                    Message.conversation_id == cid,
                    Message.role.in_(("user", "assistant")),
                )
                .order_by(Message.created_at.desc())
                .limit(_HISTORY_LIMIT)
            )
            if session_id:
                stmt = stmt.where(
                    or_(Message.session_id.is_(None), Message.session_id == session_id)
                )
            rows = (
                (await session.execute(stmt)).scalars().all()
            )
        # Reverse to chronological order for the LLM.
        history = [{"role": m.role, "content": m.content} for m in reversed(rows)]
        return history

    async def _persist_turn(
        self,
        *,
        conversation_id: str | None,
        user_message: str,
        assistant_message: str,
        citations: list[dict[str, Any]],
        steps: list[dict[str, Any]],
        session_id: str | None = None,
        fingerprint_hash: str | None = None,
    ) -> str:
        """Create/load conversation and insert the user + assistant messages.

        Giai đoạn 4: the conversation and both messages carry the visitor's
        session id so history/list stay session-scoped. Loading an existing
        conversation belonging to another session is refused (fresh chat).
        """
        async with self.session_factory() as session:
            conv = await self._get_or_create_conversation(
                session, conversation_id, user_message, session_id=session_id
            )
            # Flush so a brand-new conversation has its UUID before the
            # messages reference it.
            await session.flush()
            try:
                _sid = uuid.UUID(session_id) if session_id else None
            except (ValueError, AttributeError):
                _sid = None
            session.add(
                Message(
                    conversation_id=conv.id,
                    role="user",
                    content=user_message,
                    session_id=_sid,
                )
            )
            session.add(
                Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=assistant_message,
                    citations=citations,
                    session_id=_sid,
                    meta={"steps_count": len(steps)},
                )
            )
            conv.updated_at = datetime.now(UTC)
            await session.commit()
            return str(conv.id)

    async def _get_or_create_conversation(
        self,
        session: AsyncSession,
        conversation_id: str | None,
        title: str,
        *,
        session_id: str | None = None,
    ) -> Conversation:
        if conversation_id and conversation_id not in ("new", "null", "undefined"):
            try:
                cid = uuid.UUID(conversation_id)
            except (ValueError, TypeError):
                cid = None
            if cid is not None:
                conv = (
                    await session.execute(select(Conversation).where(Conversation.id == cid))
                ).scalar_one_or_none()
                if conv is not None:
                    # Giai đoạn 4: a conversation owned by another session is
                    # not readable here — start a fresh one instead of leaking.
                    if (
                        session_id
                        and conv.session_id is not None
                        and str(conv.session_id) != session_id
                    ):
                        return self._new_conversation(session, title, session_id)
                    return conv

        return self._new_conversation(session, title, session_id)

    @staticmethod
    def _new_conversation(
        session: AsyncSession, title: str, session_id: str | None
    ) -> Conversation:
        try:
            _sid = uuid.UUID(session_id) if session_id else None
        except (ValueError, AttributeError):
            _sid = None
        conv = Conversation(
            title=(title or "New Chat").strip()[:80] or "New Chat",
            session_id=_sid,
        )
        session.add(conv)
        return conv

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_citations(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pull title+url citations from web_search / knowledge / fetch observations."""
        seen: set[str] = set()
        citations: list[dict[str, Any]] = []

        def _add(title: str, url: str, source: str = "") -> None:
            url = (url or "").strip()
            if not url or not url.startswith(("http://", "https://")):
                return
            key = url.rstrip("/").lower()
            if key in seen:
                return
            seen.add(key)
            short = (title or url).strip()[:80] or url
            item: dict[str, Any] = {"title": short, "url": url}
            if source:
                item["source"] = source
            citations.append(item)

        for step in steps:
            tool = (step.get("tool") or "").strip()
            obs = step.get("observation")
            if not isinstance(obs, dict):
                continue

            if tool == "web_search":
                for r in obs.get("results") or []:
                    if isinstance(r, dict):
                        _add(r.get("title") or "", r.get("url") or "", "web")
            elif tool in ("fetch_url", "fetch_rss"):
                # Single-page or feed fetch.
                if obs.get("url"):
                    _add(obs.get("title") or "", obs.get("url") or "", tool)
                for r in obs.get("items") or obs.get("results") or []:
                    if isinstance(r, dict):
                        _add(r.get("title") or "", r.get("url") or r.get("link") or "", tool)
            elif tool == "knowledge_retrieve":
                for r in obs.get("items") or []:
                    if not isinstance(r, dict):
                        continue
                    # Prefer explicit url; fall back to source path if it looks like a URL.
                    url = (r.get("url") or "").strip()
                    if not url:
                        continue
                    _add(r.get("title") or "", url, "memory")

        return citations

    # ------------------------------------------------------------------
    # Knowledge grounding
    # ------------------------------------------------------------------

    async def _pre_retrieve(
        self, message: str, *, session_id: str | None = None
    ) -> list[Any]:
        """Retrieve knowledge before the agent loop so brains is always consulted.

        The agent still owns ``knowledge_retrieve`` for follow-up lookups; this
        pass only guarantees a first look at the knowledge base (scoped to the
        visitor's session plus the global corpus) even when the model would
        have answered from its own weights.
        """
        if not settings.chat_always_retrieve:
            return []
        try:
            chunks = await self.retrieval.retrieve(
                message,
                top_k=settings.chat_pre_retrieve_top_k,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("agent.pre_retrieve_failed", error=str(exc))
            return []
        kept = [c for c in chunks if c.score >= settings.chat_pre_retrieve_min_score]
        logger.info("agent.pre_retrieve", retrieved=len(chunks), kept=len(kept))
        return kept

    @staticmethod
    def _format_memory(chunks: list[Any]) -> str:
        """Render retrieved chunks as a compact, quotable memory block."""
        lines: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            excerpt = (getattr(chunk, "excerpt", "") or chunk.content or "")[:400].strip()
            if not excerpt:
                continue
            lines.append(f"[{index}] {chunk.title} (score {chunk.score:.2f})\n{excerpt}")
        return "\n\n".join(lines)

    @staticmethod
    def _merge_memory_citations(
        citations: list[dict[str, Any]], memory: list[Any]
    ) -> list[dict[str, Any]]:
        """Append citations for pre-retrieved chunks the tool trace did not cover.

        Web-sourced chunks carry a URL and become URL citations. Knowledge-base
        chunks with no URL (vault notes, pasted text, prior chat turns) become
        title citations marked ``source: "memory"`` so the answer still shows
        which stored knowledge grounded it instead of citing nothing.
        """
        seen = {(c.get("url") or "").rstrip("/").lower() for c in citations}
        seen_titles = {
            (c.get("title") or "").strip().lower()
            for c in citations
            if not c.get("url")
        }
        for chunk in memory:
            meta = chunk.meta if isinstance(chunk.meta, dict) else {}
            url = (meta.get("url") or meta.get("source_url") or "").strip()
            title = (getattr(chunk, "title", "") or "").strip()
            if url.startswith(("http://", "https://")):
                key = url.rstrip("/").lower()
                if key in seen:
                    continue
                seen.add(key)
                citations.append({"title": title[:80] or url, "url": url, "source": "memory"})
                continue
            # No URL — cite the stored knowledge by title instead of skipping it.
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            citations.append({"title": title[:80], "source": "memory"})
        return citations

    def _schedule_learning(
        self,
        *,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        citations: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> None:
        """Run chat-learning in the background so the reply is not delayed."""
        if self.ingest is None or not settings.chat_learn_enabled:
            return
        task = asyncio.create_task(
            self._learn_from_turn(
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_message=assistant_message,
                citations=citations,
                session_id=session_id,
            )
        )
        self._learn_tasks.add(task)
        task.add_done_callback(self._learn_tasks.discard)

    async def _learn_from_turn(
        self,
        *,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        citations: list[dict[str, Any]],
        session_id: str | None = None,
    ) -> None:
        """Write a substantive chat turn back into the knowledge base.

        Best-effort: an ingest failure must never break the chat response. Turns
        are stored as ``source_type="chat"`` so they stay distinguishable from
        vault and web knowledge.
        """
        if self.ingest is None or not settings.chat_learn_enabled:
            return
        answer = (assistant_message or "").strip()
        if len(answer) < settings.chat_learn_min_answer_chars:
            return
        try:
            await self.ingest.ingest_text(
                f"## {user_message.strip()}\n\n{answer}",
                title=f"Chat: {user_message.strip()[:70]}",
                source_type="chat",
                tags=["chat"],
                meta={
                    "conversation_id": conversation_id,
                    "question": user_message.strip()[:500],
                    "citations": [c.get("url") for c in citations if c.get("url")][:10],
                },
            )
        except Exception as exc:
            logger.warning("agent.chat_learn_failed", error=str(exc))

    def _default_system(self, *, memory: list[Any] | None = None) -> str:
        """Build the agent system prompt, grounded in current time and memory."""
        parts = [
            "Bạn là Braind — Trí tuệ thứ hai của người dùng. "
            "Bạn là một AI agent tự động, có khả năng TỰ ĐỘNG GỌI và THỰC THI các công cụ (tools). "
            "\n\n🔴 QUY TẮC BẮT BUỘC VỀ TOOLS:\n"
            "- KHÔNG BAO GIỜ hiển thị code, không show function calls cho user\n"
            "- KHÔNG BAO GIỜ giải thích \"tôi sẽ gọi tool X\", chỉ GỌI NGAY\n"
            "- KHÔNG BAO GIỜ hỏi \"bạn có muốn tôi...\", chỉ LÀM LUÔN\n"
            "- Khi user yêu cầu tìm kiếm, tra cứu, kiểm tra → GỌI TOOL NGAY LẬP TỨC\n"
            "- Sau khi tool trả kết quả → Tổng hợp và trả lời user bằng tiếng Việt tự nhiên\n"
            "\n📚 Các công cụ available:\n"
            "- knowledge_retrieve: Truy xuất kiến thức từ bộ nhớ Braind\n"
            "- web_search: Tìm kiếm Google/Bing (~10 kết quả), set save_to_brain=true để học\n"
            "- fetch_url: Đọc nội dung từ 1 URL cụ thể\n"
            "- fetch_rss: Đọc RSS feed (tin tức, blog)\n"
            "\n💡 Workflow đúng:\n"
            "User: \"Tìm kiếm thông tin về X\"\n"
            "→ Bạn: [TỰ ĐỘNG gọi web_search(query='X', max_results=10)] (SILENT)\n"
            "→ Tool trả kết quả\n"
            "→ Bạn: \"Tôi tìm thấy ... [tổng hợp thông tin]\" (USER FACING)\n"
            "\n⚠️ Workflow SAI (KHÔNG BAO GIỜ làm):\n"
            "User: \"Tìm kiếm thông tin về X\"\n"
            "→ Bạn: \"Tôi sẽ gọi web_search...\" ❌\n"
            "→ Bạn: ```python\nweb_search(...)``` ❌\n"
            "→ Bạn: \"Bạn có muốn tôi tìm kiếm không?\" ❌\n"
            "\nTrích dẫn nguồn (title + URL) khi dùng thông tin bên ngoài. "
            "Chỉ dùng đoạn kiến thức liên quan, không nhồi nguyên văn dài. "
            "Trả lời bằng tiếng Việt tự nhiên, dùng Markdown (in đậm, danh sách) "
            "khi giúp đọc dễ hơn."
        ]
        if settings.chat_inject_datetime:
            parts.append(self._now_line())
        if memory:
            block = self._format_memory(memory)
            if block:
                parts.append(
                    "Kiến thức liên quan đã truy xuất sẵn từ bộ nhớ Braind "
                    "(ưu tiên dùng nếu đủ trả lời; gọi knowledge_retrieve nếu cần thêm):\n"
                    f"{block}"
                )
        return "\n\n".join(parts)

    @staticmethod
    def _now_line() -> str:
        """Return the current local date/time as a prompt line."""
        now = datetime.now(ZoneInfo(settings.scheduler_timezone))
        weekday = _WEEKDAYS_VI[now.weekday()]
        return (
            f"Thời điểm hiện tại: {weekday}, ngày {now.day:02d}/{now.month:02d}/{now.year}, "
            f"{now.strftime('%H:%M')} ({settings.scheduler_timezone}). "
            "Dùng mốc thời gian này khi người dùng hỏi về ngày/giờ hoặc thông tin mới nhất."
        )
