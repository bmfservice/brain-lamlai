"""Braind API - chat routes (agentic, Dify-style thought→tool→answer trace)."""

from __future__ import annotations

import html
import json
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response
from markdown_it import MarkdownIt
from sse_starlette.sse import EventSourceResponse

from api.deps import get_agent, get_session_context, get_unified_retrieval
from api.errors import safe_error_payload
from api.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from core.logging import get_logger
from core.session_context import SESSION_COOKIE
from services.agent.service import AgentService
from services.rag.pipeline import UnifiedRetrievalService

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])
_md = (
    MarkdownIt("commonmark", {"html": False, "typographer": True})
    .enable("table")
    .enable("strikethrough")
)


def _render_markdown(text: str) -> str:
    """Convert Markdown to safe HTML. Falls back to escaped plain text."""
    if not text:
        return ""
    try:
        rendered = _md.render(text)
        return re.sub(r"<a\s+href=", '<a target="_blank" rel="noopener noreferrer" href=', rendered)
    except Exception as exc:
        logger.warning("chat.markdown_render_failed", error=str(exc))
        return f"<p>{html.escape(text)}</p>"


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    agent_service: Annotated[AgentService, Depends(get_agent)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
    response: Response,
) -> ChatResponse:
    """Send a chat message — the agent reasons, calls tools, then answers."""
    if session_ctx.minted:
        response.set_cookie(SESSION_COOKIE, session_ctx.session_id, samesite="lax", httponly=True)
    result = await agent_service.chat(
        req.message,
        conversation_id=req.conversation_id,
        session_id=session_ctx.session_id,
        fingerprint_hash=session_ctx.fingerprint_hash,
    )
    content = result.get("content") or ""
    return ChatResponse(
        conversation_id=result.get("conversation_id") or "new",
        content=content,
        content_html=_render_markdown(content),
        citations=result.get("citations") or [],
        steps=result.get("steps") or [],
        tool_calls=result.get("tool_calls") or [],
    )


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest,
    agent_service: Annotated[AgentService, Depends(get_agent)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
) -> EventSourceResponse:
    """Send a chat message — agent trace then answer, streamed as SSE events."""

    async def event_source():
        try:
            result = await agent_service.chat(
                req.message,
                conversation_id=req.conversation_id,
                session_id=session_ctx.session_id,
                fingerprint_hash=session_ctx.fingerprint_hash,
            )
            content = result.get("content") or ""
            for step in result.get("steps") or []:
                yield {
                    "event": "step",
                    "data": json.dumps(step, ensure_ascii=False, default=str),
                }
            yield {
                "event": "token",
                "data": json.dumps(
                    {"content": content, "content_html": _render_markdown(content)},
                    ensure_ascii=False,
                    default=str,
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "conversation_id": result.get("conversation_id") or "new",
                        "steps": result.get("steps") or [],
                        "citations": result.get("citations") or [],
                        "content_html": _render_markdown(content),
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            }
        except Exception as exc:
            logger.error("api.chat_stream_error", error=str(exc), exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(safe_error_payload(exc)["error"], ensure_ascii=False),
            }

    return EventSourceResponse(event_source())


@router.get("/conversations")
async def list_conversations(
    agent_service: Annotated[AgentService, Depends(get_agent)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
    limit: int = Query(30, ge=1, le=100),
) -> dict:
    """List recent chat conversations (title + last-message preview)."""
    return {
        "conversations": await agent_service.list_conversations(
            limit=limit, session_id=session_ctx.session_id
        )
    }


@router.get("/conversations/{conversation_id}/messages")
async def conversation_messages(
    conversation_id: str,
    agent_service: Annotated[AgentService, Depends(get_agent)],
    session_ctx: Annotated[Any, Depends(get_session_context)],
) -> dict:
    """Return all messages for one conversation (oldest first)."""
    return {
        "conversation_id": conversation_id,
        "messages": await agent_service.get_messages(
            conversation_id, session_id=session_ctx.session_id
        ),
    }


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    retrieval: Annotated[UnifiedRetrievalService, Depends(get_unified_retrieval)],
) -> SearchResponse:
    """Hybrid retrieval search over the knowledge base."""
    chunks = await retrieval.retrieve(req.query, top_k=req.top_k, tags=req.tags or None)
    return SearchResponse(results=[chunk.to_dict() for chunk in chunks])
