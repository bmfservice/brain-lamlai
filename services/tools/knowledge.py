"""Braind Tool - Knowledge retrieval (RAG memory lookup).

The brain's core ability: given a query, retrieve the most relevant
chunks stored in the knowledge base (hybrid: vector + full-text + trgm).

Only focused excerpts are shipped to the agent (not full 800-char chunks)
so answers stay precise and non-robotic.
"""

from __future__ import annotations

from typing import Any

from core.agent.tool import BaseTool
from core.logging import get_logger
from services.rag.pipeline import UnifiedRetrievalService

logger = get_logger(__name__)


class KnowledgeRetrieveTool(BaseTool):
    """Retrieve relevant knowledge from the Braind memory bank."""

    name = "knowledge_retrieve"
    description = (
        "Truy xuất kiến thức liên quan từ bộ nhớ của Braind (knowledge base). "
        "Dùng khi cần trả lời dựa trên những gì Braind đã học. "
        "Chỉ trả về đoạn trích ngắn (~280 ký tự) khớp nhất với câu hỏi."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Câu hỏi / chủ đề cần truy xuất kiến thức",
            },
            "top_k": {
                "type": "integer",
                "description": "Số lượng đoạn kết quả (mặc định 5, tối đa 8)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, retrieval: UnifiedRetrievalService) -> None:
        self.retrieval = retrieval

    async def execute(self, query: str, top_k: int = 5) -> Any:
        chunks = await self.retrieval.retrieve(query, top_k=min(int(top_k or 5), 8))
        if not chunks:
            return {
                "found": 0,
                "message": "Không tìm thấy kiến thức liên quan trong bộ nhớ.",
                "items": [],
            }
        items: list[dict[str, Any]] = []
        for c in chunks:
            # Prefer focused excerpt; fall back to a short head of content.
            excerpt = (getattr(c, "excerpt", None) or "").strip()
            if not excerpt:
                excerpt = (c.content or "")[:280]
            item: dict[str, Any] = {
                "title": c.title,
                "excerpt": excerpt,
                "score": round(c.score, 3),
                "tags": c.tags or [],
            }
            # Surface URL when the source recorded one (for citations).
            url = ""
            if isinstance(c.meta, dict):
                url = (c.meta.get("url") or c.meta.get("source_url") or "").strip()
            if url:
                item["url"] = url
            items.append(item)
        return {
            "found": len(items),
            "items": items,
        }
