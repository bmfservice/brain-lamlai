"""Braind Writer Service.

Generates and edits long-form content (blog, marketing, social) powered by
RAG retrieval + the local LLM. Implements the user's core vision: the AI is a
composition layer that pulls from the brain's knowledge and produces requested
files — never a heavy general-purpose writer.

Content kinds supported:
- blog, social, email, ad, seo, summary, newsletter, case_study,
  product_desc, press_release, thread, landing_page

Each request renders 1..N sequential variants (no parallel GPU load), runs an
optional external web duplicate check, and (optionally) ingests the generated
content + duplicate findings back into the brain so it learns style and avoids
copying existing web content.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from core.chunking import count_tokens
from core.config import settings
from core.database import GeneratedContent
from core.logging import get_logger
from core.vllm import VLLMClient
from services.rag.pipeline import UnifiedRetrievalService
from services.rag.preferences import for_writing as writer_preferences
from services.rag.service import RetrievedChunk
from services.writer.duplicate import ExternalDuplicateChecker

logger = get_logger(__name__)

# Number of tokens of content we will accept from the LLM (max output)
MAX_OUTPUT_TOKENS = 4000
# Grounding context budget. Chat model window is 16384; this leaves room for
# the instructions, the selectors and the full output allowance.
MAX_CONTEXT_TOKENS = 9000

# --- Content type definitions -------------------------------------------------

CONTENT_TYPES = {
    "blog": {
        "label": "Bài viết blog",
        "instructions": (
            "Viết bài blog dài dạng markdown có tiêu đề H1, các mục H2/H3 rõ ràng, "
            "statement mở đầu hấp dẫn, các luận điểm có dẫn chứng từ tài liệu, "
            "phần kết luận và CTA. Ngôn ngữ: tiếng Việt (trừ khi người dùng yêu cầu khác)."
        ),
        "extra_fields": [
            "keywords",
            "tone",
            "audience",
            "length",
            "cta",
            "lang",
            "seo_title",
            "outline",
        ],
    },
    "social": {
        "label": "Bài đăng mạng xã hội",
        "instructions": (
            "Viết bài đăng ngắn gọn, hấp dẫn cho mạng xã hội (LinkedIn/Facebook/X). "
            "Dùng hashtag, emoji, giọng điệu gần gũi, câu mở đầu bắt tai. "
            "Giữ dưới 280 ký tự nếu cần."
        ),
        "extra_fields": ["keywords", "tone", "audience", "platform", "hashtags", "length"],
    },
    "email": {
        "label": "Email marketing / newsletter",
        "instructions": (
            "Viết email marketing/newsletter với subject line hấp dẫn, mở đầu cá nhân hóa, "
            "nội dung ngắn gọn, CTA rõ ràng. Ngôn ngữ thân thiện, chuyên nghiệp."
        ),
        "extra_fields": ["keywords", "tone", "audience", "subject", "cta", "length"],
    },
    "ad": {
        "label": "Quảng cáo (ad copy)",
        "instructions": (
            "Viết quảng cáo ngắn (Google/Facebook Ads) với headline, mô tả, CTA. "
            "Ngắn gọn, tập trung vào lợi ích, từ khóa mạnh."
        ),
        "extra_fields": ["keywords", "tone", "audience", "platform", "variants", "cta"],
    },
    "seo": {
        "label": "Nội dung SEO",
        "instructions": (
            "Viết bài SEO tối ưu: outline, meta description, từ khóa chính/phụ, "
            "các tiêu đề H1/H2/H3, nội dung có dẫn chứng, CTA. Ngôn ngữ tự nhiên."
        ),
        "extra_fields": ["keywords", "tone", "audience", "primary_kw", "secondary_kw", "outline"],
    },
    "summary": {
        "label": "Tóm tắt nội dung",
        "instructions": (
            "Tóm tắt ngắn gọn, súc tích từ tài liệu đã thu thập. "
            "Giữ các ý chính, số liệu quan trọng, không thêm nội dung mới."
        ),
        "extra_fields": ["keywords", "tone", "audience", "length", "focus"],
    },
    "newsletter": {
        "label": "Bản tin định kỳ",
        "instructions": (
            "Viết bản tin định kỳ (newsletter) với tiêu đề hấp dẫn, các mục tin tức ngắn, "
            "CTA đăng ký/đọc thêm. Giọng điệu gần gũi, chuyên nghiệp."
        ),
        "extra_fields": ["keywords", "tone", "audience", "subject", "cta", "length"],
    },
    "case_study": {
        "label": "Case study / Nghiên cứu điển hình",
        "instructions": (
            "Viết case study dạng markdown: bối cảnh, thách thức, giải pháp, kết quả, "
            "số liệu cụ thể, trích dẫn người dùng, CTA. Ngôn ngữ thuyết phục, chuyên nghiệp."
        ),
        "extra_fields": ["keywords", "tone", "audience", "cta", "outline"],
    },
    "product_desc": {
        "label": "Mô tả sản phẩm",
        "instructions": (
            "Viết mô tả sản phẩm ngắn gọn, hấp dẫn: đặc điểm nổi bật, lợi ích, "
            "CTA mua hàng. Ngôn ngữ thuyết phục, tập trung vào giá trị."
        ),
        "extra_fields": ["keywords", "tone", "audience", "platform", "cta", "length"],
    },
    "press_release": {
        "label": "Thông cáo báo chí",
        "instructions": (
            "Viết thông cáo báo chí theo chuẩn: tiêu đề, phụ đề, ngày tháng, "
            "lead paragraph, chi tiết, trích dẫn, boilerplate, contact. Ngôn ngữ trang trọng."
        ),
        "extra_fields": ["keywords", "tone", "audience", "cta", "outline"],
    },
    "thread": {
        "label": "Thread / Chuỗi tweet",
        "instructions": (
            "Viết chuỗi tweet (thread) 5–10 tweet ngắn, số thứ tự, hook đầu tiên mạnh, "
            "CTA ở tweet cuối. Giọng gần gũi, dễ chia sẻ."
        ),
        "extra_fields": ["keywords", "tone", "audience", "platform", "hashtags", "length"],
    },
    "landing_page": {
        "label": "Landing page / Trang đích",
        "instructions": (
            "Viết landing page dạng markdown: headline mạnh, subheadline, các section "
            "giải quyết nỗi đau, CTA rõ ràng, form capture. Ngôn ngữ thuyết phục, ngắn gọn."
        ),
        "extra_fields": ["keywords", "tone", "audience", "cta", "outline"],
    },
}

VALID_TYPES = list(CONTENT_TYPES.keys())

# --- Controlled option lists (UI selects) -------------------------------------

TONE_OPTIONS = [
    "chuyên nghiệp",
    "thân thiện",
    "trang trọng",
    "hài hước",
    "thuyết phục",
    "truyền cảm hứng",
    "bình dân",
]

AUDIENCE_OPTIONS = [
    "người mới",
    "chuyên gia",
    "lãnh đạo",
    "sinh viên",
    "khách hàng B2B",
    "khách hàng B2C",
]

WORD_COUNT_OPTIONS = [200, 300, 600, 900, 1200, 1500, 2000]

FORMAT_OPTIONS = ["markdown", "plain", "html"]

LANGUAGE_OPTIONS = [
    ("vi", "Tiếng Việt"),
    ("en", "Tiếng Anh"),
    ("vi-en", "Song ngữ Việt - Anh"),
]

LANGUAGE_INSTRUCTIONS = {
    "vi": "Viết toàn bộ bằng tiếng Việt.",
    "en": "Write the entire content in English.",
    "vi-en": "Viết song ngữ Việt - Anh: mỗi phần chính viết tiếng Việt và kèm bản tiếng Anh.",
}


def writer_options() -> dict:
    """Return the controlled option lists for the frontend selects."""
    return {
        "content_types": [{"value": k, "label": v["label"]} for k, v in CONTENT_TYPES.items()],
        "tones": TONE_OPTIONS,
        "audiences": AUDIENCE_OPTIONS,
        "word_counts": WORD_COUNT_OPTIONS,
        "formats": FORMAT_OPTIONS,
        "languages": [{"value": code, "label": label} for code, label in LANGUAGE_OPTIONS],
        "max_article_count": settings.writer_max_variants,
    }


def _selectors_to_text(req: WriterRequest) -> str:
    """Render the normalized selectors for the prompt."""
    lang_inst = LANGUAGE_INSTRUCTIONS.get(req.language, LANGUAGE_INSTRUCTIONS["vi"])
    format_inst = {
        "markdown": "Xuất ra Markdown (heading #, ##, list, in đậm...)",
        "plain": "Xuất ra văn bản thuần (không dùng cú pháp Markdown)",
        "html": "Xuất ra HTML hoàn chỉnh (dùng thẻ <h1>, <h2>, <p>, <ul>...)",
    }.get(req.output_format, "Markdown")
    return (
        f"- Độ dài mục tiêu: {req.word_count} từ (±10%)\n"
        f"- Ngôn ngữ: {lang_inst}\n"
        f"- Format xuất: {format_inst}"
    )


@dataclass
class WriterRequest:
    """Request to generate content (batch of 1..N variants)."""

    content_type: str
    topic: str
    keywords: list[str] = field(default_factory=list)
    tone: str = "chuyên nghiệp"
    audience: str = ""
    word_count: int = 600
    output_format: str = "plain"
    article_count: int = 1
    language: str = "vi"
    check_web_duplicates: bool = False
    ingest_to_brain: bool = False
    extra: dict = field(default_factory=dict)

    def context_fields(self, all_fields: set[str]) -> dict:
        """Return config fields relevant to this content type."""
        result = {}
        for f in all_fields:
            if f in self.extra:
                result[f] = self.extra[f]
        if "keywords" in all_fields and self.keywords:
            result.setdefault("keywords", ", ".join(self.keywords))
        if "tone" in all_fields and self.tone:
            result.setdefault("tone", self.tone)
        if "audience" in all_fields and self.audience:
            result.setdefault("audience", self.audience)
        return result


class WriterService:
    """Generate and refine marketing/content from the knowledge base."""

    def __init__(
        self,
        session_factory,
        vllm: VLLMClient,
        retrieval: UnifiedRetrievalService,
        ingest=None,
        web_tool=None,
        default_model: str | None = None,
    ):
        self.session_factory = session_factory
        self.vllm = vllm
        self.retrieval = retrieval
        self.ingest = ingest
        self.web_tool = web_tool
        self.model = default_model or settings.llm_model
        self.dupe_checker = (
            ExternalDuplicateChecker(web_tool=self.web_tool, vllm=vllm) if web_tool else None
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    async def generate(self, req: WriterRequest) -> dict:
        """Generate content from a title hint through a grounded Writer flow."""
        self._validate_request(req)
        count = max(1, min(req.article_count, settings.writer_max_variants))
        batch_id = str(uuid.uuid4())
        title_hint = req.topic.strip()
        # Ground generation in material worth stating as fact: specifications
        # and data outrank opinion, and marketing copy is demoted hardest —
        # writing new marketing from old marketing makes the brain repeat itself.
        chunks = await self.retrieval.retrieve(
            self._build_search_query(req),
            tags=req.keywords or None,
            preferences=writer_preferences(req.keywords),
        )

        items: list[dict] = []
        warnings: list[str] = []
        for index in range(1, count + 1):
            try:
                items.append(
                    await self._generate_one(
                        req,
                        title_hint,
                        chunks,
                        batch_id,
                        variant_index=index,
                    )
                )
            except Exception:
                logger.error("writer.variant_failed", index=index, exc_info=True)
                warnings.append(f"Biến thể {index} không thể hoàn tất.")
        logger.info(
            "writer.batch_complete",
            batch_id=batch_id,
            requested=count,
            completed=len(items),
        )
        return {
            "batch_id": batch_id,
            "requested_count": count,
            "completed_count": len(items),
            "items": items,
            "warnings": warnings,
        }

    async def check_external(self, content_id: str) -> dict:
        """Re-run the external duplicate check and learn fetched reference pages.

        Called by the post-creation "Kiểm tra trùng web" flow. Unlike the
        generate path, this always ingests the fetched external pages into the
        brain (the user explicitly opted in by clicking the button), regardless
        of the request's ``ingest_to_brain`` flag stored on the article.
        """
        if self.dupe_checker is None:
            raise RuntimeError("External duplicate checker has not been initialized")

        # Read the article first and close the session: the duplicate check fetches
        # external pages over the network and the reference ingest embeds them, so
        # neither should run inside an open transaction.
        async with self.session_factory() as session:
            generated = await session.get(GeneratedContent, content_id)
            if generated is None:
                raise FileNotFoundError(f"Không tìm thấy nội dung có id: {content_id}")
            generation_id = generated.id
            generated_title = generated.title
            article = generated.content
            content_type = generated.content_type
            tags = list(generated.tags or [])
            metadata = dict(generated.meta or {})

        title_hint = str(metadata.get("title_hint") or generated_title)
        audit = await self.dupe_checker.check(generated_title, article)

        # Learn the fetched external pages into the brain on explicit request.
        # Reconstruct a minimal WriterRequest from the stored metadata so the
        # reference ingest keeps its provenance + tags wiring intact. Tone and
        # audience are read defensively in case an older article predates those
        # metadata fields; _ingest_reference_pages ignores them for provenance,
        # but WriterRequest's own validator is not run on this reconstructed req.
        stored_tone = str(metadata.get("tone") or "")
        stored_audience = str(metadata.get("audience") or "")
        req = WriterRequest(
            content_type=content_type,
            topic=title_hint,
            keywords=tags,
            tone=stored_tone if stored_tone in TONE_OPTIONS else "chuyên nghiệp",
            audience=stored_audience if stored_audience in AUDIENCE_OPTIONS else "người mới",
            word_count=int(metadata.get("word_count") or 600),
            output_format=str(metadata.get("output_format") or "plain"),
            language=str(metadata.get("language") or "vi"),
            check_web_duplicates=True,
            ingest_to_brain=False,
        )
        learned = await self._ingest_reference_pages(
            req,
            generated_title=generated_title,
            title_hint=title_hint,
            generation_id=generation_id,
            batch_id=str(metadata.get("batch_id") or ""),
            web_check=audit,
            ingest_enabled=True,
        )

        # Re-read the row before persisting so a concurrent writer's metadata edits
        # are not clobbered by the snapshot taken above.
        async with self.session_factory() as session:
            db_gen = await session.get(GeneratedContent, content_id)
            if db_gen is not None:
                fresh = dict(db_gen.meta or {})
                fresh["title_hint"] = title_hint
                fresh["generated_title"] = generated_title
                fresh["web_duplicate_check"] = audit
                fresh["ingest_reference_page_count"] = (
                    int(fresh.get("ingest_reference_page_count") or 0) + learned
                )
                fresh["ingest_status"] = (
                    "ingested" if learned > 0 else (fresh.get("ingest_status") or "skipped")
                )
                db_gen.meta = fresh
                await session.commit()
        logger.info(
            "writer.check_external_done",
            content_id=str(content_id),
            web=audit.get("status", "none"),
            references_learned=learned,
        )
        return {
            "content_id": str(content_id),
            "web_duplicate_check": audit,
            "references_learned": learned,
        }

    def _validate_request(self, req: WriterRequest) -> None:
        """Validate service inputs after schema validation."""
        if req.content_type not in CONTENT_TYPES:
            raise ValueError(f"Loại nội dung không hợp lệ: {req.content_type}")
        if req.tone not in TONE_OPTIONS:
            raise ValueError(f"Giọng điệu không hợp lệ: {req.tone}")
        if req.audience not in AUDIENCE_OPTIONS:
            raise ValueError(f"Đối tượng không hợp lệ: {req.audience}")
        if req.output_format not in FORMAT_OPTIONS:
            raise ValueError(f"Format không hợp lệ: {req.output_format}")
        if req.language not in LANGUAGE_INSTRUCTIONS:
            raise ValueError(f"Ngôn ngữ không hợp lệ: {req.language}")

    async def _generate_one(
        self,
        req: WriterRequest,
        title_hint: str,
        chunks: list[RetrievedChunk],
        batch_id: str,
        *,
        variant_index: int,
    ) -> dict:
        """Generate a title, article, web audit, and learned references."""
        generated_title, title_fallback = await self._generate_title(
            req, title_hint, chunks, variant_index
        )
        prompt, system = self._build_prompt(
            req, generated_title, chunks, variant_index=variant_index
        )
        content = await self.vllm.chat(
            [{"role": "user", "content": prompt}],
            system=system,
            model=self.model,
            temperature=0.85,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        web_check: dict = {}
        if req.check_web_duplicates and self.dupe_checker is not None and content.strip():
            try:
                web_check = await self.dupe_checker.check(generated_title, content)
            except Exception:
                logger.warning("writer.webcheck_error", exc_info=True)
                web_check = {"status": "unavailable", "errors": ["web_check_failed"]}

        base_meta = {
            "title_hint": title_hint,
            "generated_title": generated_title,
            "title_generation_fallback": title_fallback,
            "tone": req.tone,
            "audience": req.audience,
            "word_count": req.word_count,
            "output_format": req.output_format,
            "language": req.language,
            "model": self.model,
            "extra": req.extra,
            "variant": variant_index,
            "batch_id": batch_id,
        }
        if web_check:
            base_meta["web_duplicate_check"] = web_check
        gen = GeneratedContent(
            title=generated_title,
            content=content,
            content_type=req.content_type,
            status="generated",
            source_documents=[chunk.to_dict() for chunk in chunks],
            tags=req.keywords,
            meta=base_meta,
        )
        async with self.session_factory() as session:
            session.add(gen)
            await session.commit()
            await session.refresh(gen)

        ingest_status = await self._ingest_generated_content(
            req, content, generated_title, gen.id, batch_id, web_check
        )
        learned_page_count = await self._ingest_reference_pages(
            req,
            generated_title,
            title_hint,
            gen.id,
            batch_id,
            web_check,
        )
        base_meta["ingest_status"] = ingest_status
        base_meta["ingest_reference_page_count"] = learned_page_count
        async with self.session_factory() as session:
            db_gen = await session.get(GeneratedContent, gen.id)
            if db_gen is not None:
                db_gen.meta = base_meta
                await session.commit()
        logger.info(
            "writer.variant_done",
            variant=variant_index,
            content_id=str(gen.id),
            web=web_check.get("status", "none"),
            ingest=ingest_status,
            references=learned_page_count,
        )
        return {
            "content_id": str(gen.id),
            "content_type": gen.content_type,
            "topic": gen.title,
            "content": gen.content,
            "meta": base_meta,
        }

    async def _generate_title(
        self,
        req: WriterRequest,
        title_hint: str,
        chunks: list[RetrievedChunk],
        variant_index: int,
    ) -> tuple[str, bool]:
        """Generate a concise title from the hint and retrieved Braind context."""
        prompt = (
            f"Title hint: {title_hint}\n"
            f"Content type: {CONTENT_TYPES[req.content_type]['label']}\n"
            f"Variant: {variant_index}\n"
            f"Keywords: {', '.join(req.keywords) or '(none)'}\n\n"
            "Return one concise final title only. Do not add quotes, markdown, or explanation.\n\n"
            f"Braind context:\n{self._format_context(chunks)}"
        )
        try:
            raw_title = await self.vllm.chat(
                [{"role": "user", "content": prompt}],
                system=(
                    "Create an accurate, non-sensational title grounded only in "
                    "the supplied context."
                ),
                model=self.model,
                temperature=0.35,
                max_tokens=100,
            )
        except Exception:
            logger.warning("writer.title_generation_failed", exc_info=True)
            return title_hint[:500], True
        title = self._clean_generated_title(raw_title)
        return (title, False) if title else (title_hint[:500], True)

    @staticmethod
    def _clean_generated_title(value: str) -> str:
        """Normalize one model-generated title into a database-safe value."""
        title = " ".join((value or "").replace("\n", " ").split())
        title = title.removeprefix("#").strip().strip("\"'`")
        return title[:500]

    async def _ingest_generated_content(
        self,
        req: WriterRequest,
        content: str,
        generated_title: str,
        generation_id: uuid.UUID,
        batch_id: str,
        web_check: dict,
    ) -> str:
        """Ingest the generated artifact without allowing it into normal RAG."""
        if not req.ingest_to_brain or self.ingest is None or not content.strip():
            return "skipped"
        try:
            result = await self.ingest.ingest_text(
                content,
                title=f"Generated: {generated_title}",
                source_type="writer_generated",
                tags=[*req.keywords, "writer-generated"],
                meta={
                    "writer_generation_id": str(generation_id),
                    "batch_id": batch_id,
                    "content_type": req.content_type,
                    "provenance_role": "generated_output",
                    "web_similarity_summary": {
                        "status": web_check.get("status"),
                        "highest_similarity_percent": web_check.get("highest_similarity_percent"),
                        "highest_semantic_percent": web_check.get("highest_semantic_percent"),
                    },
                },
            )
            return "ingested" if result.added else "duplicate"
        except Exception:
            logger.warning("writer.ingest_failed", exc_info=True)
            return "error"

    async def _ingest_reference_pages(
        self,
        req: WriterRequest,
        generated_title: str,
        title_hint: str,
        generation_id: uuid.UUID,
        batch_id: str,
        web_check: dict,
        *,
        ingest_enabled: bool | None = None,
    ) -> int:
        """Learn fetched external pages with full-text provenance metadata.

        Args:
            ingest_enabled: When ``None`` (default), the request's
                ``ingest_to_brain`` flag decides whether pages are learned.
                Pass an explicit bool to override it — used by the post-creation
                "check web" flow, which always learns the fetched pages.
        """
        if ingest_enabled is None:
            ingest_enabled = req.ingest_to_brain
        if not ingest_enabled or self.ingest is None:
            return 0
        learned = 0
        for page in web_check.get("pages") or []:
            text = str(page.get("page_text") or "").strip()
            if not page.get("fetched") or len(text) < 120:
                continue
            try:
                await self.ingest.ingest_text(
                    text,
                    title=f"Writer web reference: {str(page.get('title') or '')[:180]}",
                    source_type="writer_duplicate_reference",
                    url=page.get("url"),
                    tags=["writer-web-reference", *req.keywords],
                    meta={
                        "writer_generation_id": str(generation_id),
                        "batch_id": batch_id,
                        "provenance_role": "external_full_page_reference",
                        "title_hint": title_hint,
                        "generated_title": generated_title,
                        "canonical_url": page.get("canonical_url"),
                        "search_queries": page.get("queries") or web_check.get("queries") or [],
                        "lexical_similarity_percent": page.get("similarity_percent"),
                        "semantic_similarity_percent": page.get("semantic_percent"),
                        "match_type": page.get("match_type"),
                        "checked_at": web_check.get("checked_at"),
                        "seo": page.get("seo") or {},
                    },
                )
                learned += 1
            except Exception:
                logger.warning(
                    "writer.reference_ingest_failed",
                    url=str(page.get("url") or "")[:80],
                    exc_info=True,
                )
        return learned

    async def refine(
        self,
        content_id: str,
        instruction: str,
        *,
        temperature: float = 0.6,
    ) -> dict:
        """Edit/refine a previously generated content using the LLM."""
        async with self.session_factory() as session:
            gen = await session.get(GeneratedContent, content_id)
            if gen is None:
                raise FileNotFoundError(f"Không tìm thấy nội dung có id: {content_id}")
            original = gen.content

        system = (
            "Bạn là chuyên gia content nâng cao. Người dùng yêu cầu chỉnh sửa nội dung bên dưới "
            f"theo chỉ dẫn: {instruction}\n"
            "Chỉ trả về nội dung đã chỉnh sửa hoàn chỉnh, không giải thích thêm."
        )
        refined = await self.vllm.chat(
            [{"role": "user", "content": f"NỘI DUNG GỐC:\n{original}"}],
            system=system,
            model=self.model,
            temperature=temperature,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

        async with self.session_factory() as session:
            gen = await session.get(GeneratedContent, content_id)
            gen.content = refined
            gen.meta = {**(gen.meta or {}), "refined_with": instruction}
            await session.commit()
            await session.refresh(gen)
            return {
                "content_id": str(gen.id),
                "content": gen.content,
                "meta": gen.meta,
            }

    async def list_content(
        self, content_type: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict]:
        """List generated content (newest first)."""
        from sqlalchemy import select

        query = select(GeneratedContent).order_by(GeneratedContent.created_at.desc())
        if content_type:
            query = query.where(GeneratedContent.content_type == content_type)
        query = query.limit(limit).offset(offset)
        async with self.session_factory() as session:
            rows = (await session.execute(query)).scalars().all()
            return [
                {
                    "content_id": str(g.id),
                    "content_type": g.content_type,
                    "topic": g.title,
                    "status": g.status,
                    "tags": g.tags,
                    "created_at": g.created_at.isoformat() if g.created_at else None,
                }
                for g in rows
            ]

    async def get_content(self, content_id: str) -> dict:
        """Return the full content + metadata for one generated article.

        Raises ``FileNotFoundError`` when no row matches ``content_id`` or the
        identifier is not a valid UUID.
        """
        try:
            uid = uuid.UUID(str(content_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise FileNotFoundError(f"Không tìm thấy nội dung có id: {content_id}") from exc

        async with self.session_factory() as session:
            gen = await session.get(GeneratedContent, uid)
            if gen is None:
                raise FileNotFoundError(f"Không tìm thấy nội dung có id: {content_id}")
            return {
                "content_id": str(gen.id),
                "content_type": gen.content_type,
                "topic": gen.title,
                "content": gen.content,
                "status": gen.status,
                "tags": gen.tags,
                "created_at": gen.created_at.isoformat() if gen.created_at else None,
                "meta": gen.meta or {},
            }

    # ------------------------------------------------------------------ #
    # Helpers                                                           #
    # ------------------------------------------------------------------ #

    def _build_search_query(self, req: WriterRequest) -> str:
        parts = [req.topic]
        if req.keywords:
            parts.extend(req.keywords)
        if req.audience:
            parts.append(req.audience)
        return " ".join(parts[:3])

    def _build_prompt(
        self,
        req: WriterRequest,
        generated_title: str,
        chunks: list[RetrievedChunk],
        *,
        variant_index: int = 1,
    ) -> tuple[str, str]:
        """Return (user_prompt, system_prompt) for one variant."""
        info = CONTENT_TYPES[req.content_type]

        context = self._format_context(chunks)

        # Which extra fields does this content type's prompt use?
        used_fields = set()
        for f in info["extra_fields"]:
            if f in ("keywords", "tone", "audience"):
                used_fields.add(f)
        fields = req.context_fields(used_fields)
        field_lines = "\n".join(f"- {k}: {v}" for k, v in fields.items()) or "- (không có)"

        selectors = _selectors_to_text(req)

        user_content = f"""
YÊU CẦU NỘI DUNG ({info["label"]}) — BIẾN THỂ {variant_index}:
- Tiêu đề bài viết: {generated_title}
{field_lines}
{selectors}
- Chọn một cách tiếp cận phù hợp với tiêu đề, nội dung nguồn và đối tượng đọc.

{info["instructions"]}
"""
        system = f"""{info["instructions"]}

# CHỐNG TRÙNG:
- CHỈ dùng kiến thức trong # KIẾN THỨC để viết; KHÔNG bịa số liệu, trích dẫn, tên, URL.
- Không chép nguyên văn câu dài từ tài liệu; hãy diễn đạt lại (paraphrase) và tổng hợp.
- Không thêm trích dẫn nguồn URL ngoài vào nội dung — chỉ dùng nội dung như kiến thức nền.

# KIẾN THỨC (từ kho dữ liệu):
{context}
"""
        return user_content.strip(), system.strip()

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks as grounding context, within a token budget.

        The chunker now emits whole sections rather than fragments, so eight
        chunks can exceed the model's context window on their own — the writer
        failed with a 400 the first time a well-chunked source was retrieved.
        Chunks arrive ranked, so truncation drops the least relevant material.
        """
        if not chunks:
            return "(Không có tài liệu liên quan được tìm thấy trong kho dữ liệu.)"
        parts: list[str] = []
        used = 0
        for i, c in enumerate(chunks, 1):
            block = f"--- [Tài liệu {i}: {c.title}] ---\n{c.content}"
            cost = count_tokens(block)
            if used + cost > MAX_CONTEXT_TOKENS:
                if not parts:
                    # Even one chunk overflows: keep a truncated head so the
                    # writer still has something to ground on.
                    parts.append(block[: MAX_CONTEXT_TOKENS * 3])
                break
            parts.append(block)
            used += cost
        return "\n\n".join(parts)
