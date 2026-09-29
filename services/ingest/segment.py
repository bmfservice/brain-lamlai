"""LLM topic-shift segmentation for unstructured documents (§2).

Structure-aware chunking (``core.chunking.HierarchicalChunker``) handles anything
with headings. What it cannot help is the document that has *no* structure at
all: a transcript, an OCR dump, a pasted email thread, a scraped page whose
markup was flattened. There the only real signal for "where does one topic end"
is meaning, and the ingestion contract is explicit about how to read it:

- **Never cut on embedding similarity alone.** A cosine threshold between
  adjacent windows is cheap and wrong — it fires on vocabulary drift inside one
  argument and stays silent across a clean topic switch that reuses the same
  words. The boundary decision is made by a model *reading* the text.
- **The model chooses among sentences, it does not rewrite them.** It returns
  indices into a sentence list, so a hallucinated or duplicated index can be
  dropped without corrupting content. Nothing the model emits reaches storage.
- **Fixed-size stays the safety net.** If the model is unavailable, returns
  nothing usable, or the text is too big to segment on a 12GB box, we fall back
  to sentence-boundary packing and say so in the log.

Every returned segment is a contiguous run of the original sentences, so the
concatenation of all segments reproduces the input — verified by the caller's
coverage log rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.chunking import count_tokens, split_sentences
from core.config import settings
from core.logging import get_logger
from core.vllm import VLLMClient

logger = get_logger(__name__)

_SEGMENT_PROMPT = """You segment an unstructured document into topic sections.

You are given the document as a numbered list of sentences. Decide which
sentences START a new topic.

Return ONLY a JSON object:
{"boundaries": [<sentence numbers that start a new topic>]}

Rules:
- Sentence 0 always starts the first topic; do NOT include 0.
- A boundary means the subject genuinely changes: a new question, a new entity
  under discussion, a shift from problem to solution, a new procedure step group.
- Do NOT create a boundary for a mere change of wording, an example, or a
  supporting sentence that continues the same point.
- Aim for topics of several sentences. Prefer FEWER, more meaningful boundaries.
- Use only sentence numbers that exist in the list. No other keys, no prose."""


@dataclass
class Segment:
    """One topic-coherent run of sentences."""

    text: str
    start_sentence: int
    end_sentence: int  # exclusive
    method: str  # "llm_topic_shift" | "sentence_pack"


class TopicShiftSegmenter:
    """Split heading-less prose into topic segments using an LLM (§2).

    The model only proposes sentence indices; this class validates them, drops
    anything out of range or out of order, enforces a minimum segment size so a
    chatty model cannot shred the document, and packs sentences at boundaries
    when the model is unusable.
    """

    def __init__(
        self,
        vllm: VLLMClient,
        *,
        chunk_size: int | None = None,
        min_sentences: int | None = None,
        max_sentences: int | None = None,
        pack_chars: int | None = None,
    ):
        self.vllm = vllm
        self.chunk_size = chunk_size or settings.chunk_size
        # Below this a "topic" is a sentence fragment, not a topic.
        self.min_sentences = min_sentences or settings.segment_min_sentences
        # Above this the prompt no longer fits a 7B context on one GPU.
        self.max_sentences = max_sentences or settings.segment_max_sentences
        # Character budget for the fallback packer. It is the same threshold that
        # declared the block too long to be one topic: if we cannot ask a model
        # where the topic changes, we at least stop returning one oversized blob.
        self.pack_chars = pack_chars or settings.segment_trigger_chars

    async def segment(self, text: str, *, title: str = "") -> list[Segment]:
        """Return topic segments for ``text``, longest-effort then fallback."""
        sentences = split_sentences(text)
        if len(sentences) <= self.min_sentences:
            return self._pack(sentences, reason="too_few_sentences")

        if len(sentences) > self.max_sentences:
            logger.info(
                "segment.too_large",
                title=title[:60],
                sentences=len(sentences),
                limit=self.max_sentences,
            )
            return self._pack(sentences, reason="too_many_sentences")

        boundaries = await self._ask_boundaries(sentences, title=title)
        if not boundaries:
            return self._pack(sentences, reason="no_llm_boundaries")

        segments = self._apply(sentences, boundaries, method="llm_topic_shift")
        logger.info(
            "segment.topic_shift",
            title=title[:60],
            sentences=len(sentences),
            boundaries=len(boundaries),
            segments=len(segments),
        )
        return segments

    async def _ask_boundaries(self, sentences: list[str], *, title: str) -> list[int]:
        """Ask the model for boundary indices; return only valid, ordered ones."""
        numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sentences))
        try:
            payload = await self.vllm.chat_json(
                [{"role": "user", "content": f"Document: {title}\n\n{numbered}"}],
                system=_SEGMENT_PROMPT,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("segment.llm_failed", title=title[:60], error=str(exc))
            return []

        raw = payload.get("boundaries") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            return []

        # Model output is untrusted: keep only in-range integers, deduplicated
        # and ordered. A hallucinated index is dropped, never clamped onto a
        # real sentence it does not describe.
        seen: set[int] = set()
        valid: list[int] = []
        for item in raw:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 < idx < len(sentences) and idx not in seen:
                seen.add(idx)
                valid.append(idx)
        valid.sort()
        if len(valid) != len(raw):
            logger.debug(
                "segment.boundaries_filtered",
                title=title[:60],
                proposed=len(raw),
                kept=len(valid),
            )
        return valid

    def _apply(self, sentences: list[str], boundaries: list[int], *, method: str) -> list[Segment]:
        """Cut ``sentences`` at ``boundaries``, enforcing the size limits.

        A boundary that would leave fewer than ``min_sentences`` in the open
        segment is dropped rather than honoured: the topic keeps running instead
        of emitting a one-line "topic". The final boundary is always the end of
        the document, so every sentence lands in exactly one segment.
        """
        segments: list[Segment] = []
        start = 0
        end = len(sentences)
        for boundary in [*boundaries, end]:
            if boundary != end and boundary - start < self.min_sentences:
                continue
            segments.extend(self._budget(sentences, start, boundary, method=method))
            start = boundary
        return [s for s in segments if s.text.strip()]

    def _budget(
        self,
        sentences: list[str],
        start: int,
        end: int,
        *,
        method: str,
        max_chars: int | None = None,
    ) -> list[Segment]:
        """Split one topic run that exceeds the token (or character) budget."""
        out: list[Segment] = []
        current: list[str] = []
        current_start = start
        for i in range(start, end):
            candidate = [*current, sentences[i]]
            joined = " ".join(candidate)
            too_long = count_tokens(joined) > self.chunk_size or (
                max_chars is not None and len(joined) > max_chars
            )
            if current and too_long:
                out.append(
                    Segment(
                        text=" ".join(current),
                        start_sentence=current_start,
                        end_sentence=i,
                        method=method,
                    )
                )
                current = [sentences[i]]
                current_start = i
            else:
                current = candidate
        if current:
            out.append(
                Segment(
                    text=" ".join(current),
                    start_sentence=current_start,
                    end_sentence=end,
                    method=method,
                )
            )
        return out

    def _pack(self, sentences: list[str], *, reason: str) -> list[Segment]:
        """Fallback: pack whole sentences up to the token budget (§1 safety net)."""
        if not sentences:
            return []
        segments = self._budget(
            sentences, 0, len(sentences), method="sentence_pack", max_chars=self.pack_chars
        )
        logger.info("segment.fallback", reason=reason, segments=len(segments))
        return segments
