"""Braind Core - Text chunking utilities.

Splits markdown/text into overlapping chunks optimal for embedding and
retrieval. Uses tiktoken-aware boundaries to avoid splitting code blocks
and to keep semantic units together.

Two chunkers:
- ``TextChunker``: legacy flat, overlapping paragraph chunking.
- ``HierarchicalChunker``: semantic **tree** chunking (Brain Second §5/§6).
  It preserves the document → section → paragraph hierarchy, attaches a
  full context contract (section_path, parent_id, prev/next, context_summary,
  byte offsets) so a small chunk always knows the document, section, topic
  and context it belongs to (golden CASE A — no context loss).
"""

from __future__ import annotations

import hashlib
import re
import uuid

from core.config import settings

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None and tiktoken is not None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _ENCODER


def count_tokens(text: str) -> int:
    """Estimate token count for a text."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return len(text) // 4  # fallback heuristic


class TextChunker:
    """Splits text into overlapping chunks using markdown-aware boundaries."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        split_on: str = "\n\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_on = split_on

    def _clean(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        text = self._clean(text)
        if not text:
            return []
        if count_tokens(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        # First split by double newlines (paragraphs / markdown blocks)
        units = text.split("\n\n")
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                    # Keep overlap from the tail of current
                    tail = current[-count_tokens(current) * 3 // 2 :] if current else ""
                    current = f"{tail.strip()}\n\n{unit}".strip()
                else:
                    # Unit is larger than chunk_size — hard split
                    sub_chunks = self._hard_split(unit)
                    chunks.extend(sub_chunks[:-1])
                    current = sub_chunks[-1]
        if current:
            chunks.append(current)

        # Merge tiny trailing chunks into previous if overlap is small
        merged: list[str] = []
        for c in chunks:
            if (
                merged
                and count_tokens(merged[-1]) + count_tokens(c)
                <= self.chunk_size + self.chunk_overlap
            ):
                merged[-1] = f"{merged[-1]}\n\n{c}"
            else:
                merged.append(c)
        return merged

    def _hard_split(self, text: str) -> list[str]:
        """Split oversized paragraph into sentence-boundary chunks."""
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        chunks: list[str] = []
        current = ""
        for s in sentences:
            candidate = f"{current} {s}".strip()
            if count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = s if count_tokens(s) <= self.chunk_size else s[: self.chunk_size * 3]
        if current:
            chunks.append(current)
        return chunks or [text]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Convenience function to chunk text."""
    return TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).chunk(text)


# ---------------------------------------------------------------------------
# Hierarchical (semantic) chunking — Brain Second Phase 5
# ---------------------------------------------------------------------------


class SemanticChunk:
    """A node in the semantic **tree** of a document.

    Every node carries the full context contract required by the master
    prompt §6: section_path, parent link, prev/next sibling, byte offsets in
    the raw document, a self-contained context summary, and the document it
    belongs to.  Combined with the ``chunks`` table this lets retrieval walk
    children ↔ parents and embed with context (CASE A — no context loss).
    """

    __slots__ = (
        "chunk_id",
        "document_id",
        "title",
        "content",
        "section_path",
        "section_heading",
        "hierarchy_level",
        "start_offset",
        "end_offset",
        "parent_id",
        "prev_chunk_id",
        "next_chunk_id",
        "context_summary",
        "chunk_kind",
        "meta",
        "root_id",
        "order",
        "page_number",
        "section_number",
        "source_location",
        "content_fingerprint",
    )

    def __init__(
        self,
        *,
        content: str,
        chunk_kind: str,
        hierarchy_level: int,
        section_path: str,
        start_offset: int,
        end_offset: int,
        document_id: str | None = None,
        title: str = "",
        section_heading: str | None = None,
        parent_id: str | None = None,
        context_summary: str = "",
        chunk_id: str | None = None,
        root_id: str | None = None,
        order: int | None = None,
        page_number: int | None = None,
        section_number: int | None = None,
        source_location: str | None = None,
        content_fingerprint: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.document_id = document_id
        self.title = title
        self.content = content
        self.chunk_kind = chunk_kind  # document | section | paragraph | table | list
        self.hierarchy_level = hierarchy_level
        self.section_path = section_path
        self.section_heading = section_heading
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.parent_id = parent_id
        self.prev_chunk_id: str | None = None
        self.next_chunk_id: str | None = None
        self.context_summary = context_summary
        self.meta: dict = {}
        # Phase 5 provenance (additive; set by :meth:`HierarchicalChunker.chunk`).
        self.root_id = root_id
        self.order = order
        self.page_number = page_number
        self.section_number = section_number
        self.source_location = source_location
        self.content_fingerprint = content_fingerprint

    def to_row(self) -> dict:
        """Return a dict shaped for the ``chunks`` table (Phase 3 + 5 schema)."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "parent_id": self.parent_id,
            "prev_chunk_id": self.prev_chunk_id,
            "next_chunk_id": self.next_chunk_id,
            "section_path": self.section_path,
            "hierarchy_level": self.hierarchy_level,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "context_summary": self.context_summary,
            "content": self.content,
            "root_id": self.root_id,
            "order": self.order,
            "page_number": self.page_number,
            "section_number": self.section_number,
            "source_location": self.source_location,
            "content_fingerprint": self.content_fingerprint,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"SemanticChunk(kind={self.chunk_kind}, level={self.hierarchy_level}, "
            f"path={self.section_path!r}, chars={len(self.content)})"
        )


class _ParsedBlock:
    """Internal parse result: heading or content block with raw offsets."""

    __slots__ = ("kind", "level", "text", "start", "end")

    def __init__(self, kind: str, level: int, text: str, start: int, end: int):
        self.kind = kind  # heading | content
        self.level = level  # heading depth (1-6) or 0 for content
        self.text = text
        self.start = start
        self.end = end


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class HierarchicalChunker:
    """Parse markdown/text into a semantic chunk **tree**.

    Strategy (per §5 — never hard character counts as the primary method):

    1. Parse the document into heading + content blocks with raw byte offsets.
    2. A heading opens a section at its level; content belongs to the
       currently open section.  Section paths are built from the heading
       stack (e.g. ``Database > Logging > Retention``).
    3. Section nodes are emitted once per heading; oversized paragraphs are
       sub-split at sentence boundaries only when the *semantic* grouping
       exceeds the chunk budget.
    4. Parent/prev/next relationships are assigned so retrieval can walk the
       tree and auto-merge children → parent context (Phase 10).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        max_section_chars: int | None = None,
        min_chunk_chars: int | None = None,
    ):
        self.chunk_size = chunk_size
        # Sections themselves are not embedded wholesale; keep them bounded
        # to what a parent chunk may realistically need for context.
        self.max_section_chars = max_section_chars or chunk_size * 3
        # Below this, a block is a fragment (a table cell, a label) and is
        # merged with its neighbours rather than embedded on its own.
        self.min_chunk_chars = min_chunk_chars or settings.chunk_min_chars

    def chunk(
        self,
        text: str,
        *,
        document_id: str | None = None,
        title: str = "",
        compact: bool = False,
        prune: bool = True,
        root_path: str = "",
    ) -> list[SemanticChunk]:
        """Return the semantic chunk tree for ``text`` in document order.

        ``compact`` is the storage mode the ingest pipeline uses: runs of short
        blocks are merged, undersized content is folded into its owning
        section, and sections that absorbed nothing are dropped. Without it the
        chunk table fills with bare labels — Office and PDF extraction emit one
        block per table cell, which produced 1085 chunks with a median length
        of 13 characters from a single report.

        It is off by default because the uncompacted tree — one node per
        heading, one per paragraph — is what structural callers inspect.

        ``prune`` (only meaningful with ``compact``) controls the second, lossy
        half of compaction: folding undersized nodes into their section and
        dropping sections that absorbed nothing. The retrieval-facing ingest
        path passes ``prune=False`` — it wants the coalescing (which merges
        table-cell fragments and *raises* coverage) without the pruning, which
        discards content a retrieval row must still be able to return.

        ``root_path`` prefixes every ``section_path``. Web pages carry no
        markdown headings, so without it their chunks have no locator at all.
        """
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip("\n")
        if not cleaned:
            return []
        blocks = self._parse(cleaned)
        nodes = self._build_tree(blocks, compact=compact)
        if compact and prune:
            nodes = self._prune_fragments(nodes)

        # Linear order: prev/next within the flattened list.
        for i, node in enumerate(nodes):
            if i > 0:
                node.prev_chunk_id = nodes[i - 1].chunk_id
            if i + 1 < len(nodes):
                node.next_chunk_id = nodes[i + 1].chunk_id

        root = root_path.strip()
        for node in nodes:
            node.document_id = document_id
            node.title = title
            if root:
                node.section_path = f"{root} > {node.section_path}" if node.section_path else root
            if not node.context_summary:
                node.context_summary = _describe(node.content, node.section_path)

        # Phase 5: provenance (order/section/location/fingerprint). Additive —
        # lets a chunk be relocated in the source and idempotently re-processed.
        by_id = {n.chunk_id: n for n in nodes}
        for i, node in enumerate(nodes):
            node.order = i
            node.content_fingerprint = _sha256(node.content)
            node.section_number = _section_depth(node.section_path)
            node.root_id = _root_id(node, by_id)
            node.source_location = _source_location(node)
        return nodes

    def _parse(self, text: str) -> list[_ParsedBlock]:
        """Split text into heading/content blocks keeping raw offsets."""
        blocks: list[_ParsedBlock] = []
        lines = text.split("\n")
        offset = 0
        content_lines: list[tuple[str, int, int]] = []  # (text, start, end)

        def flush_content() -> None:
            if not content_lines:
                return
            joined = "\n".join(line[0] for line in content_lines)
            blocks.append(
                _ParsedBlock(
                    "content", 0, joined, content_lines[0][1], content_lines[-1][2]
                )
            )
            content_lines.clear()

        for line in lines:
            match = _HEADING_RE.match(line.lstrip())
            if match:
                flush_content()
                level = len(match.group(1))
                blocks.append(
                    _ParsedBlock(
                        "heading",
                        level,
                        match.group(2).strip(),
                        offset,
                        offset + len(line),
                    )
                )
            else:
                stripped = line.strip()
                if stripped:
                    content_lines.append((line, offset, offset + len(line)))
                else:
                    # A blank line separates paragraphs: keep content flowing but
                    # remember the break so paragraphs stay distinct for section
                    # grouping below.
                    content_lines.append(("", offset, offset))
            offset += len(line) + 1  # +1 for the newline

        flush_content()
        return blocks

    def _build_tree(
        self, blocks: list[_ParsedBlock], *, compact: bool = False
    ) -> list[SemanticChunk]:
        """Group blocks into document/section/paragraph nodes with links."""
        nodes: list[SemanticChunk] = []
        # Heading stack: (level, heading text) with matching section node.
        stack: list[tuple[int, str, SemanticChunk | None]] = []

        for block in blocks:
            if block.kind == "heading":
                # Pop deeper headings; keep the current section chain.
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                parent_node = stack[-1][2] if stack else None
                section_path = self._path_for(stack, block)
                node = SemanticChunk(
                    chunk_kind="section",
                    hierarchy_level=block.level,
                    section_path=section_path,
                    section_heading=block.text,
                    content=block.text,
                    start_offset=block.start,
                    end_offset=block.end,
                    parent_id=parent_node.chunk_id if parent_node else None,
                )
                nodes.append(node)
                stack.append((block.level, block.text, node))
                continue

            # Content block: assign to the current open section.
            parent_node = stack[-1][2] if stack else None
            section_path = self._path_for(stack, None)
            self._emit_content(
                text=block.text,
                start=block.start,
                end=block.end,
                section_path=section_path,
                level=(stack[-1][0] if stack else 1),
                parent_node=parent_node,
                nodes=nodes,
                compact=compact,
            )

        return nodes

    @staticmethod
    def _path_for(
        stack: list[tuple[int, str, SemanticChunk | None]],
        current: _ParsedBlock | None,
    ) -> str:
        parts = [entry[1] for entry in stack]
        if current is not None:
            parts.append(current.text)
        return " > ".join(parts) or ""

    def _emit_content(
        self,
        *,
        text: str,
        start: int,
        end: int,
        section_path: str,
        level: int,
        parent_node: SemanticChunk | None,
        nodes: list[SemanticChunk],
        compact: bool = False,
    ) -> None:
        """Emit one or more paragraph-level chunks for a content block.

        Long paragraphs are split at sentence boundaries only when needed;
        short ones are kept intact to preserve meaning (§5). Tables and lists
        keep their raw structure (pipe rows / indentation) — never
        sentence-split inside a cell or list item (Phase 5).

        Consecutive short blocks are coalesced first. Office and PDF extraction
        emit one blank-line-separated block per table cell, so splitting on
        blank lines alone produced chunks like ``Yes`` and ``6379`` — 80% of the
        chunk table was under 200 characters and carried no retrievable meaning.
        """
        trimmed = text.strip()
        start += text.find(trimmed)
        paragraphs = re.split(r"\n\s*\n", trimmed)
        if compact:
            paragraphs = self._coalesce(paragraphs)
        for paragraph in paragraphs:
            para = paragraph.strip()
            if not para:
                continue
            kind = _content_kind(para)
            if count_tokens(para) <= self.chunk_size or kind in ("table", "list"):
                nodes.append(
                    SemanticChunk(
                        chunk_kind=kind,
                        hierarchy_level=level + 1,
                        section_path=section_path,
                        content=para,
                        start_offset=start,
                        end_offset=start + len(para),
                        parent_id=parent_node.chunk_id if parent_node else None,
                        context_summary=_describe(para, section_path),
                    )
                )
            else:
                # Sentence-boundary sub-split, seeded with section context.
                fragments = _split_long_paragraph(para, self.chunk_size)
                for frag in fragments:
                    nodes.append(
                        SemanticChunk(
                            chunk_kind="paragraph",
                            hierarchy_level=level + 1,
                            section_path=section_path,
                            content=frag,
                            start_offset=start,
                            end_offset=start + len(frag),
                            parent_id=parent_node.chunk_id if parent_node else None,
                            context_summary=_describe(frag, section_path),
                        )
                    )
            start += len(paragraph) + 1

    def _coalesce(self, blocks: list[str]) -> list[str]:
        """Merge runs of short blocks so no chunk is a bare label or value.

        A block that already stands on its own is passed through untouched; a
        run of fragments is joined with newlines, which keeps a label/value
        table readable as one unit instead of scattering it across rows.
        """
        merged: list[str] = []
        buffer: list[str] = []
        buffer_len = 0

        def flush() -> None:
            nonlocal buffer_len
            if buffer:
                merged.append("\n".join(buffer))
                buffer.clear()
                buffer_len = 0

        for raw in blocks:
            block = raw.strip()
            if not block:
                continue
            if len(block) >= self.min_chunk_chars and not buffer:
                merged.append(block)
                continue
            buffer.append(block)
            buffer_len += len(block) + 1
            if buffer_len >= self.min_chunk_chars:
                flush()
        flush()
        return merged

    def _prune_fragments(self, nodes: list[SemanticChunk]) -> list[SemanticChunk]:
        """Fold undersized content nodes into the section that owns them.

        A heading followed by one short line — an outline, a note template — is
        two chunks that each mean nothing alone. Merging the content up into
        its section keeps the heading's context attached to the text and stops
        the chunk table filling with 3-character rows.
        """
        by_id = {n.chunk_id: n for n in nodes}
        dropped: set[str] = set()
        for node in nodes:
            if node.chunk_kind == "section" or len(node.content) >= self.min_chunk_chars:
                continue
            parent = by_id.get(node.parent_id or "")
            if parent is None or parent.chunk_kind != "section":
                continue
            parent.content = f"{parent.content}\n{node.content}".strip()
            parent.end_offset = max(parent.end_offset, node.end_offset)
            dropped.add(node.chunk_id)

        kept = [n for n in nodes if n.chunk_id not in dropped]
        # Sections that absorbed nothing and carry only their own heading are
        # navigation, not knowledge.
        pruned = [
            n
            for n in kept
            if n.chunk_kind != "section" or len(n.content) >= self.min_chunk_chars
        ]
        # A document shorter than one chunk still has to be retrievable: keep
        # the longest node rather than storing a source with no chunks at all.
        if not pruned and kept:
            pruned = [max(kept, key=lambda n: len(n.content))]

        surviving = {n.chunk_id for n in pruned}
        for node in pruned:
            if node.parent_id not in surviving:
                node.parent_id = None
        return pruned


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences at terminal punctuation (never mid-sentence).

    Shared by the fixed-size splitter and the LLM topic-shift segmenter so both
    reason about the same units: the segmenter picks *which* sentence starts a
    new topic, and never invents a boundary inside one.
    """
    return [s for s in re.split(r"(?<=[.!?。！？])\s+", text.strip()) if s.strip()]


def _split_long_paragraph(text: str, chunk_size: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries (never mid-sentence)."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if count_tokens(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = (
                sentence
                if count_tokens(sentence) <= chunk_size
                else sentence[: chunk_size * 4]
            )
    if current:
        chunks.append(current)
    return chunks or [text]


# --- Phase 5 structural helpers ---------------------------------------------

_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+\S"
)
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_PIPE_COUNT = 2  # a table row has leading and trailing or interior pipes


def _content_kind(text: str) -> str:
    """Classify a paragraph block as paragraph / list / table.

    Tables and lists are kept as a single structured chunk (never sentence-split
    inside a cell or item). Heuristic detection is deliberately cheap:
      - a table looks like ≥2 pipe rows (or a single row still preserved).
      - a list has ≥1 ``-``/``*``/``+``/numbered item line.
    """
    lines = text.split("\n")
    pipe_rows = sum(1 for ln in lines if _TABLE_ROW_RE.match(ln) and ln.count("|") >= 2)
    list_rows = sum(1 for ln in lines if _LIST_MARKER_RE.match(ln))
    if pipe_rows >= 2 or (lines and _TABLE_ROW_RE.match(lines[0]) and "|" in lines[0]):
        return "table"
    if list_rows and list_rows >= 1 and list_rows >= len(lines) - 1:
        return "list"
    return "paragraph"


def _sha256(text: str) -> str:
    """sha256 content fingerprint (idempotent reprocessing key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _section_depth(section_path: str | None) -> int:
    """Number of section segments in ``section_path`` (a hierarchy ordinal)."""
    if not section_path:
        return 0
    return len([s for s in section_path.split(">") if s.strip()])


def _root_id(node: SemanticChunk, by_id: dict[str, SemanticChunk]) -> str | None:
    """Chunk id of the top-most ancestor of ``node`` (its subtree root)."""
    if node.parent_id is None:
        return node.chunk_id
    seen: set[str] = set()
    cur = node
    root = cur.chunk_id
    while cur.parent_id is not None and cur.parent_id in by_id and cur.parent_id not in seen:
        seen.add(cur.parent_id)
        cur = by_id[cur.parent_id]
        root = cur.chunk_id
    return root


def _source_location(node: SemanticChunk) -> str | None:
    """A human-locatable reference like ``§AI/Models > Retriever``."""
    if node.section_path:
        return f"§{node.section_path} / para {node.order or 0}"
    if node.section_heading:
        return f"§{node.chunk_kind}:{node.section_heading}"
    return f"{node.chunk_kind} #{node.order or 0}"


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (period/?/!/newline terminated)."""
    if not text:
        return ""
    match = re.split(r"[.!?。！？]|(?:\s+[A-Z])", text, maxsplit=1)
    first = match[0].strip()
    # Re-attach the terminating punctuation we split on, if any remains.
    if len(match) > 1 and text[len(match[0]) :]:
        first += text[len(match[0])].strip()
    return first[:200]


def _describe(text: str, section_path: str) -> str:
    """Build a self-contained context summary for a chunk.

    Prefers explicit heading context, then the first sentence, then a head
    of the text — so a retrieved chunk always carries its topic locally
    (golden CASE A).
    """
    if section_path:
        return f"Topic: {section_path}"
    first = _first_sentence(text)
    if first:
        return f"Topic: {first[:160]}"
    return f"Topic: {text[:160]}"
