"""LLM-judged relation verification (Giai đoạn 3 — evidence-based linking).

The lexical extractor (``services.extract.relations``) proposes candidates from
verb cues and co-occurrence. This module asks the chat model to *judge* each
candidate: is the proposed relation actually supported by the quoted sentence,
and which taxonomy type does the sentence assert? The model must quote the
evidence and justify its ruling — an edge without a readable rationale is not
worth storing (§12).

House rules preserved:

- The LLM never sees or uses embedding cosine values; its judgment is an
  independent *semantic* signal on top of structural/lexical ones (§9).
- Model output is coerced onto ``VALID_EDGE_TYPES``; anything unknown degrades
  to ``POSSIBLE_RELATION`` (capped confidence) in the validator.
- Every failure path degrades to "no LLM signal" — a vLLM outage must never
  fail an ingest.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.logging import get_logger
from core.vllm import VLLMClient
from services.relation.validator import VALID_EDGE_TYPES

logger = get_logger(__name__)

_SYSTEM = (
    "You verify semantic relations between two entities using ONLY the given "
    "supporting sentence. You never use outside knowledge about the entities."
)

# 4 logical relation families (master blueprint) → taxonomy types. The model
# picks a family; we coerce to the project's existing vocabulary.
_FAMILIES = {
    # 4 logical relation families (master blueprint) → taxonomy types.
    "entailment": "DEPENDS_ON",  # A leads to / requires / causes B
    "ontology": "IS_A",  # A is a kind of / part of B
    "contradiction": "CONTRADICTS",  # A conflicts with B
    "corroboration": "SUPPORTS",  # A provides evidence for B
}


@dataclass(frozen=True)
class LLMRelationJudgment:
    """The model's ruling on one candidate pair."""

    supported: bool
    relation_type: str  # coerced to VALID_EDGE_TYPES
    rationale: str
    confidence: float  # model's self-assessed certainty in [0, 1]
    quote: str  # the exact span of the sentence that justifies the ruling


_PROMPT = """Judge the relation between two entities from one sentence.

Sentence: {sentence}
Entity A: {source}
Entity B: {target}
Lexical guess: {proposed}

Decide ONLY from the sentence text:
1. supported — does the sentence actually assert a relationship between A and B?
2. family — one of: entailment (A requires/causes/leads to B), ontology
   (A is a kind of / part of B), contradiction (A conflicts with B),
   corroboration (A backs up or gives evidence for B), none.
3. quote — the shortest span of the sentence that justifies your ruling.
4. rationale — one short sentence explaining the meaning of the relation.
5. confidence — 0.0-1.0 certainty in your ruling.

Return ONLY JSON:
{{"supported": true|false, "family": "entailment|ontology|contradiction|corroboration|none",
  "quote": "...", "rationale": "...", "confidence": 0.0}}"""

_BATCH_PROMPT = """Judge each relation below between two entities, using ONLY
its own sentence. Do not use outside knowledge.

{numbered}

For EACH item decide:
1. supported — does the sentence actually assert a relationship between A and B?
2. family — one of: entailment (A requires/causes/leads to B), ontology
   (A is a kind of / part of B), contradiction (A conflicts with B),
   corroboration (A backs up or gives evidence for B), none.
3. quote — the shortest span of that sentence justifying your ruling.
4. rationale — one short sentence explaining the relation's meaning.
5. confidence — 0.0-1.0 certainty.

Return ONLY JSON with one object per input item, same order:
{{"items": [{{"id": <number>, "supported": true|false,
  "family": "entailment|ontology|contradiction|corroboration|none",
  "quote": "...", "rationale": "...", "confidence": 0.0}}]}}"""


class LLMRelationVerifier:
    """Ask the chat model to confirm/refine lexical relation candidates."""

    def __init__(
        self, vllm: VLLMClient | None, model_version: str = "braind-relation-llm-v1"
    ) -> None:
        self.vllm = vllm
        self.model_version = model_version

    async def judge(
        self,
        *,
        sentence: str,
        source_name: str,
        target_name: str,
        proposed: str,
    ) -> LLMRelationJudgment | None:
        """Judge one candidate. ``None`` when the LLM is unavailable/fails —
        callers then fall back to the lexical-only verdict unchanged."""
        if self.vllm is None:
            return None
        try:
            payload = await self.vllm.chat_json(
                [
                    {
                        "role": "user",
                        "content": _PROMPT.format(
                            sentence=sentence[:1200],
                            source=source_name,
                            target=target_name,
                            proposed=proposed,
                        ),
                    }
                ],
                system=_SYSTEM,
                temperature=0.0,
                max_tokens=300,
            )
        except Exception as exc:
            logger.warning("relation_llm.judge_failed", error=str(exc)[:120])
            return None
        if not isinstance(payload, dict) or "raw" in payload:
            logger.warning("relation_llm.unparseable_response")
            return None
        return self._parse_judgment(payload)

    async def judge_batch(
        self, pairs: list[tuple[str, str, str, str]]
    ) -> list[LLMRelationJudgment | None]:
        """Judge many candidate pairs in a single LLM call (Giai đoạn 5, B3).

        ``pairs`` is a list of ``(sentence, source_name, target_name, proposed)``
        tuples; the return list is aligned by index, each entry a judgment or
        ``None`` when that item could not be decided. One call replaces N —
        the dominant per-ingest latency win. Any item missing or unparseable in
        the batch response degrades to an individual :meth:`judge` call so a
        single malformed entry never poisons the rest of the batch.
        """
        if self.vllm is None or not pairs:
            return [None] * len(pairs)

        numbered = "\n".join(
            f"[{i + 1}] Sentence: {sentence[:1200]}\n"
            f"    Entity A: {source}\n    Entity B: {target}\n"
            f"    Lexical guess: {proposed}"
            for i, (sentence, source, target, proposed) in enumerate(pairs)
        )
        try:
            payload = await self.vllm.chat_json(
                [{"role": "user", "content": _BATCH_PROMPT.format(numbered=numbered)}],
                system=_SYSTEM,
                temperature=0.0,
                max_tokens=300 * len(pairs) + 100,
            )
        except Exception as exc:
            logger.warning("relation_llm.judge_batch_failed", error=str(exc)[:120])
            return [None] * len(pairs)

        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            logger.warning("relation_llm.batch_unparseable", n=len(pairs))
            return [None] * len(pairs)

        # Map each returned item back to its input slot by its ``id``; tolerate
        # ids given as str/float and out-of-range values.
        results: list[LLMRelationJudgment | None] = [None] * len(pairs)
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                slot = int(item.get("id")) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= slot < len(pairs) and results[slot] is None:
                results[slot] = self._parse_judgment(item)

        # Any slot the batch did not answer falls back to a single-pair judge so
        # a malformed entry does not silently drop a real relation.
        missing = [i for i, r in enumerate(results) if r is None]
        if missing and len(missing) < len(pairs):
            for i in missing:
                sentence, source, target, proposed = pairs[i]
                results[i] = await self.judge(
                    sentence=sentence,
                    source_name=source,
                    target_name=target,
                    proposed=proposed,
                )
        return results

    @staticmethod
    def _parse_judgment(payload: dict) -> LLMRelationJudgment:
        """Coerce one model JSON object onto the project taxonomy.

        Shared by :meth:`judge` and :meth:`judge_batch` so the family→type
        mapping and confidence clamping live in exactly one place.
        """
        supported = bool(payload.get("supported"))
        family = str(payload.get("family") or "none").strip().lower()
        quote = str(payload.get("quote") or "").strip()
        rationale = str(payload.get("rationale") or "").strip()
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0

        # Coerce family → project taxonomy; the LLM may also answer with a raw
        # taxonomy type — accept it only if it is in the vocabulary.
        relation_type = _FAMILIES.get(family, "")
        if not relation_type:
            guess = family.upper()
            relation_type = guess if guess in VALID_EDGE_TYPES else ""
        if not supported or family == "none" or not relation_type:
            return LLMRelationJudgment(
                supported=False,
                relation_type="",
                rationale=rationale or "no relation asserted by the sentence",
                confidence=confidence,
                quote=quote,
            )
        return LLMRelationJudgment(
            supported=True,
            relation_type=relation_type,
            rationale=rationale,
            confidence=confidence,
            quote=quote,
        )
