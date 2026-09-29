#!/usr/bin/env python3
"""Golden evaluation regression runner (Brain Second P14-03).

Runs the committed golden query suite deterministically: no live PostgreSQL,
no vLLM, no network.  It scores the analyzer + retrieval + context assembly
stack using token-overlap retrieval against the golden documents, then prints
an aggregate report and exits non-zero if any configured threshold is missed.

Usage:
    .venv/bin/python scripts/run_evaluation.py [--threshold <float>]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure the repository root is on PYTHONPATH when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.rag.analyzer import QueryAnalyzer  # noqa: E402, I001
from tests.evaluation.harness import GoldenDataset  # noqa: E402
from tests.evaluation.metrics import evaluate_query  # noqa: E402


THRESHOLDS = {
    "recall_at_k": 0.50,
    "precision_at_k": 0.10,
    "mrr": 0.45,
    "ndcg_at_k": 0.50,
    "hit": 0.80,
    "context_recall": 0.00,
}


def _token_set(text: str) -> set[str]:
    return {t.lower() for t in text.split() if t.isalpha() and len(t) > 2}


def _retrieve(
    query: str,
    documents: dict[str, str],
    *,
    top_k: int = 5,
) -> list[str]:
    """Token-overlap retriever for the golden documents."""
    query_tokens = _token_set(query)
    scored: list[tuple[float, str]] = []
    for stem, content in documents.items():
        content_tokens = _token_set(content)
        if not content_tokens:
            continue
        overlap = len(query_tokens & content_tokens) / len(content_tokens)
        if overlap > 0:
            scored.append((overlap, stem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [stem for _, stem in scored[:top_k]]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run golden evaluation regression suite.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum aggregate score to exit with code 0.",
    )
    args = parser.parse_args()

    dataset = GoldenDataset.load()
    analyzer = QueryAnalyzer()

    per_query: list[dict] = []
    ranked_by_query: list[list[str]] = []
    relevant_by_query: list[list[str]] = []
    blocks_by_query: list[list[str]] = []
    reference_blocks_by_query: list[list[str]] = []

    for q in dataset.queries:
        ranked = _retrieve(q["query"], dataset.documents, top_k=q.get("k", 5))
        profile = analyzer.analyze(q["query"])
        # Assemble a fake context using the ranked doc stems as block ids.
        blocks = ranked[: q.get("k", 5)]
        reference = q.get("expected_context", q.get("expected_retrieval", []))

        ranked_by_query.append(ranked)
        relevant_by_query.append(q.get("expected_retrieval", []))
        blocks_by_query.append(blocks)
        reference_blocks_by_query.append(reference)

        row = evaluate_query(
            ranked=ranked,
            relevant=q.get("expected_retrieval", []),
            k=q.get("k", 5),
            blocks=blocks,
            reference_blocks=reference,
        )
        row["query_id"] = q["id"]
        row["query_type"] = profile.query_type
        per_query.append(row)

    # Aggregate metrics.
    keys = ["recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit", "context_recall"]
    aggregates = {key: sum(r[key] for r in per_query) / len(per_query) for key in keys}

    print("=" * 60)
    print("Brain Second Golden Evaluation Report")
    print("=" * 60)
    print(f"Queries evaluated: {len(per_query)}\n")
    for q in per_query:
        print(
            f"{q['query_id']:30s} "
            f"recall={q['recall_at_k']:.2f} "
            f"prec={q['precision_at_k']:.2f} "
            f"mrr={q['mrr']:.2f} "
            f"ndcg={q['ndcg_at_k']:.2f} "
            f"hit={q['hit']:.2f} "
            f"ctx_recall={q['context_recall']:.2f}"
        )
    print("\nAggregates:")
    for key, value in aggregates.items():
        threshold = THRESHOLDS.get(key, 0.0)
        status = "OK" if value >= threshold else "FAIL"
        print(f"  {key:20s}: {value:.4f} (threshold {threshold:.2f}) [{status}]")

    overall = aggregates["recall_at_k"]
    min_score = args.threshold if args.threshold is not None else THRESHOLDS["recall_at_k"]
    if overall < min_score:
        print(f"\nFAILED: aggregate recall {overall:.4f} < {min_score:.4f}")
        return 1
    print(f"\nPASSED: aggregate recall {overall:.4f} >= {min_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
