#!/usr/bin/env python3
"""Download vLLM models into braind/models/ (Phase 16 asset prep).

By default the vLLM/Transformers stack caches models under ``~/.cache/huggingface``.
This script forces the cache into ``braind/models/hf`` so the systemd units can
mount the project self-contained and backups capture the assets.

Usage:
    .venv/bin/python scripts/download_models.py [--skip-llm] [--skip-embed]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from any cwd and without the project installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _run_snapshot(model_name: str, models_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        print(f"huggingface_hub is required: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    cache_dir = models_dir / "hf"
    print(f"Downloading {model_name} into {cache_dir} ...")
    snapshot_download(model_name, cache_dir=str(cache_dir))
    print(f"Done: {model_name}")


def main() -> int:
    # Model identifiers come from settings (which reads .env) — never hard-coded.
    from core.config import settings

    parser = argparse.ArgumentParser(description="Download vLLM models into the project tree.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip chat model download.")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding model download.")
    parser.add_argument("--llm-model", default=settings.llm_model, help="Chat model identifier.")
    parser.add_argument(
        "--embed-model", default=settings.embed_model, help="Embedding model identifier."
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure HF cache lives inside the project.
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))

    if not args.skip_llm:
        _run_snapshot(args.llm_model, MODELS_DIR)
    if not args.skip_embed:
        _run_snapshot(args.embed_model, MODELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
