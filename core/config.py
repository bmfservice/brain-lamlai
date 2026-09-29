"""Braind Core - Configuration module.

Central configuration for the Braind Server system, loaded from environment
variables or a .env file using pydantic-settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from env / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Core ---
    app_name: str = "Braind Server"
    debug: bool = False
    data_dir: Path = Path("./data")

    # --- PostgreSQL ---
    database_url: str = "postgresql+asyncpg://braind:braind_dev_2026@localhost:5432/braind"

    # --- vLLM (separate local GPU servers for chat and embeddings) ---
    vllm_llm_base_url: str = "http://127.0.0.1:8001"
    vllm_embed_base_url: str = "http://127.0.0.1:8002"
    llm_model: str = "google/gemma-4-12B-it-qat-w4a16-ct"
    embed_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    embedding_dimensions: int = 1024
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    # Two-card split: chat and embedding each own a GPU so heavy embed batches
    # never contend with chat inference. Indices and VRAM fractions are read
    # from .env — nothing about the hardware layout is hard-coded here.
    vllm_llm_gpu: int = 0
    vllm_embed_gpu: int = 1
    vllm_llm_gpu_mem_util: float = 0.62
    vllm_embed_gpu_mem_util: float = 0.12
    vllm_llm_max_model_len: int = 16384
    vllm_embed_max_model_len: int = 8192
    vllm_tool_call_parser: str = "gemma4"

    # --- Vault (Obsidian) ---
    vault_dir: Path = Path("./vault")
    watch_interval_seconds: int = 30

    # --- Obsidian display/Canvas export (ingest §3) ---
    # Generated notes live in a subdirectory of the vault so Obsidian can open
    # them; a leading dot would hide them from graph view and Canvas, which is
    # the entire point of the layer. The vault scanner skips this directory by
    # name, and every generated file also carries a `braind_generated`
    # frontmatter marker so a note dragged elsewhere is still never re-ingested.
    obsidian_export_enabled: bool = True
    obsidian_export_dirname: str = "braind"

    # --- RAG ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    # A block shorter than this is a fragment (a table cell, a bare label) and
    # is merged with its neighbours instead of becoming its own chunk.
    chunk_min_chars: int = 200

    # --- LLM topic-shift segmentation (unstructured documents, ingest §2) ---
    # A structure-aware node longer than this has no internal heading to cut on,
    # so its boundaries are decided by an LLM reading the sentences, never by
    # embedding cosine alone.
    segment_enabled: bool = True
    segment_trigger_chars: int = 2000
    # Fewer sentences than this is already one topic; more than this no longer
    # fits a 7B prompt on a single 12GB GPU, so it falls back to sentence packing.
    segment_min_sentences: int = 4
    segment_max_sentences: int = 120
    top_k: int = 8
    score_threshold: float = 0.4

    # --- Writer ---
    writer_max_variants: int = 3
    writer_web_check_enabled: bool = True
    writer_web_max_results: int = 5
    writer_web_similarity_threshold: float = 0.65
    writer_web_fetch_timeout_seconds: float = 12.0
    writer_web_max_page_chars: int = 30000
    writer_web_max_response_bytes: int = 1_500_000

    # --- Scheduler (thay thế n8n) ---
    scheduler_enabled: bool = True
    rss_refresh_minutes: int = 60
    rss_feeds: list[dict] = []
    digest_hour: int = 7
    scheduler_timezone: str = "Asia/Ho_Chi_Minh"

    # --- 00:00 Daily Maintenance (configurable, never hard-coded in logic) ---
    maintenance_time: str = "00:00"
    maintenance_timezone: str = "Asia/Ho_Chi_Minh"
    maintenance_lock_timeout_seconds: int = 300

    # --- Knowledge graph traversal bounds ---
    graph_max_hops: int = 2
    graph_max_nodes: int = 200
    graph_max_edges_per_node: int = 50

    # --- Retrieval evaluation gate ---
    eval_golden_dir: Path = Path("./tests/evaluation/golden")
    eval_recall_gate: float = 0.5        # min Recall@10 vs golden baseline
    eval_ndcg_gate: float = 0.5          # min NDCG@10 vs golden baseline
    eval_mrr_gate: float = 0.4           # min MRR vs golden baseline

    # --- Deep Learning (spider crawl) ---
    learn_concurrency: int = 4
    learn_max_depth: int = 2
    learn_per_level: int = 5
    learn_max_total_pages: int = 40
    learn_links_per_page: int = 5
    learn_hard_timeout_sec: float = 240.0
    learn_min_link_len: int = 40
    learn_max_log: int = 400

    # --- Site crawl (learn every page of one website) ---
    # Depth is the BFS tier limit; the page cap and wall-clock timeout are what
    # actually bound the run, since depth 10 on a real site is unbounded.
    crawl_max_depth: int = 10
    crawl_max_pages: int = 300
    crawl_links_per_page: int = 20
    crawl_concurrency: int = 4
    crawl_hard_timeout_sec: float = 1800.0
    crawl_min_content_chars: int = 200

    # --- Server ---
    # Local-first deployment: bind loopback unless an operator opts into LAN
    # exposure. There is no authentication layer in front of the API.
    host: str = "127.0.0.1"
    port: int = 8000
    cors_allow_origins: list[str] = ["http://127.0.0.1:8000", "http://localhost:8000"]

    # --- Outbound HTTP safety (shared by web tool, RSS, crawler) ---
    outbound_fetch_timeout_seconds: float = 20.0
    outbound_max_response_bytes: int = 2_000_000

    # --- Stealth web layer (Giai đoạn 2) ---
    # Default engine for fetch_text_bounded: curl_cffi (TLS-fingerprint
    # impersonation) when true, plain httpx when false. Override per call.
    outbound_stealth_default: bool = True
    # curl_cffi impersonation targets. Each emits a coherent header set matching
    # its TLS/JA3/H2 fingerprint — never assemble UA + sec-ch-ua by hand, that
    # mismatch is an instant bot flag. Invalid entries are skipped with a
    # warning at first use.
    outbound_impersonate_targets: list[str] = ["chrome136", "chrome133a", "chrome131"]
    # Accept-Language pool for the rotation. A global product must read pages
    # the way its users do — no single hardcoded locale.
    outbound_accept_languages: list[str] = ["en-US,en;q=0.9", "vi-VN,vi;q=0.9,en;q=0.8"]
    # Plain (httpx) engine headers, e.g. for RSS XML endpoints that never check
    # TLS fingerprints. Replaces the old hardcoded USER_AGENT constant.
    outbound_httpx_profiles: list[dict] = [
        {
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/136.0 Safari/537.36"
            ),
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
    ]
    # Same host always sees the same browser+language within a run — a real
    # browser does not switch between pages.
    outbound_profile_host_affinity: bool = True
    # 0..1 chance to re-roll the profile per request despite host affinity.
    outbound_profile_jitter: float = 0.0
    # Cap applies to BOTH engines (httpx max_redirects and the stealth manual
    # loop) so a redirect chain can never exceed the configured bound.
    outbound_max_redirects: int = 5
    # Retry policy for transient outbound failures (429/5xx/timeout/connect).
    # Retries happen AFTER the first attempt; cap 2 to bound crawl fan-out.
    outbound_retry_attempts: int = 2
    outbound_retry_base_delay: float = 1.0
    outbound_retry_max_delay: float = 8.0

    # --- Readability extraction (Giai đoạn 2) ---
    # "trafilatura" is the primary extractor; "bs4" is the escape hatch.
    readability_engine: str = "trafilatura"
    readability_favor_recall: bool = True
    # Below this many extracted characters the bs4 fallback runs — trafilatura
    # occasionally returns near-empty output on nav-heavy SPA pages.
    readability_min_chars: int = 0

    # --- Evidence-based graph linking (Giai đoạn 3) ---
    # LLM-judged relation verification: the model rules on whether the quoted
    # sentence actually asserts the candidate relation, refining the type and
    # supplying the rationale stored on the edge. Failures degrade to
    # lexical-only candidates — an LLM outage never fails an ingest.
    relation_llm_enabled: bool = True
    # Model may judge at most this many candidates per ingest document (the
    # co-occurrence window can propose many pairs; judging them all would turn
    # one document into an unbounded chat-model workload).
    relation_llm_max_pairs: int = 12
    # Minimum signal strength for a candidate to be worth an LLM judgment
    # (0 disables the floor; POSSIBLE_RELATION pairs are the main target).
    relation_llm_min_structural: float = 0.0

    # --- Governance policy engine (Giai đoạn 5, Layer 6) ---
    # DB-backed declarative rules (allow/deny/ask_first) evaluated BEFORE any
    # mutating action. Deny beats allow at equal priority; human overrides sit
    # above everything. Rules are edited via REST — no restart needed.
    policy_enabled: bool = True
    # File-import policy: reject code/config files by their real mime type
    # (never trust the extension) plus a configurable extension denylist.
    # These are the *defaults*; DB rules take precedence when present.
    policy_file_deny_mimes: list[str] = [
        "text/x-python",
        "text/x-script.python",
        "application/javascript",
        "text/javascript",
        "text/x-java",
        "text/x-c",
        "text/x-c++",
        "application/x-sh",
        "application/json",
        "application/xml",
        "text/x-shellscript",
    ]
    policy_file_deny_extensions: list[str] = [
        ".py", ".pyw", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".java", ".kt",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh",
        ".bash", ".ps1", ".bat", ".cmd", ".sql", ".r", ".swift", ".scala",
        ".toml", ".cfg", ".ini", ".env", ".lock", ".ipynb", ".class", ".jar",
        ".whl", ".exe", ".dll", ".so", ".bin",
    ]

    # --- Consolidated classifier (Giai đoạn 5, Layer 4) ---
    # A small HistGradientBoosting model distilled from the LLM's past relation
    # decisions (complementary learning systems). Handles familiar cases
    # without any LLM call; unfamiliar/low-confidence cases escalate.
    consolidation_enabled: bool = True
    # Model answers below this confidence escalate to the LLM instead of being
    # trusted. High default: a wrong cheap answer is worse than a slow one.
    consolidation_confidence_threshold: float = 0.85
    # Directory holding the joblib model artifact (survives restarts).
    consolidation_model_dir: Path = Path("./data")
    # Retrain only when at least this many new decisions accumulated since the
    # last training run — consolidation is a background job, not per-request.
    consolidation_min_new_samples: int = 20

    # --- Consolidation job (Giai đoạn 5, Layer 7) ---
    # Nightly background pass: retrain the classifier from accumulated
    # decisions, apply edge-weight decay/reuse, flag stale low-confidence
    # rules. Scheduled after the 00:00 maintenance window.
    consolidation_time: str = "00:30"
    consolidation_timezone: str = "Asia/Ho_Chi_Minh"
    # Edge-weight decay: an edge unused for `half_life_days` halves its
    # reuse bonus. Reused edges (log-weighted) resist decay.
    consolidation_weight_half_life_days: float = 90.0
    # Self-monitoring (B7): an active edge whose decayed weight falls below
    # this floor is flagged lifecycle="review" — stale or weakly-supported
    # relations surface for a human instead of silently ranking low forever.
    consolidation_review_weight_floor: float = 0.05

    # --- Settings foundation (Brain Second Phase 1) ---
    # Seed the declarative catalog into setting_definitions/setting_values at
    # startup. Idempotent: a re-seed never overwrites a value an operator edited.
    settings_seed_enabled: bool = True
    # Shared-secret guard for the settings REST API (Phase 1c). Empty = the
    # settings endpoints are not exposed yet, which is the case until 1c.
    settings_admin_token: str = ""

    # --- Uploads ---
    upload_max_bytes: int = 25_000_000

    # --- Chat / agent grounding ---
    # Inject the current date/time into the agent system prompt.
    chat_inject_datetime: bool = True
    # Always retrieve from the knowledge base before the agent loop, so brains
    # is consulted even when the model does not call knowledge_retrieve itself.
    chat_always_retrieve: bool = True
    chat_pre_retrieve_top_k: int = 6
    # Drop pre-retrieved chunks below this fused RRF score to avoid noise
    # injection. RRF sums 1/(60+rank) per component, so scores sit near 0.05.
    chat_pre_retrieve_min_score: float = 0.01
    # Persist finished chat turns back into the knowledge base as source_type "chat".
    chat_learn_enabled: bool = True
    chat_learn_min_answer_chars: int = 200

    # --- Ingestion upgrade ---
    # Language detection: below this confidence the classifier calls the LLM.
    language_heuristic_confidence_threshold: float = 0.7
    # Freshness classification ("heuristic" default; "llm"/"hybrid" optional).
    freshness_method: str = "heuristic"
    freshness_taxonomy: list[dict] = [
        {"name": "STATIC", "table": "semantic_units",
         "llm_probe": False, "description": "Evergreen reference material"},
        {"name": "LONG_LIVED", "table": "semantic_units",
         "llm_probe": False, "description": "Changes slowly (years)"},
        {"name": "TIME_SENSITIVE", "table": "semantic_units",
         "llm_probe": True, "description": "Changes quickly; search-validation target"},
        {"name": "REALTIME", "table": "semantic_units",
         "llm_probe": True, "description": "Live/current; highest validation priority"},
    ]
    # Search validation gating (Phase 4) — hard outbound cap per ingest.
    search_validation_enabled: bool = True
    search_validation_max_outbound: int = 3
    search_validation_max_results: int = 3
    search_validation_fetch_page: bool = True
    search_validation_verify_threshold: float = 0.9   # input ≈ web -> VERIFIED
    search_validation_conflict_threshold: float = 0.5  # below -> CONFLICTED/DIVERGED


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
