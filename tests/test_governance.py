"""Governance policy tests (Giai đoạn 5, B1 / Layer 6). No DB, no network.

Covers acceptance criterion #1: a code file is blocked at the gate by its *real*
mime/extension (never a hard-coded check function), while legitimate prose is
allowed. The PolicyEngine itself is DB-backed; here we drive FileImportPolicy
with a fake engine so the deny/allow precedence and mime-sniffing logic are
tested in isolation.
"""

from __future__ import annotations

import asyncio
import uuid

from services.governance.engine import ALLOW, DENY, PolicyDecision
from services.governance.file_policy import FileImportPolicy

# ---------------------------------------------------------------------------
# _sniff_mime — an extension is a claim, not a fact
# ---------------------------------------------------------------------------


def test_sniff_prefers_declared_content_type():
    mime = FileImportPolicy._sniff_mime("notes.txt", "text/markdown", b"")
    assert mime == "text/markdown"


def test_sniff_strips_charset_and_lowercases():
    mime = FileImportPolicy._sniff_mime("x", "Text/HTML; charset=UTF-8", b"")
    assert mime == "text/html"


def test_sniff_ignores_octet_stream_falls_back_to_extension():
    # application/octet-stream is "unknown", so the extension decides.
    mime = FileImportPolicy._sniff_mime("script.py", "application/octet-stream", b"")
    assert mime == "text/x-python"


def test_sniff_magic_bytes_shebang():
    mime = FileImportPolicy._sniff_mime("run", None, b"#!/bin/sh\necho hi\n")
    assert mime == "text/x-shellscript"


def test_sniff_magic_bytes_pdf():
    mime = FileImportPolicy._sniff_mime("doc", None, b"%PDF-1.7\n...")
    assert mime == "application/pdf"


def test_sniff_magic_bytes_executable():
    assert FileImportPolicy._sniff_mime("a", None, b"\x7fELF...") == "application/x-executable"
    assert FileImportPolicy._sniff_mime("b", None, b"MZ...") == "application/x-executable"


def test_sniff_magic_bytes_zip_and_gzip():
    assert FileImportPolicy._sniff_mime("a", None, b"PK\x03\x04...") == "application/zip"
    assert FileImportPolicy._sniff_mime("b", None, b"\x1f\x8b...") == "application/gzip"


def test_sniff_unknown_is_octet_stream():
    mime = FileImportPolicy._sniff_mime("mystery", None, b"\x00\x01\x02")
    assert mime == "application/octet-stream"


# ---------------------------------------------------------------------------
# Fake DB-backed engine
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Returns a scripted decision per (scope, pattern); else default-allow.

    Mirrors the real engine's contract: ``evaluate`` returns a PolicyDecision,
    ``audit`` records a non-DB-rule decision. ``default_allow`` (rule_id=None)
    reproduces 'no matching DB rule', which makes FileImportPolicy fall through
    to the built-in settings denylists.
    """

    def __init__(self, decisions=None, enabled=True):
        self._decisions = decisions or {}
        self.enabled = enabled
        self.calls: list[tuple[str, str]] = []
        self.audits: list[tuple[str, str]] = []  # (pattern, verdict) recorded

    async def evaluate(self, *, scope: str, pattern: str) -> PolicyDecision:
        self.calls.append((scope, pattern))
        if not self.enabled:
            return PolicyDecision(f"{scope}:{pattern}", ALLOW, None, "disabled")
        key = (scope, pattern)
        if key in self._decisions:
            verdict, rid, reason = self._decisions[key]
            return PolicyDecision(f"{scope}:{pattern}", verdict, rid, reason)
        return PolicyDecision(f"{scope}:{pattern}", ALLOW, None, "no matching rule (default allow)")

    async def audit(self, *, module, scope, pattern, verdict, reason=None):
        self.audits.append((pattern, verdict))


def _check(policy, *, filename, content_type=None, head=b""):
    return asyncio.run(
        policy.check(filename=filename, content_type=content_type, head_bytes=head)
    )


# ---------------------------------------------------------------------------
# check() — built-in defaults block code, allow prose
# ---------------------------------------------------------------------------


def test_python_file_blocked_by_extension():
    """Acceptance #1: a .py upload is denied before reaching the extractor."""
    engine = _FakeEngine()  # no DB rules → built-in defaults
    policy = FileImportPolicy(engine)
    result = _check(policy, filename="evil.py", content_type=None, head=b"import os\n")
    assert result.allowed is False
    assert result.verdict == DENY
    assert result.detected_mime == "text/x-python"
    # The built-in deny must be audited as DENY — not left as the engine's
    # earlier "default allow" rows (regression: a blocked file used to look
    # allowed in the audit trail).
    assert (".py", DENY) in engine.audits or ("text/x-python", DENY) in engine.audits
    assert all(v == DENY for _p, v in engine.audits)


def test_python_file_blocked_by_declared_mime():
    policy = FileImportPolicy(_FakeEngine())
    result = _check(policy, filename="data", content_type="text/x-python", head=b"")
    assert result.allowed is False
    assert result.verdict == DENY


def test_javascript_blocked():
    policy = FileImportPolicy(_FakeEngine())
    result = _check(policy, filename="app.js", content_type=None, head=b"")
    assert result.allowed is False
    assert result.detected_mime in ("application/javascript", "text/javascript")


def test_shellscript_blocked_via_magic_bytes():
    policy = FileImportPolicy(_FakeEngine())
    result = _check(policy, filename="deploy", content_type=None, head=b"#!/bin/bash\n")
    assert result.allowed is False
    assert result.detected_mime == "text/x-shellscript"


def test_sql_and_env_blocked_by_extension():
    policy = FileImportPolicy(_FakeEngine())
    assert _check(policy, filename="schema.sql").allowed is False
    assert _check(policy, filename=".env").allowed is False


def test_markdown_prose_allowed():
    engine = _FakeEngine()
    policy = FileImportPolicy(engine)
    result = _check(policy, filename="notes.md", content_type="text/markdown", head=b"# Title\n")
    assert result.allowed is True
    assert result.verdict == ALLOW
    # An allowed file is audited as ALLOW (the trail says what actually happened).
    assert ("text/markdown", ALLOW) in engine.audits


def test_plain_text_allowed():
    policy = FileImportPolicy(_FakeEngine())
    result = _check(policy, filename="readme.txt", content_type="text/plain", head=b"hello")
    assert result.allowed is True


# ---------------------------------------------------------------------------
# check() — DB rules take precedence over built-in defaults
# ---------------------------------------------------------------------------


def test_db_deny_rule_blocks_otherwise_allowed_type():
    """An explicit DB deny on a mime wins even though it is not in defaults."""
    rid = uuid.uuid4()
    engine = _FakeEngine(
        decisions={("file_import", "text/csv"): (DENY, rid, "csv disabled by policy")}
    )
    policy = FileImportPolicy(engine)
    result = _check(policy, filename="data.csv", content_type="text/csv", head=b"a,b\n")
    assert result.allowed is False
    assert result.verdict == DENY
    assert result.reason == "csv disabled by policy"


def test_db_allow_rule_rescues_denied_extension():
    """An explicit DB allow for a mime overrides the built-in extension deny —
    so an admin can permit, e.g., JSON config for a trusted source."""
    rid = uuid.uuid4()
    engine = _FakeEngine(
        decisions={("file_import", "application/json"): (ALLOW, rid, "json allowed for imports")}
    )
    policy = FileImportPolicy(engine)
    result = _check(policy, filename="config.json", content_type="application/json", head=b"{}")
    assert result.allowed is True
    assert result.verdict == ALLOW
    assert result.reason == "json allowed for imports"


def test_db_deny_on_extension_pattern():
    """A deny rule keyed on the extension (not the mime) also blocks."""
    rid = uuid.uuid4()
    engine = _FakeEngine(decisions={("file_import", ".md"): (DENY, rid, "no markdown")})
    policy = FileImportPolicy(engine)
    result = _check(policy, filename="notes.md", content_type="text/markdown", head=b"#")
    assert result.allowed is False
    # The mime check ran first (default-allow), then the extension rule denied.
    assert ("file_import", ".md") in engine.calls


def test_disabled_engine_allows_everything():
    """policy_enabled=False → the gate is open (no built-in defaults applied)."""
    policy = FileImportPolicy(_FakeEngine(enabled=False))
    result = _check(policy, filename="evil.py", content_type="text/x-python", head=b"")
    assert result.allowed is True
    assert result.verdict == ALLOW


# ---------------------------------------------------------------------------
# PolicyEngine.evaluate — disabled short-circuit (no DB touched)
# ---------------------------------------------------------------------------


def test_engine_disabled_returns_allow_without_session():
    """A disabled engine must not touch the DB at all."""
    from services.governance.engine import PolicyEngine

    def _boom():  # session_factory must never be called when disabled
        raise AssertionError("session_factory should not be used when disabled")

    engine = PolicyEngine(_boom, enabled=False)
    decision = asyncio.run(engine.evaluate(scope="file_import", pattern="text/x-python"))
    assert decision.verdict == ALLOW
    assert decision.rule_id is None
    assert "disabled" in decision.reason


def test_engine_fail_open_on_db_error():
    """A DB error must fail OPEN (logged) — policy is a gate, not the foundation."""
    from services.governance.engine import PolicyEngine

    def _factory():
        class _Ctx:
            async def __aenter__(self):
                raise RuntimeError("db down")

            async def __aexit__(self, *exc):
                return False

        return _Ctx()

    engine = PolicyEngine(_factory, enabled=True)
    decision = asyncio.run(engine.evaluate(scope="file_import", pattern="text/plain"))
    assert decision.verdict == ALLOW
    assert "fail-open" in decision.reason
