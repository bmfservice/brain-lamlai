"""Giai đoạn 4 — anonymous session isolation tests. No network, no LLM."""

from __future__ import annotations

from core.session_context import (
    SESSION_COOKIE,
    SESSION_HEADER,
    resolve_session,
    session_filter_clause,
)

# ---------------------------------------------------------------------------
# Session resolution
# ---------------------------------------------------------------------------


class FakeRequest:
    def __init__(self, headers=None, cookies=None, client_host="203.0.113.5"):
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.client = type("C", (), {"host": client_host})()


def test_header_session_wins():
    sid = "12345678-1234-5678-1234-567812345678"
    ctx = resolve_session(FakeRequest(headers={SESSION_HEADER: sid}))
    assert ctx.session_id == sid
    assert not ctx.minted


def test_cookie_session_used_when_no_header():
    sid = "12345678-1234-5678-1234-567812345678"
    ctx = resolve_session(FakeRequest(cookies={SESSION_COOKIE: sid}))
    assert ctx.session_id == sid
    assert not ctx.minted


def test_malformed_session_mints_fresh():
    """A forged/malformed id never becomes a tenant identity."""
    ctx = resolve_session(FakeRequest(headers={SESSION_HEADER: "not-a-uuid"}))
    assert len(ctx.session_id) == 36  # fresh uuid4
    assert ctx.minted


def test_minted_sessions_are_unique():
    a = resolve_session(FakeRequest())
    b = resolve_session(FakeRequest())
    assert a.session_id != b.session_id


def test_fingerprint_stable_for_same_client():
    r = FakeRequest(headers={"user-agent": "TestUA", "accept-language": "vi-VN"})
    a = resolve_session(r)
    b = resolve_session(r)
    assert a.fingerprint_hash == b.fingerprint_hash
    assert len(a.fingerprint_hash) == 64  # sha256 hex


def test_locale_from_accept_language():
    ctx = resolve_session(FakeRequest(headers={"accept-language": "fr-FR,fr;q=0.9"}))
    assert ctx.locale == "fr-FR"  # never a hardcoded default


# ---------------------------------------------------------------------------
# Session filter clause (the retrieval gate primitive)
# ---------------------------------------------------------------------------


def test_filter_clause_shape():
    from sqlalchemy import Column
    from sqlalchemy import String as SaString
    from sqlalchemy.sql.elements import BooleanClauseList

    col = Column("session_id", SaString)
    clause = session_filter_clause(col, "abc")
    assert isinstance(clause, BooleanClauseList)  # OR of is-null and equality


# ---------------------------------------------------------------------------
# RAGService: three raw streams carry the session clause
# ---------------------------------------------------------------------------


def test_rag_streams_add_session_clause():
    from core.vllm import VLLMClient
    from services.rag.service import RAGService

    svc = RAGService(session_factory=None, vllm=VLLMClient(client=None))
    tag_clause, type_clause, params, session_clause = svc._filter_clauses(
        None, None, session_id="abc-123"
    )
    assert "d.session_id IS NULL OR d.session_id" in session_clause
    assert params["session_id"] == "abc-123"

    # No session → no clause (system context sees everything)
    _, _, params2, session_clause2 = svc._filter_clauses(None, None, session_id=None)
    assert session_clause2 == ""
    assert "session_id" not in params2
