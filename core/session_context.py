"""Anonymous per-request session context (Giai đoạn 4 — no auth, no login).

Every visitor gets a UUID session tied to a device fingerprint. There is no
account system on purpose: the session is the tenancy boundary, and its only
job is to keep one visitor's ingested knowledge and conversations from leaking
into another's retrieval.

Identity resolution order:
1. ``X-Session-ID`` header (client-persisted UUID — survives across devices
   the user chooses to share).
2. ``braind_session`` cookie (first-touch, so plain browser clients keep their
   session without any JS change).
3. A fresh UUID minted for this request (and set on the response cookie).

The device fingerprint is a SHA-256 of the coarse client attributes the
request already carries (IP, user-agent, language) — it is recorded for
audit/ops, never used to deny access.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

SESSION_COOKIE = "braind_session"
SESSION_HEADER = "X-Session-ID"


@dataclass(frozen=True)
class SessionContext:
    """The tenancy identity of one request."""

    session_id: str
    fingerprint_hash: str
    locale: str  # best-effort client language (never a hardcoded default)
    minted: bool  # True when this request created the session


def _fingerprint(ip: str, user_agent: str, locale: str) -> str:
    raw = f"{ip}|{user_agent}|{locale}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()


def resolve_session(request) -> SessionContext:
    """Build the SessionContext for an inbound request.

    Reads (in order) the ``X-Session-ID`` header, the ``braind_session``
    cookie, then mints a new UUID. Validates any supplied id — a malformed or
    forged-looking value simply becomes a fresh session.
    """
    locale = (request.headers.get("accept-language") or "").split(",")[0].strip()
    supplied = (
        request.headers.get(SESSION_HEADER)
        or request.cookies.get(SESSION_COOKIE)
        or ""
    ).strip()
    try:
        session_id = str(uuid.UUID(supplied))
        minted = False
    except (ValueError, AttributeError):
        session_id = str(uuid.uuid4())
        minted = True

    client_host = request.client.host if request.client else ""
    return SessionContext(
        session_id=session_id,
        fingerprint_hash=_fingerprint(
            client_host, request.headers.get("user-agent", ""), locale
        ),
        locale=locale,
        minted=minted,
    )


def session_filter_clause(column, session_id: str):
    """SQLAlchemy clause scoping ``column`` to one session or global rows.

    ``column == :sid OR column IS NULL`` — the disjunction is the whole point:
    global knowledge stays shared across sessions.
    """
    from sqlalchemy import or_

    return or_(column.is_(None), column == session_id)
