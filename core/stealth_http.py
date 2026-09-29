"""Stealth HTTP engine — curl_cffi with real Chrome TLS fingerprints (Giai đoạn 2).

Anti-bot services (Cloudflare, Akamai, Datadome) fingerprint the TLS handshake
(JA3/JA4), HTTP/2 SETTINGS frames and header order. A plain httpx client is
flagged no matter what User-Agent it sends. This engine delegates the wire
representation to curl_cffi's ``impersonate`` targets so the fingerprint is
coherent end to end.

SAFETY — redirect handling is the critical part:
    curl_cffi's ``allow_redirects=True`` follows redirects inside libcurl where
    the Python layer never sees intermediate hops, so a public→private redirect
    would land silently. This engine NEVER enables that flag: it loops manually,
    validating :func:`core.net.is_safe_url` on the entry URL and on EVERY hop,
    capping the chain at ``settings.outbound_max_redirects``.

The SSRF authority remains :mod:`core.net` — this module imports its checks and
never duplicates them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urljoin

from core.config import settings
from core.http_profiles import build_profile, host_of
from core.logging import get_logger

logger = get_logger(__name__)

_TEXTUAL_CONTENT_MARKERS = ("text/", "html", "xml", "json")
_REDIRECT_STATUS = {301, 302, 303, 307, 308}

try:  # imported lazily-friendly so tests can stub the module
    from curl_cffi.requests import AsyncSession
    from curl_cffi.requests.exceptions import HTTPError as CurlHTTPError
except ImportError:  # pragma: no cover - dependency is pinned in requirements
    AsyncSession = None  # type: ignore[assignment]
    CurlHTTPError = Exception  # type: ignore[assignment,misc]

_VALIDATED_TARGETS: set[str] | None = None


@dataclass(frozen=True)
class StealthFetchResult:
    """Outcome of one stealth fetch."""

    text: str
    final_url: str
    status: int
    elapsed_seconds: float


def _final_url(response, fallback: str) -> str:
    raw = str(getattr(response, "url", "") or "")
    return raw if raw.startswith("http") else fallback


def _valid_impersonate_targets() -> set[str]:
    """Lazily validate configured targets against the installed curl_cffi."""
    global _VALIDATED_TARGETS
    if _VALIDATED_TARGETS is None:
        try:
            from curl_cffi.requests.impersonate import BrowserTypeLiteral

            known = set(getattr(BrowserTypeLiteral, "__args__", ()) )
        except Exception:
            known = set()
        targets = {t for t in settings.outbound_impersonate_targets if not known or t in known}
        skipped = set(settings.outbound_impersonate_targets) - targets
        if skipped:
            logger.warning("stealth_http.invalid_impersonate_targets", skipped=sorted(skipped))
        _VALIDATED_TARGETS = targets or {"chrome136"}
    return _VALIDATED_TARGETS


async def stealth_fetch_text(
    url: str,
    *,
    max_bytes: int | None = None,
    timeout_seconds: float | None = None,
    require_textual: bool = True,
    max_redirects: int | None = None,
    session_factory=None,
) -> StealthFetchResult:
    """Fetch ``url`` as text through a Chrome-fingerprinted curl_cffi session.

    Args:
        url: absolute http(s) URL; must pass :func:`core.net.is_safe_url` and so
            must every redirect hop.
        max_bytes: response size cap (default ``outbound_max_response_bytes``).
        timeout_seconds: per-request timeout (default ``outbound_fetch_timeout_seconds``).
        require_textual: reject non-text content types.
        max_redirects: hop cap (default ``outbound_max_redirects``).
        session_factory: injectable async-session factory for tests; defaults to
            ``curl_cffi.requests.AsyncSession``.

    Raises:
        ValueError: the URL or any redirect hop is not a public http(s) target.
        RuntimeError: non-textual response or response exceeds ``max_bytes``.
        curl_cffi errors: transport failure; 4xx/5xx raise via raise_for_status.
    """
    if AsyncSession is None:  # pragma: no cover - pinned dependency missing
        raise RuntimeError("curl_cffi is not installed; stealth engine unavailable")

    # Late import: core.net delegates here, so importing it at module level
    # would be circular.
    from core.net import is_safe_url

    max_bytes = max_bytes or settings.outbound_max_response_bytes
    timeout_seconds = timeout_seconds or settings.outbound_fetch_timeout_seconds
    max_hops = max_redirects if max_redirects is not None else settings.outbound_max_redirects

    if not is_safe_url(url):
        raise ValueError(f"Unsafe URL blocked: {url}")

    profile = build_profile(host_of(url))
    factory = session_factory or AsyncSession
    started = time.monotonic()

    current = url
    status = 0
    response = None
    followed = 0
    while True:
        async with factory(impersonate=profile.impersonate) as session:
            response = await session.get(
                current,
                allow_redirects=False,  # SSRF: every hop must pass is_safe_url here
                timeout=timeout_seconds,
                headers=profile.stealth_kwargs()["headers"],
            )
        status = getattr(response, "status_code", 0) or 0
        if status in _REDIRECT_STATUS:
            location = response.headers.get("Location") or ""
            followed += 1
            if not location or followed > max_hops:
                break  # no target / too many hops: surface the redirect as-is
            current = urljoin(current, location)
            if not is_safe_url(current):
                raise ValueError(f"Redirected to unsafe URL: {current}")
            continue
        break

    if status in _REDIRECT_STATUS:
        # Redirect chain exhausted the hop cap — that is a hard failure, not a
        # page to hand back (the caller would store a 30x body).
        raise RuntimeError(f"Too many redirects (>{max_hops}) fetching: {url}")
    response.raise_for_status()
    ctype = response.headers.get("content-type", "")
    if require_textual and not any(m in ctype for m in _TEXTUAL_CONTENT_MARKERS):
        raise RuntimeError(f"Non-text content-type: {ctype}")
    body = response.content
    if len(body) > max_bytes:
        raise RuntimeError(f"Response exceeds {max_bytes} bytes")
    text = body.decode("utf-8", errors="replace")

    logger.info(
        "stealth_http.fetched",
        host=host_of(url),
        status=status,
        bytes=len(body),
        elapsed=round(time.monotonic() - started, 3),
        impersonate=profile.impersonate,
    )
    return StealthFetchResult(
        text=text,
        final_url=_final_url(response, current),
        status=status,
        elapsed_seconds=time.monotonic() - started,
    )
