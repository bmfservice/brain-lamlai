"""Braind Core - Outbound network safety.

Single source of truth for SSRF protection and bounded HTTP fetching. Every
outbound request made on behalf of a user, an agent tool, or a scheduled job
must go through :func:`is_safe_url` and :func:`fetch_text_bounded` so that
private, loopback, link-local and reserved targets stay unreachable.

Two engines live behind :func:`fetch_text_bounded` (Giai đoạn 2):
- stealth (default): :mod:`core.stealth_http` — curl_cffi with real Chrome TLS
  fingerprints for anti-bot targets.
- plain: httpx with settings-provided browser profiles (RSS XML and friends).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

from core.config import settings
from core.http_profiles import build_profile
from core.logging import get_logger
from core.stealth_http import stealth_fetch_text

logger = get_logger(__name__)

_TEXTUAL_CONTENT_MARKERS = ("text/", "html", "xml", "json")


def is_public_host(host: str) -> bool:
    """Return True only if ``host`` resolves entirely to public IP addresses.

    Rejects loopback, private, link-local, reserved, multicast and unspecified
    targets, for both literal IPs and every address a hostname resolves to.
    """
    if not host:
        return False
    candidates: list[str] = []
    try:
        ipaddress.ip_address(host)
        candidates.append(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        candidates = [info[4][0] for info in infos]
    if not candidates:
        return False
    for addr in candidates:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def is_safe_url(url: str) -> bool:
    """Validate that ``url`` is an absolute, public http(s) target."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    return is_public_host(parsed.hostname or "")


async def fetch_text_bounded(
    url: str,
    *,
    max_bytes: int | None = None,
    timeout_seconds: float | None = None,
    require_textual: bool = True,
    stealth: bool | None = None,
) -> tuple[str, str]:
    """Fetch a URL as text with SSRF, size, timeout and content-type limits.

    Returns ``(text, final_url)``. With ``stealth`` unset, the engine comes
    from ``settings.outbound_stealth_default``; pass ``True`` for the
    curl_cffi Chrome-fingerprint engine (anti-bot targets) or ``False`` to
    force the plain httpx engine (RSS XML and friends). Both engines re-validate
    the URL after every redirect hop so a redirect chain cannot land on a
    private host.

    Raises:
        ValueError: the URL or a redirect hop is not a public http(s) target.
        RuntimeError: the response is non-textual or exceeds ``max_bytes``.
        httpx.HTTPError / curl_cffi errors: transport failure or non-2xx status.
    """
    max_bytes = max_bytes or settings.outbound_max_response_bytes
    timeout_seconds = timeout_seconds or settings.outbound_fetch_timeout_seconds

    if not is_safe_url(url):
        raise ValueError(f"Unsafe URL blocked: {url}")

    if stealth is None:
        stealth = settings.outbound_stealth_default

    if stealth:
        result = await stealth_fetch_text(
            url,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
            require_textual=require_textual,
        )
        return result.text, result.final_url

    profile = build_profile(urlparse(url).hostname or "").httpx_headers()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout_seconds),
        follow_redirects=True,
        max_redirects=settings.outbound_max_redirects,
        headers=profile,
    ) as client, client.stream("GET", url) as resp:
        resp.raise_for_status()
        if not is_safe_url(str(resp.url)):
            raise ValueError(f"Redirected to unsafe URL: {resp.url}")
        ctype = resp.headers.get("content-type", "")
        if require_textual and not any(m in ctype for m in _TEXTUAL_CONTENT_MARKERS):
            raise RuntimeError(f"Non-text content-type: {ctype}")
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise RuntimeError(f"Response exceeds {max_bytes} bytes")
            chunks.append(chunk)
        final_url = str(resp.url)

    return b"".join(chunks).decode("utf-8", errors="replace"), final_url
