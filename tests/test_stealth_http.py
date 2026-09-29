"""Stealth engine tests — SSRF, redirect loop, caps, rotation. No network."""

from __future__ import annotations

import pytest

from core.config import settings
from core.http_profiles import build_profile
from core.stealth_http import stealth_fetch_text
from tests.conftest import FakeResponse, FakeStealthSession


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Map any *.test host to a public IP so is_safe_url passes without DNS."""
    import socket as _socket

    real = _socket.getaddrinfo

    def fake_getaddrinfo(host, *args, **kwargs):
        if host.endswith(".test") or "example.com" in str(host):
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(_socket, "getaddrinfo", fake_getaddrinfo)

PUBLIC = "https://example.com/"


def _redirect(location: str) -> FakeResponse:
    return FakeResponse(302, headers={"content-type": "text/html", "Location": location})


def _targets() -> set[str]:
    return set(settings.outbound_impersonate_targets)


def _languages() -> set[str]:
    return set(settings.outbound_accept_languages)


def _factory(responses, calls):
    return lambda **kw: FakeStealthSession(responses, calls, **kw)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8010/api",
        "http://10.0.0.1/x",
        "http://169.254.169.254/meta-data",
        "http://0.0.0.0/",
    ],
)
async def test_private_url_blocked_before_request(url):
    calls: list = []
    with pytest.raises(ValueError, match="Unsafe URL blocked"):
        await stealth_fetch_text(url, session_factory=_factory({}, calls))
    assert calls == []  # no socket ever opened


@pytest.mark.asyncio
async def test_redirect_to_private_blocked():
    calls: list = []
    responses = {
        PUBLIC: FakeResponse(
            302,
            headers={"content-type": "text/html", "Location": "http://192.168.1.1/secret"},
        ),
    }
    with pytest.raises(ValueError, match="Redirected to unsafe URL"):
        await stealth_fetch_text(PUBLIC, session_factory=_factory(responses, calls))
    assert all(call["url"] != "http://192.168.1.1/secret" for call in calls)


@pytest.mark.asyncio
async def test_redirect_chain_all_public_ok():
    calls: list = []
    responses = {
        "https://a.test/start": _redirect("https://b.test/mid"),
        "https://b.test/mid": _redirect("https://c.test/end"),
        "https://c.test/end": FakeResponse(200, text="final body"),
    }
    result = await stealth_fetch_text(
        "https://a.test/start", session_factory=_factory(responses, calls)
    )
    assert result.status == 200
    assert result.final_url == "https://c.test/end"
    assert [c["url"] for c in calls] == [
        "https://a.test/start",
        "https://b.test/mid",
        "https://c.test/end",
    ]


@pytest.mark.asyncio
async def test_redirect_cap_enforced():
    responses = {
        f"https://h{i}.test/": _redirect(f"https://h{i+1}.test/") for i in range(10)
    }
    with pytest.raises(RuntimeError):
        await stealth_fetch_text(
            "https://h0.test/", max_redirects=5, session_factory=_factory(responses, [])
        )


@pytest.mark.asyncio
async def test_byte_cap():
    responses = {PUBLIC: FakeResponse(200, text="x" * (3 * 1024 * 1024))}
    with pytest.raises(RuntimeError, match="exceeds"):
        await stealth_fetch_text(
            PUBLIC, max_bytes=2_000_000, session_factory=_factory(responses, [])
        )


@pytest.mark.asyncio
async def test_non_textual_content_type():
    responses = {PUBLIC: FakeResponse(200, headers={"content-type": "application/octet-stream"})}
    with pytest.raises(RuntimeError, match="Non-text"):
        await stealth_fetch_text(PUBLIC, session_factory=_factory(responses, []))


@pytest.mark.asyncio
async def test_never_uses_libcurl_redirects():
    """The engine must send allow_redirects=False — SSRF depends on it."""
    calls: list = []
    responses = {PUBLIC: FakeResponse(200, text="ok")}
    await stealth_fetch_text(PUBLIC, session_factory=_factory(responses, calls))
    assert all(call["kw"].get("allow_redirects") is False for call in calls)


def test_header_rotation_varied_but_valid():
    targets = {build_profile(f"host{i}.test").impersonate for i in range(30)}
    assert len(targets) >= 2  # rotation actually rotates
    for i in range(20):
        profile = build_profile(f"host{i}.test")
        assert profile.impersonate in _targets()
        assert profile.accept_language in _languages()


def test_host_affinity_stable():
    a = build_profile("same.test")
    b = build_profile("same.test")
    assert a == b
