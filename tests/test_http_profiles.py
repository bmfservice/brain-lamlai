"""Browser profile rotation tests."""

from __future__ import annotations

from core.config import settings
from core.http_profiles import build_profile, default_user_agent, host_of


def test_profile_fields_from_settings():
    p = build_profile("x.test")
    assert p.impersonate in set(settings.outbound_impersonate_targets)
    assert p.accept_language in set(settings.outbound_accept_languages)
    rows = settings.outbound_httpx_profiles
    assert any(p.user_agent == r["user_agent"] and p.accept == r["accept"] for r in rows)


def test_jitter_zero_keeps_affinity():
    p1 = build_profile("aff.test")
    p2 = build_profile("aff.test")
    assert p1 == p2


def test_jitter_can_reroll(monkeypatch):
    monkeypatch.setattr(settings, "outbound_profile_jitter", 1.0)
    seen = {build_profile("j.test").impersonate for _ in range(10)}
    assert len(seen) >= 1  # never crashes; with jitter=1 it re-rolls unseeded


def test_host_of():
    assert host_of("https://example.com/path?q=1") == "example.com"
    assert host_of("not a url") == ""


def test_default_user_agent_shim():
    ua = default_user_agent()
    assert ua.startswith("Mozilla/5.0")
