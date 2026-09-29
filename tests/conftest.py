"""Shared fixtures for the Giai đoạn 2 stealth-layer tests. No real network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeResponse:
    """Minimal response surface shared by the stealth and httpx engines."""

    def __init__(self, status_code=200, text="", content=None, headers=None, url="https://x.test/"):
        self.status_code = status_code
        self.text = text
        self.content = content if content is not None else text.encode()
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError(f"HTTP {self.status_code}")
            error.status_code = self.status_code
            error.response = self
            raise error


class FakeStealthSession:
    """Curl_cffi AsyncSession stand-in driven by a per-URL response map."""

    last_profile: dict | None = None

    def __init__(self, responses=None, calls=None, impersonate=None):
        self._responses = responses or {}
        self._calls = calls if calls is not None else []
        self._impersonate = impersonate

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, allow_redirects=True, timeout=None, headers=None, **kw):
        self._calls.append(
            {"url": url, "headers": headers, "kw": {**kw, "allow_redirects": allow_redirects}}
        )
        FakeStealthSession.last_profile = {"impersonate": self._impersonate, "headers": headers}
        resp = self._responses[url]
        if callable(resp):
            resp = resp()
        resp.url = url  # mirror curl_cffi: response.url is the requested URL
        return resp


@pytest.fixture
def article_html():
    path = Path(__file__).parent / "fixtures" / "article_page.html"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def nav_html():
    path = Path(__file__).parent / "fixtures" / "nav_only.html"
    return path.read_text(encoding="utf-8")
