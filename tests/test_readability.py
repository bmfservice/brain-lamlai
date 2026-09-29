"""Readability extraction tests — trafilatura primary, bs4 fallback, gate contract."""

from __future__ import annotations

from core.quality import assess
from core.readability import extract_article, extract_article_html, extract_title


def test_article_fixture_extracts_prose(article_html):
    text = extract_article(article_html, url="https://example.test/article")
    assert "Verification page heading" in text or "verification paragraph" in text.lower()
    assert "Copyright notice" not in text  # footer chrome gone
    assert "Home About" not in text  # nav chrome gone


def test_trafilatura_output_passes_quality_gate(article_html):
    """The gate contract: extractor output must survive quality.assess."""
    text = extract_article(article_html, url="https://example.test/article")
    report = assess(text)
    assert report.prose_ratio > 0.5
    assert report.accepted


def test_nav_only_falls_back_or_extracts_small(nav_html):
    text = extract_article(nav_html, url="https://example.test/")
    # Either trafilatura finds the tiny main list or the bs4 fallback runs —
    # either way we get link labels, not an exception or page furniture soup.
    assert "Alpha" in text or "Site name" in text


def test_engine_setting_bs4_uses_fallback(article_html, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "readability_engine", "bs4")
    text = extract_article(article_html)
    assert "Copyright notice" not in text
    assert len(text) > 100  # fallback still extracts the article body


def test_han_gate_still_rejects():
    """quality.assess must still reject Han ideographs — gate untouched."""
    report = assess("This is Han: 中文文本内容很长很长" * 10)
    assert not report.accepted or report.reason in ("chinese", "not_prose")


def test_extract_article_html_keeps_headings(article_html):
    md = extract_article_html(article_html, url="https://example.test/article")
    assert "#" in md or "Verification" in md


def test_extract_title(article_html):
    assert "Stealth Web Layer" in extract_title(article_html)
