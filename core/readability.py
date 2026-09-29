"""Readability-grade article extraction (Giai đoạn 2).

Primary extractor is trafilatura (text-density algorithm, multilingual) with
the project's proven BeautifulSoup ``article/main/body`` path as fallback.
Downstream, :func:`core.quality.assess` stays the final arbiter — extraction
improves, the quality gate is untouched.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript")

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency pinned in requirements
    trafilatura = None  # type: ignore[assignment]


def _bs4_article_text(html: str) -> str:
    """Strip site chrome and return the article/main/body text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_STRIP_TAGS):
        tag.decompose()
    container = soup.find("article") or soup.find("main") or soup.body or soup
    return container.get_text("\n", strip=True)


def extract_article(html: str, *, url: str | None = None, max_chars: int | None = None) -> str:
    """Return the main article text for ``html``.

    Trafilatura runs first unless ``settings.readability_engine`` says
    otherwise; its output below ``readability_min_chars`` (or empty) falls back
    to the BeautifulSoup path so a nav-heavy page never yields nothing.
    """
    text = _extract_with(settings.readability_engine, html, url=url, fmt="txt")
    if len(text) < settings.readability_min_chars:
        text = _bs4_article_text(html)
    if max_chars is not None:
        text = text[:max_chars]
    return text


def _extract_with(engine: str, html: str, *, url: str | None, fmt: str) -> str:
    """One trafilatura call with the bs4 fallback, honoring the engine setting."""
    if engine == "trafilatura" and trafilatura is not None:
        text = trafilatura.extract(
            html,
            url=url,
            output_format=fmt,
            favor_recall=settings.readability_favor_recall,
        )
        if text:
            return text
    return _bs4_article_text(html)


def extract_article_html(html: str, *, url: str | None = None) -> str:
    """Return the main article as markdown-ish HTML (headings preserved)."""
    return _extract_with(settings.readability_engine, html, url=url, fmt="markdown")


def extract_title(html: str) -> str:
    """Return the page <title>, or the first heading when absent."""
    soup = BeautifulSoup(html, "lxml")
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)
    heading = soup.find(["h1", "h2"])
    return heading.get_text(strip=True) if heading else ""
