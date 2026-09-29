"""Browser profile rotation for outbound fetching (Giai đoạn 2).

A global product must not present a single hardcoded fingerprint. Profiles are
read from settings only — no UA string or header set is ever assembled by hand
here: mismatches between a hand-built User-Agent and the real TLS fingerprint
are the most common bot-detection signal.

Two consumers:
- The curl_cffi engine (:mod:`core.stealth_http`) uses ``impersonate`` targets
  plus an Accept-Language; curl_cffi emits the coherent header set itself.
- The plain httpx engine (:func:`core.net.fetch_text_bounded`) uses the
  settings-provided ``outbound_httpx_profiles``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from urllib.parse import urlparse

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BrowserProfile:
    """One coherent browser identity for a request chain.

    ``impersonate`` is a curl_cffi target name (e.g. ``chrome136``); the httpx
    fields are used only by the plain engine. All values originate from
    settings — nothing here is hardcoded.
    """

    impersonate: str
    accept_language: str
    user_agent: str
    accept: str

    def stealth_kwargs(self) -> dict:
        """Keyword args for a curl_cffi AsyncSession/GET call."""
        return {
            "impersonate": self.impersonate,
            "headers": {"Accept-Language": self.accept_language},
        }

    def httpx_headers(self) -> dict:
        return {
            "User-Agent": self.user_agent,
            "Accept": self.accept,
            "Accept-Language": self.accept_language,
        }


def _pick(seq: list, rng: random.Random, what: str) -> object | None:
    if not seq:
        logger.warning("http_profiles.empty_pool", pool=what)
        return None
    return seq[rng.randrange(len(seq))]


def build_profile(host: str | None = None, *, rng_seed: str | None = None) -> BrowserProfile:
    """Pick one profile deterministically per host.

    Host affinity (``settings.outbound_profile_host_affinity``) seeds the RNG
    with the host so the same site always sees the same browser+language within
    a run — a real browser does not switch between pages of one site.
    ``outbound_profile_jitter`` re-rolls with the given probability.
    """
    seed_source = f"{rng_seed or ''}|{host or ''}"
    rng = random.Random(seed_source)
    if (
        settings.outbound_profile_host_affinity is False
        or (settings.outbound_profile_jitter and rng.random() < settings.outbound_profile_jitter)
    ):
        rng = random.Random()  # unseeded re-roll

    impersonate = _pick(
        list(settings.outbound_impersonate_targets), rng, "impersonate"
    ) or "chrome136"
    accept_language = _pick(
        list(settings.outbound_accept_languages), rng, "accept_languages"
    ) or "en-US,en;q=0.9"
    profile_row = _pick(list(settings.outbound_httpx_profiles), rng, "httpx_profiles") or {}
    user_agent = str(profile_row.get("user_agent") or "Mozilla/5.0")
    accept = str(profile_row.get("accept") or "*/*")
    return BrowserProfile(
        impersonate=str(impersonate),
        accept_language=str(accept_language),
        user_agent=user_agent,
        accept=accept,
    )


def default_user_agent() -> str:
    """Interim shim for callers that still want a single UA string.

    Returns the first plain-engine profile's UA so no caller needs the removed
    ``core.net.USER_AGENT`` constant. New code should use :func:`build_profile`.
    """
    return build_profile().user_agent


def host_of(url: str) -> str:
    return urlparse(url).hostname or ""
