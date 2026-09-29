"""File-import governance (Giai đoạn 5, Layer 6).

Blocks code/config files before they ever reach the structural ingest.
Decisions are based on the *real* mime type reported by the upload plus
content sniffing — an extension is a claim, not a fact.

Evaluation order:
1. DB policy rules (scope ``file_import``) via :class:`PolicyEngine`.
2. Built-in deny defaults from settings (deny mimes + deny extensions) —
   these apply when no DB rule covers the file, so a fresh deployment is
   protected without seeding rules.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from core.config import settings
from core.logging import get_logger
from services.governance.engine import ALLOW, DENY, PolicyEngine

logger = get_logger(__name__)


@dataclass(frozen=True)
class FilePolicyResult:
    """Outcome of a file-import policy check."""

    allowed: bool
    verdict: str  # allow | deny | ask_first
    reason: str
    detected_mime: str


class FileImportPolicy:
    """Gate every upload against governance rules before extraction."""

    def __init__(self, engine: PolicyEngine) -> None:
        self.engine = engine

    async def check(
        self,
        *,
        filename: str,
        content_type: str | None,
        head_bytes: bytes,
    ) -> FilePolicyResult:
        """Evaluate one file. ``head_bytes`` is the first few KB of the file
        for content sniffing — never trust the client's declared type alone."""
        detected_mime = self._sniff_mime(filename, content_type, head_bytes)
        # policy_enabled is the master switch: when off, the whole gate is open
        # (no DB rules AND no built-in defaults), so an operator can disable
        # governance in one place without editing the denylists.
        if not self.engine.enabled:
            return FilePolicyResult(True, ALLOW, "policy engine disabled", detected_mime)
        extension = self._extension(filename)

        # 1. DB rules first — they override built-in defaults in both
        #    directions (an explicit allow rescues a denied extension).
        decision = await self.engine.evaluate(scope="file_import", pattern=detected_mime)
        if decision.verdict == DENY:
            return FilePolicyResult(False, DENY, decision.reason, detected_mime)
        if decision.verdict == ALLOW and decision.rule_id is not None:
            # Explicit allow rule for this mime.
            return FilePolicyResult(True, ALLOW, decision.reason, detected_mime)

        # 2. Extension denylist (DB rule may target extensions too).
        if extension:
            ext_decision = await self.engine.evaluate(scope="file_import", pattern=extension)
            if ext_decision.verdict == DENY:
                return FilePolicyResult(False, DENY, ext_decision.reason, detected_mime)

        # 3. Built-in defaults from settings (config, not hard-code). These are
        #    real gate decisions but happen outside the engine's DB-rule path, so
        #    audit them explicitly — otherwise the trail would only show the
        #    engine's "default allow" rows and contradict the 403 returned.
        if detected_mime in settings.policy_file_deny_mimes:
            reason = f"mime {detected_mime} is in the configured code/config denylist"
            await self.engine.audit(
                module="file_policy",
                scope="file_import",
                pattern=detected_mime,
                verdict=DENY,
                reason=reason,
            )
            return FilePolicyResult(False, DENY, reason, detected_mime)
        if extension and extension in settings.policy_file_deny_extensions:
            reason = f"extension {extension} is in the configured code/config denylist"
            await self.engine.audit(
                module="file_policy",
                scope="file_import",
                pattern=extension,
                verdict=DENY,
                reason=reason,
            )
            return FilePolicyResult(False, DENY, reason, detected_mime)

        await self.engine.audit(
            module="file_policy",
            scope="file_import",
            pattern=detected_mime,
            verdict=ALLOW,
            reason="no policy match",
        )
        return FilePolicyResult(True, ALLOW, "no policy match", detected_mime)

    @staticmethod
    def _extension(filename: str) -> str:
        """Return the lowercase extension, treating a leading-dot name as one.

        ``Path('.env').suffix`` is ``''`` — pathlib reads the leading dot as a
        hidden-file marker, not an extension. But ``.env`` is exactly the kind of
        secret-bearing file the denylist must catch, so a name that *is* its own
        extension (``.env``, ``.gitignore``) is returned whole.
        """
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
        name = Path(filename).name.lower()
        # A bare dotfile: leading dot, no further separator → the name is the ext.
        if name.startswith(".") and "." not in name[1:]:
            return name
        return ""

    @staticmethod
    def _sniff_mime(filename: str, content_type: str | None, head: bytes) -> str:
        """Best-effort real mime: client header → extension guess → content.

        Content sniffing here is deliberately cheap (magic-byte prefixes for
        the formats that matter); a full libmagic dependency is unnecessary
        because the extractor downstream re-validates anything it cannot parse.
        """
        if content_type and content_type not in ("application/octet-stream", ""):
            return content_type.split(";")[0].strip().lower()
        guessed = mimetypes.guess_type(filename)[0]
        if guessed:
            return guessed
        # Magic bytes: scripts and archives reveal themselves in the first KB.
        if head.startswith(b"#!"):
            return "text/x-shellscript"
        if head.startswith(b"PK\x03\x04"):
            return "application/zip"
        if head.startswith(b"\x1f\x8b"):
            return "application/gzip"
        if head.startswith(b"%PDF"):
            return "application/pdf"
        if head.startswith(b"\x7fELF") or head.startswith(b"MZ"):
            return "application/x-executable"
        return "application/octet-stream"
