"""Prompt Injection & Jailbreak Defense (Giai đoạn 5, Layer 6).

Detects and neutralizes:
- Direct prompt injections (instruction overrides, system prompt manipulation)
- Jailbreak patterns (DAN, developer mode, safety bypasses)
- Indirect prompt injection via document / web content / tool outputs
- Delimiter smuggling and fake role impersonation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from core.errors import BraindError
from core.logging import get_logger

logger = get_logger(__name__)


class PromptInjectionError(BraindError):
    """Raised when prompt injection or jailbreak is detected and blocked."""

    code = "prompt_injection_detected"
    public_message = "Input was blocked due to detected injection or override patterns."


class ThreatType(StrEnum):
    SYSTEM_OVERRIDE = "system_override"
    JAILBREAK = "jailbreak"
    PROMPT_LEAK = "prompt_leak"
    INDIRECT_INJECTION = "indirect_injection"
    DELIMITER_INJECTION = "delimiter_injection"
    ROLE_IMPERSONATION = "role_impersonation"


# Pattern definitions: (regex, ThreatType, severity weight)
INJECTION_RULES: list[tuple[re.Pattern, ThreatType, float]] = [
    # 1. System instruction override
    (
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+)?(?:previous|prior|above|earlier|system)\s+"
            r"(?:instructions|prompts|commands|rules|directives|constraints)\b",
            re.IGNORECASE,
        ),
        ThreatType.SYSTEM_OVERRIDE,
        0.95,
    ),
    (
        re.compile(
            r"\b(?:new\s+instruction|system\s+override|update\s+system\s+prompt|"
            r"from\s+now\s+on\s+ignore\s+all)\b",
            re.IGNORECASE,
        ),
        ThreatType.SYSTEM_OVERRIDE,
        0.90,
    ),
    # 2. Jailbreak / DAN / unrestricted modes
    (
        re.compile(
            r"\b(?:dan\s+mode|jailbreak|do\s+anything\s+now|developer\s+mode\s+v\d+|"
            r"unrestricted\s+ai|disable\s+(?:all\s+)?safety|pretend\s+you\s+have\s+no\s+rules|"
            r"act\s+as\s+an\s+unfiltered|bypass\s+(?:content\s+)?filters?)\b",
            re.IGNORECASE,
        ),
        ThreatType.JAILBREAK,
        0.95,
    ),
    # 3. Prompt leakage / exfiltration
    (
        re.compile(
            r"\b(?:repeat|print|reveal|output|display|show|leak)\s+"
            r"(?:your\s+)?(?:system\s+prompt|initial\s+instructions|system\s+instructions|"
            r"hidden\s+prompt|base\s+instructions)\b",
            re.IGNORECASE,
        ),
        ThreatType.PROMPT_LEAK,
        0.85,
    ),
    # 4. Indirect prompt injection (often hiding inside document/web extracts)
    (
        re.compile(
            r"(?:\[(?:instruction|ai|assistant|system)[^\]]*\]|"
            r"(?:instruction|note|directive)\s+to\s+(?:the\s+)?(?:ai|assistant|llm|agent):\s*"
            r"(?:ignore|execute|delete|exfiltrate|send|modify))\b",
            re.IGNORECASE,
        ),
        ThreatType.INDIRECT_INJECTION,
        0.90,
    ),
    (
        re.compile(
            r"(?:important\s+note\s+for\s+ai|assistant:\s*you\s+must\s+immediately|"
            r"hidden\s+instruction:\s*forget)\b",
            re.IGNORECASE,
        ),
        ThreatType.INDIRECT_INJECTION,
        0.85,
    ),
    # 5. Delimiter injection and tag smuggling
    (
        re.compile(
            r"(?:<\/?system>|<\/?instruction>|\[\/?system\]|\[\/?instruction\]|"
            r"<<SYS>>|<\/SYS>|```system)",
            re.IGNORECASE,
        ),
        ThreatType.DELIMITER_INJECTION,
        0.80,
    ),
    # 6. Role impersonation
    (
        re.compile(
            r"(?:^\s*(?:System|Admin|Root|Developer):\s*(?:You\s+are|Execute|Forget|Ignore))",
            re.IGNORECASE | re.MULTILINE,
        ),
        ThreatType.ROLE_IMPERSONATION,
        0.85,
    ),
]


@dataclass
class InjectionCheckResult:
    """Detailed result of an injection scan."""

    is_safe: bool
    is_injection: bool
    threat_type: ThreatType | None = None
    confidence: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    detail: str = ""
    sanitized_text: str = ""


class InjectionGuard:
    """Scans and neutralizes adversarial prompt injection in user and external inputs."""

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold

    def detect(self, text: str, *, source_type: str = "user") -> InjectionCheckResult:
        """Scan input text for prompt injection, jailbreaks, and overrides.

        Args:
            text: Text to analyze (user prompt, document chunk, web page text, tool output)
            source_type: 'user', 'document', 'external', 'memory'

        Returns:
            InjectionCheckResult with detection flags, threat type, confidence, and sanitized text.
        """
        if not text or not text.strip():
            return InjectionCheckResult(
                is_safe=True,
                is_injection=False,
                confidence=0.0,
                sanitized_text=text,
                detail="Empty text",
            )

        matched: list[tuple[re.Pattern, ThreatType, float, str]] = []

        for pattern, threat_type, weight in INJECTION_RULES:
            match = pattern.search(text)
            if match:
                matched.append((pattern, threat_type, weight, match.group(0)))

        if not matched:
            return InjectionCheckResult(
                is_safe=True,
                is_injection=False,
                confidence=0.0,
                sanitized_text=text,
                detail="Clean: no injection patterns detected",
            )

        # Pick the highest-severity match as primary threat
        matched.sort(key=lambda m: m[2], reverse=True)
        primary = matched[0]
        max_confidence = primary[2]
        threat_type = primary[1]
        pattern_matches = [m[3] for m in matched]

        is_injection = max_confidence >= self.confidence_threshold
        sanitized = self.neutralize(text) if is_injection else text

        detail = (
            f"Detected {threat_type.value} with confidence {max_confidence:.2f} "
            f"from source '{source_type}'. Matches: {pattern_matches[:3]}"
        )

        logger.warning(
            "injection_guard.detected",
            threat_type=threat_type.value,
            confidence=max_confidence,
            source_type=source_type,
            matches=pattern_matches[:3],
        )

        return InjectionCheckResult(
            is_safe=not is_injection,
            is_injection=is_injection,
            threat_type=threat_type,
            confidence=max_confidence,
            matched_patterns=pattern_matches,
            detail=detail,
            sanitized_text=sanitized,
        )

    def neutralize(self, text: str) -> str:
        """Neutralize malicious injection instructions in text by defanging them.

        Replaces detected instruction override triggers with inert text markers
        and wraps document content in untrusted content tags.
        """
        sanitized = text
        for pattern, threat_type, _ in INJECTION_RULES:
            def _repl(match: re.Match, tt: ThreatType = threat_type) -> str:
                return f"[DEFANGED_{tt.value.upper()}: {match.group(0)}]"

            sanitized = pattern.sub(_repl, sanitized)

        return sanitized

    def check_or_raise(self, text: str, *, source_type: str = "user") -> InjectionCheckResult:
        """Scan input; raise PromptInjectionError if dangerous injection is found."""
        result = self.detect(text, source_type=source_type)
        if result.is_injection:
            raise PromptInjectionError(
                f"Prompt injection blocked ({result.threat_type}): {result.detail}"
            )
        return result
