"""Deterministic input and prompt-injection guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ResumeInputQuality:
    """Deterministic quality summary for extracted resume text."""

    valid: bool
    category: str
    character_count: int
    token_count: int
    source_type: str
    garbled_text_detected: bool


@dataclass(frozen=True)
class InjectionFinding:
    """A category-only prompt-injection finding safe to expose in diagnostics."""

    category: str


class ResumeInputError(ValueError):
    """Raised when a resume cannot be safely evaluated."""

    def __init__(self, message: str, quality: ResumeInputQuality) -> None:
        super().__init__(message)
        self.quality = quality


_TOKEN_PATTERN = re.compile(r"\b[\w][\w'-]*\b", re.UNICODE)
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,60}"
            r"\b(instructions|rules|prompt)\b",
            re.I | re.S,
        ),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"\b(system message|system prompt|developer message|hidden prompt)\b",
            re.I,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(reveal|show|print|disclose)\b.{0,60}"
            r"\b(prompt|instructions|secret|api key)\b",
            re.I | re.S,
        ),
    ),
    (
        "rubric_override",
        re.compile(
            r"\b(override|change|bypass)\b.{0,60}"
            r"\b(rubric|rating|evaluation|criteria)\b",
            re.I | re.S,
        ),
    ),
    (
        "tool_request",
        re.compile(
            r"\b(call|use|invoke)\b.{0,60}"
            r"\b(tool|browser|web search|terminal)\b",
            re.I | re.S,
        ),
    ),
    (
        "forced_answer",
        re.compile(
            r"\b(return|answer|output)\b.{0,60}"
            r"\b(exactly|only)\b.{0,40}"
            r"\b(strong|partial|missing|insufficient)\b",
            re.I | re.S,
        ),
    ),
)


def validate_resume_text(text: str, *, source_type: str) -> ResumeInputQuality:
    """Validate extracted text without making any model or network call."""

    character_count = len(text.strip())
    token_count = len(_TOKEN_PATTERN.findall(text))
    if character_count < 20 or token_count < 5:
        return ResumeInputQuality(
            False,
            "empty_or_too_short",
            character_count,
            token_count,
            source_type,
            False,
        )

    printable_count = sum(char.isprintable() or char in "\n\r\t" for char in text)
    replacement_count = text.count("\ufffd")
    printable_ratio = printable_count / max(1, len(text))
    replacement_ratio = replacement_count / max(1, len(text))
    garbled = printable_ratio < 0.8 or replacement_ratio > 0.01
    return ResumeInputQuality(
        not garbled,
        "garbled_text" if garbled else "valid",
        character_count,
        token_count,
        source_type,
        garbled,
    )


def ensure_valid_resume_text(text: str, *, source_type: str) -> ResumeInputQuality:
    """Return quality metadata or raise a categorized input error."""

    quality = validate_resume_text(text, source_type=source_type)
    if not quality.valid:
        message = {
            "empty_or_too_short": "Resume contains too little extractable text",
            "garbled_text": "Resume text appears garbled or unreadable",
        }.get(quality.category, "Resume text failed input validation")
        raise ResumeInputError(message, quality)
    return quality


def detect_prompt_injection(text: str) -> tuple[InjectionFinding, ...]:
    """Find instruction-like content without returning matched resume text."""

    return tuple(
        InjectionFinding(category)
        for category, pattern in _INJECTION_PATTERNS
        if pattern.search(text)
    )
