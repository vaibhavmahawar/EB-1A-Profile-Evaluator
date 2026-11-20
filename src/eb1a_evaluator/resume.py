"""Resume file ingestion and deterministic evidence passage creation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

from .guardrails import (
    InjectionFinding,
    ResumeInputQuality,
    detect_prompt_injection,
    ensure_valid_resume_text,
)


@dataclass(frozen=True)
class ResumePassage:
    """A stable, traceable passage extracted from a resume."""

    passage_id: str
    text: str
    source_name: str
    passage_index: int
    content_hash: str


@dataclass(frozen=True)
class ResumeDocument:
    """An extracted resume and its evidence passages."""

    source_name: str
    source_type: Literal["pdf", "txt"]
    raw_text: str
    passages: tuple[ResumePassage, ...]
    input_quality: ResumeInputQuality | None = None
    prompt_injection_findings: tuple[InjectionFinding, ...] = ()


def normalize_resume_text(text: str) -> str:
    """Normalize layout noise while preserving paragraph boundaries."""

    text = text.replace("\u00a0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_resume_text(path: Path) -> tuple[Literal["pdf", "txt"], str]:
    """Extract text from a supported resume file."""

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return "txt", normalize_resume_text(path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "pdf", normalize_resume_text("\n\n".join(pages))
    raise ValueError("Unsupported resume type; expected .pdf or .txt")


def make_passages(
    text: str,
    *,
    source_name: str,
    target_words: int = 200,
    overlap_words: int = 30,
) -> tuple[ResumePassage, ...]:
    """Split resume text into deterministic overlapping evidence passages."""

    if target_words < 1 or overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("target_words must be positive and overlap_words must be smaller")
    words = normalize_resume_text(text).split()
    if not words:
        raise ValueError("Resume contains no extractable text")
    passages: list[ResumePassage] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        passage_text = " ".join(words[start:end])
        digest = hashlib.sha256(passage_text.encode("utf-8")).hexdigest()[:12]
        passages.append(
            ResumePassage(
                passage_id=f"resume_{index:04d}_{digest}",
                text=passage_text,
                source_name=source_name,
                passage_index=index,
                content_hash=digest,
            )
        )
        if end == len(words):
            break
        start = end - overlap_words
        index += 1
    return tuple(passages)


def ingest_resume(path: str | Path) -> ResumeDocument:
    """Extract and passage-split one resume file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Resume not found: {source}")
    source_type, raw_text = extract_resume_text(source)
    quality = ensure_valid_resume_text(raw_text, source_type=source_type)
    passages = make_passages(raw_text, source_name=source.name)
    return ResumeDocument(
        source.name,
        source_type,
        raw_text,
        passages,
        input_quality=quality,
        prompt_injection_findings=detect_prompt_injection(raw_text),
    )
