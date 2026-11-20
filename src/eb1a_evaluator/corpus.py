"""USCIS source downloading, extraction, cleaning, and semantic chunking."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .config import KnowledgeBaseConfig


def download_source(config: KnowledgeBaseConfig, *, replace: bool = False) -> Path:
    """Download the configured source page; never replace a snapshot by default."""

    if config.source_snapshot.exists() and not replace:
        raise FileExistsError(
            f"Snapshot already exists at {config.source_snapshot}; pass replace=True to refresh it"
        )
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    response = httpx.get(config.source_url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    config.snapshot_html.write_bytes(response.content)
    return config.snapshot_html


def clean_html(html: str) -> str:
    """Remove non-content HTML elements and normalize visible text."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    blocks: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        value = normalize_text(element.get_text(" "))
        if value:
            blocks.append(value)
    return normalize_text("\n\n".join(blocks))


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract page text while preserving page boundaries."""

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return normalize_text("\n\n".join(pages))


def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying heading or paragraph boundaries."""

    text = text.replace("\u00a0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_criterion(text: str) -> str:
    """Map a chunk to a criterion when its heading provides a confident match."""

    lowered = text.lower()
    mappings = {
        "awards_and_prizes": ("lesser nationally or internationally recognized prizes", "awards"),
        "outstanding_membership": ("membership in associations", "membership"),
        "published_material_about_candidate": ("published material",),
        "judging_work_of_others": ("judging the work of others",),
        "original_contributions": (
            "original scientific, scholarly, artistic",
            "major significance",
        ),
        "scholarly_articles": ("authorship of scholarly articles", "scholarly articles"),
        "display_of_work": ("display of the alien's work", "artistic exhibitions"),
        "leading_or_critical_role": ("leading or critical role",),
        "high_salary": ("high salary", "significantly high remuneration"),
        "commercial_success_performing_arts": ("commercial successes", "performing arts"),
    }
    for criterion_id, phrases in mappings.items():
        if any(phrase in lowered for phrase in phrases):
            return criterion_id
    return "general"


def chunk_text(text: str, config: KnowledgeBaseConfig) -> list[dict[str, Any]]:
    """Create deterministic paragraph-aware chunks with metadata."""

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[dict[str, Any]] = []
    current: list[str] = []
    current_words = 0
    current_criterion = "general"

    for paragraph in paragraphs:
        words = paragraph.split()
        paragraph_criterion = infer_criterion(paragraph)
        if paragraph_criterion != "general" and current:
            chunk = "\n\n".join(current).strip()
            chunks.append(chunk_record(chunk, len(chunks), current_criterion))
            current = []
            current_words = 0
            current_criterion = paragraph_criterion
        elif paragraph_criterion != "general":
            current_criterion = paragraph_criterion
        if current and current_words + len(words) > config.chunk_size_words:
            chunk = "\n\n".join(current).strip()
            chunks.append(chunk_record(chunk, len(chunks), current_criterion))
            overlap = " ".join(chunk.split()[-config.chunk_overlap_words :])
            current = [overlap] if overlap else []
            current_words = len(overlap.split())
        current.append(paragraph)
        current_words += len(words)

    if current:
        chunk = "\n\n".join(current).strip()
        chunks.append(chunk_record(chunk, len(chunks), current_criterion))
    return chunks


def chunk_record(text: str, index: int, criterion_id: str) -> dict[str, Any]:
    """Create a stable chunk record."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return {
        "chunk_id": f"uscis_{criterion_id}_{index:04d}_{digest}",
        "text": text,
        "criterion_id": criterion_id,
        "source_title": "USCIS Policy Manual Volume 6 Part F Chapter 2",
        "source_url": DEFAULT_SOURCE_URL,
        "section": criterion_id,
        "content_hash": digest,
    }


DEFAULT_SOURCE_URL = (
    "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2"
)


def build_processed_corpus(config: KnowledgeBaseConfig) -> list[dict[str, Any]]:
    """Extract, chunk, and save the local baseline corpus."""

    source_path = config.source_snapshot
    if source_path.suffix.lower() == ".html":
        text = clean_html(source_path.read_text(encoding="utf-8"))
    else:
        text = extract_pdf_text(source_path)
    chunks = chunk_text(text, config)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    config.processed_json.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    manifest = {
        "source_url": config.source_url,
        "source_file": str(source_path.name),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "chunk_count": len(chunks),
        "chunk_size_words": config.chunk_size_words,
        "chunk_overlap_words": config.chunk_overlap_words,
    }
    config.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return chunks


def load_chunks(config: KnowledgeBaseConfig) -> list[dict[str, Any]]:
    """Load processed chunks from disk."""

    if not config.processed_json.exists():
        raise FileNotFoundError(f"Processed corpus not found: {config.processed_json}")
    return json.loads(config.processed_json.read_text(encoding="utf-8"))
