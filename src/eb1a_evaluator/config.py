"""Configuration defaults for the knowledge-base pipeline."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_USCIS_URL = (
    "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2"
)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """Filesystem and retrieval settings for a local knowledge base."""

    project_root: Path
    source_url: str = DEFAULT_USCIS_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size_words: int = 450
    chunk_overlap_words: int = 75
    top_k: int = 5

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "uscis" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "uscis" / "processed"

    @property
    def index_dir(self) -> Path:
        return self.project_root / "data" / "index"

    @property
    def snapshot_pdf(self) -> Path:
        return self.raw_dir / "uscis_eb1a_policy_manual.pdf"

    @property
    def snapshot_html(self) -> Path:
        return self.raw_dir / "uscis_eb1a_policy_manual.html"

    @property
    def source_snapshot(self) -> Path:
        """Prefer the actual policy chapter HTML, falling back to the PDF snapshot."""

        if self.snapshot_html.exists():
            return self.snapshot_html
        return self.snapshot_pdf

    @property
    def processed_json(self) -> Path:
        return self.processed_dir / "uscis_chunks.json"

    @property
    def manifest_json(self) -> Path:
        return self.processed_dir / "metadata.json"


def default_config(project_root: str | Path | None = None) -> KnowledgeBaseConfig:
    """Build the default local configuration."""

    root = Path(project_root or Path.cwd()).resolve()
    return KnowledgeBaseConfig(project_root=root)
