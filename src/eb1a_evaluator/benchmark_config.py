"""Reproducible benchmark configuration and runtime settings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings that define a benchmark run independently of API credentials."""

    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    prompt_version: str = "criterion-v3"
    request_delay_seconds: float = 5.0
    retry_backoff_seconds: float = 20.0
    max_retries: int = 2
    rate_limit_max_wait_seconds: float = 60.0
    timeout_seconds: float = 60.0
    batch_profile: bool = False
    batch_size: int = 2
    uscis_retrieval_top_k: int = 3
    resume_retrieval_top_k: int = 3
    resume_retrieval_min_score: float = 0.15
    evidence_span_top_k: int = 5
    evidence_span_min_score: float = 0.15
    evidence_fast_path: bool = False
    retrieval_mode: str = "hybrid"
    retrieval_candidate_pool: int = 10
    hybrid_rrf_constant: int = 60
    hybrid_vector_weight: float = 0.5
    hybrid_bm25_weight: float = 0.5
    cache_enabled: bool = True
    cache_path: str = "eval/cache/provider_responses.sqlite3"

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {field for field in cls.__dataclass_fields__}
        values = {key: value for key, value in raw.items() if key in allowed}
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def apply_environment(self) -> None:
        """Apply benchmark settings while allowing explicit provider overrides."""

        import os

        provider = (os.getenv("LLM_PROVIDER") or self.provider).lower()
        provider_model_key = "GEMINI_MODEL" if provider == "gemini" else "GROQ_MODEL"
        model = os.getenv("LLM_MODEL") or os.getenv(provider_model_key) or self.model
        values = {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": model,
            "LLM_PROMPT_VERSION": os.getenv("LLM_PROMPT_VERSION") or self.prompt_version,
            "LLM_REQUEST_DELAY_SECONDS": str(self.request_delay_seconds),
            "LLM_RETRY_BACKOFF_SECONDS": str(self.retry_backoff_seconds),
            "LLM_MAX_RETRIES": str(self.max_retries),
            "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": str(self.rate_limit_max_wait_seconds),
            "LLM_TIMEOUT_SECONDS": str(self.timeout_seconds),
            "LLM_BATCH_PROFILE": "1" if self.batch_profile else "0",
            "LLM_BATCH_SIZE": str(self.batch_size),
            "GROQ_MODEL": model,
            "GROQ_PROMPT_VERSION": self.prompt_version,
            "GROQ_REQUEST_DELAY_SECONDS": str(self.request_delay_seconds),
            "GROQ_RETRY_BACKOFF_SECONDS": str(self.retry_backoff_seconds),
            "GROQ_MAX_RETRIES": str(self.max_retries),
            "GROQ_RATE_LIMIT_MAX_WAIT_SECONDS": str(self.rate_limit_max_wait_seconds),
            "GROQ_TIMEOUT_SECONDS": str(self.timeout_seconds),
            "GROQ_BATCH_PROFILE": "1" if self.batch_profile else "0",
            "GROQ_BATCH_SIZE": str(self.batch_size),
            "USCIS_RETRIEVAL_TOP_K": str(self.uscis_retrieval_top_k),
            "RESUME_RETRIEVAL_TOP_K": str(self.resume_retrieval_top_k),
            "RESUME_RETRIEVAL_MIN_SCORE": str(self.resume_retrieval_min_score),
            "RESUME_EVIDENCE_SPAN_TOP_K": str(self.evidence_span_top_k),
            "EVIDENCE_SPAN_MIN_SCORE": str(self.evidence_span_min_score),
            "EVIDENCE_FAST_PATH": "1" if self.evidence_fast_path else "0",
            "RETRIEVAL_MODE": self.retrieval_mode,
            "RETRIEVAL_CANDIDATE_POOL": str(self.retrieval_candidate_pool),
            "HYBRID_RRF_CONSTANT": str(self.hybrid_rrf_constant),
            "HYBRID_VECTOR_WEIGHT": str(self.hybrid_vector_weight),
            "HYBRID_BM25_WEIGHT": str(self.hybrid_bm25_weight),
        }
        for key, value in values.items():
            os.environ[key] = value
