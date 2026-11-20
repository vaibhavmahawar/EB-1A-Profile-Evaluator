"""Single-criterion OpenAI evaluation with deterministic verification."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from .criteria import EB1A_CRITERIA
from .json_recovery import parse_json_payload
from .prompts import criterion_prompt, profile_prompt
from .response_cache import SQLiteResponseCache, failure_category, make_cache_key
from .resume import ResumeDocument, normalize_resume_text
from .resume_retrieval import ResumeEvidenceRetriever
from .retrieval import RetrievedChunk, USCISRetriever
from .schemas import (
    CriterionAssessment,
    CriterionLLMOutput,
    CriterionRating,
    JudgeLLMOutput,
    ProfileEvaluation,
    ProfileLLMOutput,
)
from .scoring import count_strong_or_partial, main_gaps, readiness_score, strongest_criteria


def _load_project_env() -> None:
    """Load simple KEY=VALUE entries from a local ignored .env file."""

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)
        return


def _failure_reason(error: Exception) -> str:
    """Keep raw exception type and normalized category in fallback output."""

    return (
        f"Evaluation failed for this criterion: {type(error).__name__} "
        f"[{failure_category(error)}]."
    )


class RateLimitAbortError(RuntimeError):
    """Raised when a provider asks us to wait longer than the safety limit."""


@dataclass(frozen=True)
class EvaluationInputs:
    resume: ResumeDocument
    uscis_chunks: list[RetrievedChunk]
    resume_passages: list


class OpenAICriterionClient:
    """OpenAI-compatible structured-output client configured for Groq."""

    def __init__(
        self,
        model: str | None = None,
        client=None,
        *,
        cache: SQLiteResponseCache | None = None,
        prompt_version: str | None = None,
        provider: str | None = None,
    ) -> None:
        _load_project_env()
        # Test doubles and injected clients historically use the Groq-shaped
        # Responses API unless a provider is explicitly supplied. Real clients
        # continue to honor LLM_PROVIDER from the environment.
        selected_provider = provider or (
            os.getenv("LLM_PROVIDER", "groq") if client is None else "groq"
        )
        self.provider = selected_provider.lower()
        if self.provider not in {"groq", "gemini"}:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        default_model = (
            "gemini-3.5-flash-lite"
            if self.provider == "gemini"
            else "openai/gpt-oss-20b"
        )
        self.model = model or os.getenv("LLM_MODEL") or os.getenv(
            "GEMINI_MODEL" if self.provider == "gemini" else "GROQ_MODEL",
            default_model,
        )
        self.prompt_version = prompt_version or os.getenv(
            "LLM_PROMPT_VERSION",
            os.getenv("GROQ_PROMPT_VERSION", "criterion-v3"),
        )
        self.request_delay_seconds = float(
            os.getenv(
                "LLM_REQUEST_DELAY_SECONDS",
                os.getenv("GROQ_REQUEST_DELAY_SECONDS", "5"),
            )
        )
        self.retry_backoff_seconds = float(
            os.getenv(
                "LLM_RETRY_BACKOFF_SECONDS",
                os.getenv("GROQ_RETRY_BACKOFF_SECONDS", "20"),
            )
        )
        self.max_rate_limit_retries = max(
            0, int(os.getenv("LLM_MAX_RETRIES", os.getenv("GROQ_MAX_RETRIES", "2")))
        )
        self.max_rate_limit_wait_seconds = max(
            0.0,
            float(
                os.getenv(
                    "LLM_RATE_LIMIT_MAX_WAIT_SECONDS",
                    os.getenv("GROQ_RATE_LIMIT_MAX_WAIT_SECONDS", "60"),
                )
            ),
        )
        self._last_request_at: float | None = None
        self.cache = cache
        self.provider_calls = 0
        self.request_latencies: list[float] = []
        self.token_usage: dict[str, int] = {}

        if client is None:
            from openai import OpenAI

            api_key = os.getenv("LLM_API_KEY") or os.getenv(
                "GEMINI_API_KEY" if self.provider == "gemini" else "GROQ_API_KEY"
            )
            if not api_key:
                raise RuntimeError(
                    f"API key is not configured for provider {self.provider}"
                )

            default_base_url = (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
                if self.provider == "gemini"
                else "https://api.groq.com/openai/v1"
            )

            client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("LLM_BASE_URL", default_base_url),
                timeout=float(
                    os.getenv("LLM_TIMEOUT_SECONDS", os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
                ),
                max_retries=0,
            )

        self.client = client

    def parse_assessment(self, system: str, user: str) -> CriterionAssessment:
        cache_key = self._cache_key("criterion", system, user)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return CriterionAssessment.model_validate(cached)
        self._wait_for_request()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = self._request_with_rate_limit_retry(
                lambda: self._structured_parse(messages, CriterionLLMOutput)
            )
            parsed = self._parsed_output(response)
            if parsed is None:
                raise ValueError("OpenAI returned no structured assessment")
            if isinstance(parsed, CriterionAssessment):
                assessment = parsed
            elif isinstance(parsed, CriterionLLMOutput):
                assessment = CriterionAssessment.model_validate(parsed.model_dump())
            else:
                assessment = CriterionAssessment.model_validate(parsed)
            self._store_cached(cache_key, assessment.model_dump(mode="json"), response)
            return assessment
        except Exception as error:
            fallback_errors = {"BadRequestError", "ValidationError", "ValueError"}
            if type(error).__name__ not in fallback_errors:
                raise
            fallback_model, fallback = self._parse_fallback_model(
                messages, CriterionLLMOutput
            )
            assessment = CriterionAssessment.model_validate(fallback_model.model_dump())
            self._store_cached(cache_key, assessment.model_dump(mode="json"), fallback)
            return assessment

    def parse_profile_assessments(
        self, system: str, user: str
    ) -> list[CriterionAssessment]:
        """Parse one batch response and convert its strict items to domain models."""

        cache_key = self._cache_key("profile_batch", system, user)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return [CriterionAssessment.model_validate(item) for item in cached["assessments"]]
        self._wait_for_request()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = self._request_with_rate_limit_retry(
                lambda: self._structured_parse(messages, ProfileLLMOutput)
            )
            parsed = self._parsed_output(response)
            if parsed is None:
                raise ValueError("OpenAI returned no batch assessment")
            if not isinstance(parsed, ProfileLLMOutput):
                parsed = ProfileLLMOutput.model_validate(parsed)
        except Exception as error:
            fallback_errors = {"BadRequestError", "ValidationError", "ValueError"}
            if type(error).__name__ not in fallback_errors:
                raise
            parsed, fallback = self._parse_fallback_model(messages, ProfileLLMOutput)
        assessments = [
            CriterionAssessment.model_validate(item.model_dump())
            for item in parsed.assessments
        ]
        self._store_cached(
            cache_key,
            {"assessments": [item.model_dump(mode="json") for item in assessments]},
            response if "response" in locals() else fallback,
        )
        return assessments

    def parse_judgment(self, system: str, user: str) -> JudgeLLMOutput:
        """Parse one structured LLM-as-judge response with cache and fallback."""

        cache_key = self._cache_key("judge", system, user)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return JudgeLLMOutput.model_validate(cached)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            response = self._request_with_rate_limit_retry(
                lambda: self._structured_parse(messages, JudgeLLMOutput)
            )
            parsed = self._parsed_output(response)
            if parsed is None:
                raise ValueError("OpenAI returned no structured judgment")
            judgment = (
                parsed
                if isinstance(parsed, JudgeLLMOutput)
                else JudgeLLMOutput.model_validate(parsed)
            )
        except Exception as error:
            if type(error).__name__ not in {"BadRequestError", "ValidationError", "ValueError"}:
                raise
            judgment, fallback = self._parse_fallback_model(messages, JudgeLLMOutput)
            response = fallback
        self._store_cached(cache_key, judgment.model_dump(mode="json"), response)
        return judgment

    def _structured_parse(self, messages, output_model):
        if self.provider == "gemini":
            return self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=output_model,
            )
        return self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=output_model,
        )

    def _parsed_output(self, response):
        if self.provider == "gemini":
            return response.choices[0].message.parsed
        return response.output_parsed

    def _parse_fallback_model(self, messages, output_model):
        """Use JSON mode, then one bounded repair request if parsing fails."""

        fallback = self._request_with_rate_limit_retry(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        )
        content = fallback.choices[0].message.content
        try:
            return output_model.model_validate(
                self._normalize_provider_payload(content, output_model)
            ), fallback
        except (TypeError, ValueError):
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Return only one valid JSON object matching the requested schema. "
                        "Do not add Markdown, commentary, or unknown fields."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Repair the following provider output into the requested JSON schema. "
                        "Do not invent missing facts; preserve all available values.\n\n"
                        f"Provider output:\n{content or '<empty>'}"
                    ),
                },
            ]
            repaired = self._request_with_rate_limit_retry(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=repair_messages,
                    response_format={"type": "json_object"},
                )
            )
            repaired_content = repaired.choices[0].message.content
            try:
                return output_model.model_validate(
                    self._normalize_provider_payload(repaired_content, output_model)
                ), repaired
            except (TypeError, ValueError) as repair_error:
                raise ValueError(
                    "Provider JSON remained invalid after one repair attempt"
                ) from repair_error

    @staticmethod
    def _normalize_provider_payload(content: str, output_model):
        """Normalize safe Gemini/Groq array-shaped batch responses."""

        payload = parse_json_payload(content)
        if isinstance(payload, list):
            if output_model.__name__ != "ProfileLLMOutput":
                raise ValueError("Single-criterion provider response must be an object")
            return {"assessments": payload}
        if output_model.__name__ == "ProfileLLMOutput":
            for key in ("criteria", "results", "outputs"):
                if isinstance(payload.get(key), list) and "assessments" not in payload:
                    return {"assessments": payload[key]}
        return payload

    def _wait_for_request(self) -> None:
        if self.request_delay_seconds > 0 and self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.request_delay_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _request_with_rate_limit_retry(self, request):
        """Retry with bounded exponential backoff and fail fast on long waits."""

        for attempt in range(self.max_rate_limit_retries + 1):
            self._wait_for_request()
            try:
                started = time.perf_counter()
                self.provider_calls += 1
                response = request()
                self.request_latencies.append(time.perf_counter() - started)
                return response
            except Exception as error:
                if type(error).__name__ != "RateLimitError":
                    raise
                if attempt >= self.max_rate_limit_retries:
                    raise
                server_wait = self._retry_after_seconds(error)
                exponential_wait = self.retry_backoff_seconds * (2**attempt)
                wait_seconds = max(server_wait, exponential_wait)
                if wait_seconds > self.max_rate_limit_wait_seconds:
                    raise RateLimitAbortError(
                        "Rate limit requires waiting "
                        f"{wait_seconds:.1f}s, exceeding the configured maximum of "
                        f"{self.max_rate_limit_wait_seconds:.1f}s. "
                        "Stopping instead of waiting; reduce request volume or wait "
                        "for quota reset."
                    ) from error
                time.sleep(wait_seconds)
        raise RuntimeError("Unreachable rate-limit retry state")

    @staticmethod
    def _retry_after_seconds(error: Exception) -> float:
        """Read Retry-After when available, with a conservative fallback."""

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        raw_value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            if raw_value is not None:
                return min(max(float(raw_value), 1.0), 300.0)
        except (TypeError, ValueError):
            pass
        return float(
            os.getenv("LLM_RETRY_BACKOFF_SECONDS", os.getenv("GROQ_RETRY_BACKOFF_SECONDS", "20"))
        )

    def _cache_key(self, request_type: str, system: str, user: str) -> str:
        return make_cache_key(
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            request_type=request_type,
            system=system,
            user=user,
        )

    def _get_cached(self, cache_key: str) -> dict | None:
        if self.cache is None:
            return None
        entry = self.cache.get(cache_key)
        return entry.value if entry else None

    def _store_cached(self, cache_key: str, value: dict, response) -> None:
        if self.cache is None:
            return
        usage = getattr(response, "usage", None)
        token_usage = None
        if usage is not None:
            token_usage = {
                key: getattr(usage, key)
                for key in ("input_tokens", "output_tokens", "total_tokens")
                if getattr(usage, key, None) is not None
            }
            for key, value in token_usage.items():
                self.token_usage[key] = self.token_usage.get(key, 0) + int(value)
        self.cache.set(
            cache_key,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            request_type="structured",
            value=value,
            token_usage=token_usage,
        )

    def diagnostics(self) -> dict:
        result = {
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "provider_calls": self.provider_calls,
            "request_latency_seconds": self.request_latencies,
            "average_request_latency_seconds": round(
                sum(self.request_latencies) / max(1, len(self.request_latencies)), 4
            ),
            "token_usage": self.token_usage,
            "max_rate_limit_wait_seconds": self.max_rate_limit_wait_seconds,
            "max_rate_limit_retries": self.max_rate_limit_retries,
        }
        result["cache"] = (
            self.cache.diagnostics()
            if self.cache is not None
            else {"enabled": False, "hits": 0, "misses": 0, "writes": 0}
        )
        return result


def verify_assessment(
    assessment: CriterionAssessment,
    *,
    resume_passages: list,
    uscis_chunks: list[RetrievedChunk],
    resume_text: str | None = None,
) -> CriterionAssessment:
    """Validate evidence and citation references against retrieved inputs."""

    searchable_resume = normalize_resume_text(
        resume_text or "\n".join(passage.text for passage in resume_passages)
    )
    valid_citations = {chunk.chunk_id for chunk in uscis_chunks}
    if any(not citation.strip() for citation in assessment.citation_ids):
        raise ValueError("Assessment contains a blank citation ID")
    if any(citation not in valid_citations for citation in assessment.citation_ids):
        raise ValueError("Assessment contains a citation ID that was not retrieved")
    if any(not quote.strip() for quote in assessment.resume_evidence):
        raise ValueError("Assessment contains blank resume evidence")
    unsupported_evidence = [
        quote
        for quote in assessment.resume_evidence
        if normalize_resume_text(quote) not in searchable_resume
    ]
    if unsupported_evidence:
        raise ValueError("Assessment contains resume evidence not found in retrieved passages")
    return assessment


class CriterionEvaluator:
    """Evaluate any one of the ten EB-1A criteria."""

    def __init__(self, uscis_retriever: USCISRetriever, embeddings, llm_client) -> None:
        self.uscis_retriever = uscis_retriever
        self.resume_retriever = ResumeEvidenceRetriever(embeddings)
        self.llm_client = llm_client

    @staticmethod
    def _criterion_queries(criterion) -> tuple[str, ...]:
        return (criterion.retrieval_query, *criterion.query_variants)

    def _retrieve_context(self, criterion, resume: ResumeDocument):
        top_k = int(os.getenv("USCIS_RETRIEVAL_TOP_K", "3"))
        queries = self._criterion_queries(criterion)
        if hasattr(self.uscis_retriever, "retrieve_queries"):
            uscis_chunks = self.uscis_retriever.retrieve_queries(
                queries,
                criterion_id=criterion.id,
                top_k=top_k,
            )
        else:
            uscis_chunks = self.uscis_retriever.retrieve(
                criterion.retrieval_query,
                criterion_id=criterion.id,
                top_k=top_k,
            )
        resume_passages = self.resume_retriever.retrieve_evidence_spans(
            resume.passages,
            " ".join(self._criterion_queries(criterion)),
            top_k=int(os.getenv("RESUME_EVIDENCE_SPAN_TOP_K", "5")),
            min_score=float(os.getenv("EVIDENCE_SPAN_MIN_SCORE", "0.15")),
        )
        return resume_passages, uscis_chunks

    def evaluate(self, criterion_id: str, resume: ResumeDocument) -> CriterionAssessment:
        criterion = criterion_definition(criterion_id)
        resume_passages, uscis_chunks = self._retrieve_context(criterion, resume)
        if not resume_passages and os.getenv("EVIDENCE_FAST_PATH", "0") == "1":
            return self._no_evidence_assessment(criterion)
        system, user = criterion_prompt(
            criterion.id,
            resume_passages,
            uscis_chunks,
            prompt_version=os.getenv(
                "LLM_PROMPT_VERSION", os.getenv("GROQ_PROMPT_VERSION", "criterion-v3")
            ),
        )
        try:
            assessment = self.llm_client.parse_assessment(system, user)
        except Exception as error:
            retryable_errors = {
                "APIConnectionError",
                "APITimeoutError",
                "InternalServerError",
                "ValueError",
            }
            if type(error).__name__ not in retryable_errors:
                raise
            retry_delay = float(
                os.getenv(
                    "LLM_RETRY_BACKOFF_SECONDS",
                    os.getenv("GROQ_RETRY_BACKOFF_SECONDS", "20"),
                )
            )
            time.sleep(retry_delay)
            assessment = self.llm_client.parse_assessment(system, user)
        return verify_assessment(
            assessment,
            resume_passages=resume_passages,
            uscis_chunks=uscis_chunks,
            resume_text=resume.raw_text,
        )

    def evaluate_batch(self, resume: ResumeDocument) -> list[CriterionAssessment]:
        """Evaluate criteria in small provider batches to reduce rate-limit pressure."""

        contexts = []
        for criterion in EB1A_CRITERIA:
            resume_passages, uscis_chunks = self._retrieve_context(criterion, resume)
            contexts.append((criterion, resume_passages, uscis_chunks))
        verified = []
        batch_size = max(
            1,
            int(os.getenv("LLM_BATCH_SIZE", os.getenv("GROQ_BATCH_SIZE", "2"))),
        )
        for start in range(0, len(contexts), batch_size):
            group = contexts[start : start + batch_size]
            try:
                system, user = profile_prompt(
                    group,
                    prompt_version=os.getenv(
                        "LLM_PROMPT_VERSION",
                        os.getenv("GROQ_PROMPT_VERSION", "criterion-v3"),
                    ),
                )
                assessments = self.llm_client.parse_profile_assessments(system, user)
                by_id = {assessment.criterion_id: assessment for assessment in assessments}
                for criterion, resume_passages, uscis_chunks in group:
                    assessment = by_id.get(criterion.id)
                    if assessment is None:
                        raise ValueError(f"Batch response omitted criterion {criterion.id}")
                    try:
                        verified.append(
                            verify_assessment(
                                assessment,
                                resume_passages=resume_passages,
                                uscis_chunks=uscis_chunks,
                                resume_text=resume.raw_text,
                            )
                        )
                    except Exception as error:
                        verified.append(self._failure_assessment(criterion, error))
            except Exception as error:
                verified.extend(
                    self._failure_assessment(criterion, error)
                    for criterion, _, _ in group
                )
        return verified

    @staticmethod
    def _failure_assessment(criterion, error: Exception) -> CriterionAssessment:
        return CriterionAssessment(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            rating=CriterionRating.INSUFFICIENT_INFO,
            reasoning=_failure_reason(error),
            next_steps=[],
        )

    @staticmethod
    def _no_evidence_assessment(criterion) -> CriterionAssessment:
        return CriterionAssessment(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            rating=CriterionRating.INSUFFICIENT_INFO,
            reasoning=(
                "No criterion-specific evidence span was retrieved from the supplied profile. "
                "Additional documentation is required before this criterion can be assessed."
            ),
            next_steps=["Provide profile evidence directly addressing this criterion."],
        )


class AwardsCriterionEvaluator:
    """Backward-compatible awards-only evaluator."""

    def __init__(self, uscis_retriever: USCISRetriever, embeddings, llm_client) -> None:
        self.evaluator = CriterionEvaluator(uscis_retriever, embeddings, llm_client)

    def evaluate(self, resume: ResumeDocument) -> CriterionAssessment:
        return self.evaluator.evaluate("awards_and_prizes", resume)


class ProfileEvaluator:
    """Evaluate all ten criteria and return a complete profile assessment."""

    def __init__(self, criterion_evaluator: CriterionEvaluator) -> None:
        self.criterion_evaluator = criterion_evaluator

    def evaluate(self, resume: ResumeDocument):
        from .criteria import EB1A_CRITERIA

        assessments: list[CriterionAssessment] = []
        if (
            os.getenv("LLM_BATCH_PROFILE", os.getenv("GROQ_BATCH_PROFILE", "0")) == "1"
            and hasattr(self.criterion_evaluator, "evaluate_batch")
        ):
            try:
                assessments = self.criterion_evaluator.evaluate_batch(resume)
            except Exception as error:
                assessments = [
                    CriterionAssessment(
                        criterion_id=criterion.id,
                        criterion_name=criterion.name,
                        rating=CriterionRating.INSUFFICIENT_INFO,
                        reasoning=_failure_reason(error),
                        next_steps=[],
                    )
                    for criterion in EB1A_CRITERIA
                ]
        if assessments:
            return self._build_profile_evaluation(assessments)

        for criterion in EB1A_CRITERIA:
            try:
                assessment = self.criterion_evaluator.evaluate(criterion.id, resume)
            except Exception as error:
                if type(error).__name__ == "RateLimitAbortError":
                    raise
                assessment = CriterionAssessment(
                    criterion_id=criterion.id,
                    criterion_name=criterion.name,
                    rating=CriterionRating.INSUFFICIENT_INFO,
                    reasoning=_failure_reason(error),
                    next_steps=[],
                )
            assessments.append(assessment)
        return self._build_profile_evaluation(assessments)

    @staticmethod
    def _build_profile_evaluation(assessments: list[CriterionAssessment]):
        strong_or_partial = count_strong_or_partial(assessments)
        strong = strongest_criteria(assessments)
        gaps = main_gaps(assessments)
        return ProfileEvaluation(
            candidate_summary=(
                f"Evaluated all ten EB-1A criteria. {len(strong)} criteria were rated Strong, "
                f"and {strong_or_partial} were rated Strong or Partial."
            ),
            assessments=assessments,
            strongest_criteria=strong,
            main_gaps=gaps,
            criteria_count_strong_or_partial=strong_or_partial,
            readiness_score=readiness_score(assessments),
        )


def criterion_definition(criterion_id: str):
    """Return a criterion definition or fail clearly."""

    return next(item for item in EB1A_CRITERIA if item.id == criterion_id)
