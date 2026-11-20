"""LLM-as-judge evaluation for completed profile assessments."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import default_config
from .corpus import load_chunks
from .embeddings import LocalSentenceTransformerEmbeddings
from .evaluation import load_cases
from .evaluator import OpenAICriterionClient, criterion_definition
from .prompts import judge_prompt
from .response_cache import SQLiteResponseCache, failure_category
from .resume import ingest_resume
from .resume_retrieval import ResumeEvidenceRetriever
from .retrieval import USCISRetriever
from .schemas import CriterionAssessment, JudgeDecision


def _judgment_distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(
        Counter(
            record["judgment"][field]
            for record in records
            if record.get("judgment", {}).get(field)
        )
    )


def _judge_decision_matches_label(record: dict[str, Any]) -> bool | None:
    decision = record.get("judgment", {}).get("rating_decision")
    manual_match = record.get("manual_label_match")
    if decision == JudgeDecision.UNCERTAIN.value or manual_match is None:
        return None
    return (decision == JudgeDecision.CORRECT.value) == manual_match


def summarize_judgments(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate judge quality and grounding metrics from completed records."""

    completed = [record for record in records if record.get("status") == "completed"]
    judge_matches = [
        match
        for match in (_judge_decision_matches_label(record) for record in completed)
        if match is not None
    ]
    grounded = [record for record in completed if "judgment" in record]
    return {
        "evaluated_count": len(records),
        "completed_count": len(completed),
        "judge_failure_count": len(records) - len(completed),
        "judge_decision_accuracy": round(sum(judge_matches) / len(judge_matches), 4)
        if judge_matches
        else 0.0,
        "judge_uncertain_rate": round(
            sum(
                record["judgment"]["rating_decision"] == JudgeDecision.UNCERTAIN.value
                for record in completed
            )
            / max(1, len(completed)),
            4,
        ),
        "manual_label_match_rate": round(
            sum(record["manual_label_match"] for record in completed) / max(1, len(completed)),
            4,
        ),
        "resume_evidence_support": _judgment_distribution(
            grounded, "resume_evidence_support"
        ),
        "citation_support": _judgment_distribution(grounded, "citation_support"),
        "reasoning_grounded_rate": round(
            sum(record["judgment"]["reasoning_grounded"] for record in completed)
            / max(1, len(completed)),
            4,
        ),
        "failure_categories": dict(
            Counter(
                record.get("failure_category", "unknown")
                for record in records
                if record.get("status") != "completed"
            )
        ),
    }


def run_judge(
    report_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    """Judge an existing live report without rerunning the original evaluator."""

    root = Path(__file__).resolve().parents[2]
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("mode") != "live":
        raise ValueError("The judge requires a live evaluator report")
    profile_rows = report.get("profiles", [])
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        profile_rows = profile_rows[:limit]

    cases = {case.profile_id: case for case in load_cases(root / "data" / "synthetic_profiles")}
    config = default_config(root)
    if not config.index_dir.exists():
        raise FileNotFoundError(
            f"USCIS index not found at {config.index_dir}; run scripts/build_uscis_index.py first"
        )
    embeddings = LocalSentenceTransformerEmbeddings(config.embedding_model)
    retriever = USCISRetriever(config, embeddings)
    resume_retriever = ResumeEvidenceRetriever(embeddings)
    valid_citations = {chunk["chunk_id"] for chunk in load_chunks(config)}
    judge_provider = os.getenv(
        "LLM_PROVIDER",
        report.get("run_metadata", {}).get("configuration", {}).get("provider", "groq"),
    )
    judge_model = os.getenv(
        "LLM_JUDGE_MODEL",
        report.get("run_metadata", {}).get("configuration", {}).get(
            "model", "openai/gpt-oss-20b"
        ),
    )
    os.environ["LLM_PROVIDER"] = judge_provider
    os.environ["LLM_MODEL"] = judge_model
    os.environ["LLM_PROMPT_VERSION"] = "judge-v1"
    cache = None
    cache_config = report.get("run_metadata", {}).get("configuration", {})
    if cache_config.get("cache_enabled", True) and not no_cache:
        cache_path = Path(cache_config.get("cache_path", "eval/cache/provider_responses.sqlite3"))
        if not cache_path.is_absolute():
            cache_path = root / cache_path
        cache = SQLiteResponseCache(cache_path, refresh=refresh_cache)
    client = OpenAICriterionClient(
        model=judge_model,
        cache=cache,
        prompt_version="judge-v1",
        provider=judge_provider,
    )
    records: list[dict[str, Any]] = []
    provider_diagnostics = {}
    try:
        for profile in profile_rows:
            profile_id = profile["profile_id"]
            case = cases.get(profile_id)
            if case is None:
                raise ValueError(f"Profile {profile_id} is not present in expected_labels.json")
            resume = ingest_resume(case.resume_path)
            for criterion_row in profile.get("criteria", []):
                criterion_id = criterion_row["criterion"]
                expected = case.expected[criterion_id].value
                assessment = CriterionAssessment.model_validate(criterion_row["assessment"])
                criterion = criterion_definition(criterion_id)
                uscis_chunks = retriever.retrieve(
                    criterion.retrieval_query,
                    criterion_id=criterion_id,
                    top_k=int(os.getenv("USCIS_RETRIEVAL_TOP_K", "3")),
                )
                resume_passages = resume_retriever.retrieve(
                    resume.passages,
                    criterion.retrieval_query,
                    top_k=int(os.getenv("RESUME_RETRIEVAL_TOP_K", "3")),
                    min_score=float(os.getenv("RESUME_RETRIEVAL_MIN_SCORE", "0.15")),
                )
                system, user = judge_prompt(
                    criterion_id,
                    expected,
                    assessment.model_dump(mode="json"),
                    resume.raw_text,
                    resume_passages,
                    uscis_chunks,
                )
                try:
                    judgment = client.parse_judgment(system, user)
                    records.append(
                        {
                            "profile_id": profile_id,
                            "criterion": criterion_id,
                            "status": "completed",
                            "expected_rating": expected,
                            "predicted_rating": assessment.rating.value,
                            "manual_label_match": assessment.rating.value == expected,
                            "assessment": assessment.model_dump(mode="json"),
                            "judgment": judgment.model_dump(mode="json"),
                            "citation_ids_available": [chunk.chunk_id for chunk in uscis_chunks],
                            "valid_citation_ids": sorted(
                                set(assessment.citation_ids) & valid_citations
                            ),
                        }
                    )
                except Exception as error:
                    records.append(
                        {
                            "profile_id": profile_id,
                            "criterion": criterion_id,
                            "status": "failed",
                            "expected_rating": expected,
                            "predicted_rating": assessment.rating.value,
                            "manual_label_match": assessment.rating.value == expected,
                            "failure_type": type(error).__name__,
                            "failure_category": failure_category(error),
                        }
                    )
    finally:
        provider_diagnostics = client.diagnostics()
        if cache is not None:
            cache.close()

    by_criterion = {}
    for criterion_id in sorted({record["criterion"] for record in records}):
        by_criterion[criterion_id] = summarize_judgments(
            [record for record in records if record["criterion"] == criterion_id]
        )
    result = {
        "mode": "llm_as_judge",
        "source_report": str(Path(report_path)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_metadata": {
            "model": judge_model,
            "provider_name": judge_provider,
            "prompt_version": "judge-v1",
            "provider": provider_diagnostics,
        },
        "aggregate": {
            **summarize_judgments(records),
            "accuracy_by_criterion": by_criterion,
        },
        "records": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Judge report written to {destination}")
    return result
