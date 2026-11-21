"""Run the synthetic evaluation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from eb1a_evaluator.benchmark_config import BenchmarkConfig
from eb1a_evaluator.config import default_config
from eb1a_evaluator.corpus import load_chunks
from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.embeddings import LocalSentenceTransformerEmbeddings
from eb1a_evaluator.evaluation import combine_metrics, evidence_metrics, load_cases, rating_metrics
from eb1a_evaluator.evaluator import (
    CriterionEvaluator,
    OpenAICriterionClient,
    ProfileEvaluator,
    _load_project_env,
)
from eb1a_evaluator.heuristics import HeuristicAwardsProvider
from eb1a_evaluator.response_cache import SQLiteResponseCache
from eb1a_evaluator.resume import ingest_resume
from eb1a_evaluator.retrieval import USCISRetriever


def run(
    mode: str,
    data_dir: Path,
    output_dir: Path,
    limit: int | None = None,
    *,
    benchmark_config: BenchmarkConfig | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
    report_name: str = "live_full_profile_report.json",
    strategy_name: str | None = None,
    batch_profile: bool | None = None,
) -> dict:
    if mode == "live":
        return run_live(
            data_dir,
            output_dir,
            limit=limit,
            benchmark_config=benchmark_config,
            no_cache=no_cache,
            refresh_cache=refresh_cache,
            report_name=report_name,
            strategy_name=strategy_name,
            batch_profile=batch_profile,
        )

    cases = load_cases(data_dir)
    if limit is not None:
        cases = cases[:limit]
    provider = HeuristicAwardsProvider()
    results = []
    for case in cases:
        predicted = provider.evaluate(case.profile_id, "awards_and_prizes", case.resume_path)
        predicted_map = {predicted.criterion_id: predicted}
        expected_map = {"awards_and_prizes": case.expected["awards_and_prizes"]}
        rating = rating_metrics(expected_map, predicted_map)
        evidence = evidence_metrics(
            [predicted],
            valid_citations=set(),
            valid_resume_text=case.resume_path.read_text(encoding="utf-8"),
        )
        results.append(
            {
                "profile_id": case.profile_id,
                "criterion": "awards_and_prizes",
                "rating": rating,
                "evidence": evidence,
            }
        )
    report = {
        "mode": mode,
        "scope": "awards_and_prizes only",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "aggregate": combine_metrics(results),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{mode}_awards_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report written to {output_path}")
    return report


def run_live(
    data_dir: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    benchmark_config: BenchmarkConfig | None = None,
    no_cache: bool = False,
    refresh_cache: bool = False,
    report_name: str = "live_full_profile_report.json",
    strategy_name: str | None = None,
    batch_profile: bool | None = None,
) -> dict:
    """Run the real retrieval and Groq evaluator over all ten criteria."""

    root = Path(__file__).resolve().parents[1]
    if benchmark_config is None:
        config_path = root / "eval" / "benchmark_config.json"
        benchmark_config = (
            BenchmarkConfig.from_file(config_path)
            if config_path.exists()
            else BenchmarkConfig()
        )
    # Load .env before applying the benchmark defaults so an explicit provider
    # selection such as LLM_PROVIDER=gemini is not overwritten by the JSON config.
    _load_project_env()
    benchmark_config.apply_environment()
    if batch_profile is not None:
        os.environ["LLM_BATCH_PROFILE"] = "1" if batch_profile else "0"
        os.environ["GROQ_BATCH_PROFILE"] = os.environ["LLM_BATCH_PROFILE"]
    config = default_config(root)
    if not config.index_dir.exists():
        raise FileNotFoundError(
            f"USCIS index not found at {config.index_dir}; run scripts/build_uscis_index.py first"
        )

    cases = load_cases(data_dir)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[:limit]
    embeddings = LocalSentenceTransformerEmbeddings(config.embedding_model)
    retriever = USCISRetriever(config, embeddings)
    cache = None
    if benchmark_config.cache_enabled and not no_cache:
        cache_path = Path(benchmark_config.cache_path)
        if not cache_path.is_absolute():
            cache_path = root / cache_path
        cache = SQLiteResponseCache(cache_path, refresh=refresh_cache)
    llm_client = OpenAICriterionClient(
        cache=cache,
        prompt_version=benchmark_config.prompt_version,
    )
    criterion_evaluator = CriterionEvaluator(retriever, embeddings, llm_client)
    profile_evaluator = ProfileEvaluator(criterion_evaluator)
    valid_citations = {chunk["chunk_id"] for chunk in load_chunks(config)}

    flattened_results = []
    profile_results = []
    retrieval_latencies: list[float] = []
    retrieval_empty_count = 0
    retrieval_total = 0
    failure_types: Counter[str] = Counter()
    failure_categories: Counter[str] = Counter()
    guardrail_counts: Counter[str] = Counter()

    for profile_index, case in enumerate(cases, start=1):
        print(
            f"[live] Profile {profile_index}/{len(cases)}: {case.profile_id}",
            flush=True,
        )
        try:
            resume = ingest_resume(case.resume_path)
        except Exception as error:
            guardrail_counts["input_validation_failures"] += 1
            profile_results.append(
                {
                    "profile_id": case.profile_id,
                    "criteria": [],
                    "guardrails": {
                        "input_error_category": "input_validation",
                        "error_type": type(error).__name__,
                    },
                }
            )
            continue
        if resume.prompt_injection_findings:
            guardrail_counts["prompt_injection_detections"] += len(
                resume.prompt_injection_findings
            )
        for criterion_index, criterion in enumerate(EB1A_CRITERIA, start=1):
            print(
                f"[live]   Retrieving criterion {criterion_index}/10: {criterion.id}",
                flush=True,
            )
            started = time.perf_counter()
            retrieved = retriever.retrieve_queries(
                (criterion.retrieval_query, *criterion.query_variants),
                criterion_id=criterion.id,
                top_k=int(os.getenv("USCIS_RETRIEVAL_TOP_K", "3")),
            )
            retrieval_latencies.append(time.perf_counter() - started)
            retrieval_total += 1
            retrieval_empty_count += not bool(retrieved)

        evaluation = profile_evaluator.evaluate(resume)
        print(f"[live]   Completed model evaluation for {case.profile_id}", flush=True)
        predicted = {item.criterion_id: item for item in evaluation.assessments}
        profile_criteria = []
        for criterion_id, expected_rating in case.expected.items():
            assessment = predicted[criterion_id]
            rating = rating_metrics(
                {criterion_id: expected_rating},
                {criterion_id: assessment},
            )
            evidence = evidence_metrics(
                [assessment],
                valid_citations=valid_citations,
                valid_resume_text=resume.raw_text,
            )
            structured_valid = not assessment.reasoning.startswith(
                "Evaluation failed for this criterion:"
            )
            if not structured_valid:
                failure_type = assessment.reasoning.split(":", 1)[-1].strip()
                failure_type = failure_type.split(" [", 1)[0].rstrip(".")
                failure_types[failure_type] += 1
                category = "unknown"
                if "[" in assessment.reasoning and "]" in assessment.reasoning:
                    category = assessment.reasoning.split("[", 1)[-1].split("]", 1)[0]
                failure_categories[category] += 1
                if category == "evidence_validation":
                    guardrail_counts["unsupported_evidence_rejections"] += 1
                elif category == "citation_validation":
                    guardrail_counts["invalid_citation_rejections"] += 1
                elif category == "structured_output":
                    guardrail_counts["structured_output_failures"] += 1
                elif category in {
                    "authentication",
                    "quota_exhausted",
                    "rate_limit",
                    "timeout",
                    "connection",
                    "provider_server",
                }:
                    guardrail_counts["provider_failures"] += 1
            record = {
                "criterion": criterion_id,
                "rating": rating,
                "evidence": evidence,
                "structured_output_valid": structured_valid,
                "assessment": assessment.model_dump(mode="json"),
            }
            flattened_results.append(record)
            profile_criteria.append(record)
        profile_results.append(
            {
                "profile_id": case.profile_id,
                "criteria": profile_criteria,
                "guardrails": {
                    "input_quality": (
                        resume.input_quality.__dict__ if resume.input_quality else None
                    ),
                    "prompt_injection_categories": [
                        finding.category for finding in resume.prompt_injection_findings
                    ],
                },
            }
        )

    aggregate = combine_metrics(flattened_results)
    valid_records = [
        item for item in flattened_results if item["structured_output_valid"]
    ]
    valid_aggregate = combine_metrics(valid_records)
    by_criterion = {}
    for criterion_id in sorted({item["criterion"] for item in flattened_results}):
        by_criterion[criterion_id] = combine_metrics(
            [item for item in flattened_results if item["criterion"] == criterion_id]
        )
    structured_total = len(flattened_results)
    structured_valid = sum(item["structured_output_valid"] for item in flattened_results)
    effective_configuration = benchmark_config.as_dict()
    effective_configuration["batch_profile"] = (
        os.getenv("LLM_BATCH_PROFILE", "0") == "1"
    )
    effective_configuration["batch_size"] = int(
        os.getenv("LLM_BATCH_SIZE", str(benchmark_config.batch_size))
    )
    report = {
        "mode": "live",
        "scope": "all ten criteria",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_metadata": {
            "strategy_name": strategy_name,
            "configuration_hash": benchmark_config.config_hash(),
            "configuration": effective_configuration,
            "cache": llm_client.diagnostics()["cache"],
            "provider": llm_client.diagnostics(),
        },
        "profiles": profile_results,
        "aggregate": {
            **aggregate,
            "profiles": len(cases),
            "structured_output_validity": round(
                structured_valid / max(1, structured_total), 4
            ),
            "evaluation_failure_count": structured_total - structured_valid,
            "evaluation_failure_types": dict(failure_types),
            "evaluation_failure_categories": dict(failure_categories),
            "valid_output_evaluated_count": valid_aggregate["evaluated_count"],
            "valid_output_correct_count": valid_aggregate["correct_count"],
            "valid_output_accuracy": valid_aggregate["accuracy"],
            "accuracy_by_criterion": by_criterion,
        },
        "retrieval_diagnostics": {
            "retrieval_count": retrieval_total,
            "empty_retrieval_rate": round(retrieval_empty_count / max(1, retrieval_total), 4),
            "average_retrieval_latency_seconds": round(
                sum(retrieval_latencies) / max(1, len(retrieval_latencies)), 4
            ),
        },
        "guardrails": dict(guardrail_counts),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / report_name
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(json.dumps(report["retrieval_diagnostics"], indent=2))
    if cache is not None:
        cache.close()
    print(f"Report written to {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["heuristic", "live"],
        default="heuristic",
        help="heuristic runs the offline awards baseline; live runs all ten criteria with Groq",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N synthetic profiles; useful for a smoke test",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="benchmark JSON configuration; defaults to eval/benchmark_config.json",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable successful provider-response replay for this run",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="clear the configured cache before a live run",
    )
    parser.add_argument(
        "--report-name",
        default="live_full_profile_report.json",
        help="output filename under eval/results; use a unique name to preserve runs",
    )
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--batch-profile",
        dest="batch_profile",
        action="store_true",
        help="use compact profile-level batching for live evaluation",
    )
    batch_group.add_argument(
        "--criterion-profile",
        dest="batch_profile",
        action="store_false",
        help="use the criterion-level evaluation path for live evaluation",
    )
    parser.set_defaults(batch_profile=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark_config = None
    if args.config:
        config_path = args.config if args.config.is_absolute() else root / args.config
        benchmark_config = BenchmarkConfig.from_file(config_path)
    run(
        args.mode,
        root / "data" / "synthetic_profiles",
        root / "eval" / "results",
        limit=args.limit,
        benchmark_config=benchmark_config,
        no_cache=args.no_cache,
        refresh_cache=args.refresh_cache,
        report_name=args.report_name,
        batch_profile=args.batch_profile,
    )
