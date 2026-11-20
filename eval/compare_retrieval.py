"""Compare vector-only and hybrid retrieval without calling an LLM."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from eb1a_evaluator.benchmark_config import BenchmarkConfig
from eb1a_evaluator.config import default_config
from eb1a_evaluator.corpus import load_chunks
from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.embeddings import LocalSentenceTransformerEmbeddings
from eb1a_evaluator.evaluation import load_cases
from eb1a_evaluator.resume import ingest_resume
from eb1a_evaluator.resume_retrieval import ResumeEvidenceRetriever
from eb1a_evaluator.retrieval import USCISRetriever


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / max(1, denominator), 4)


def _compare_uscis(retriever, cases, top_k: int) -> dict:
    records = []
    for case in cases:
        for criterion in EB1A_CRITERIA:
            outputs = {}
            for mode in ("vector", "hybrid"):
                os.environ["RETRIEVAL_MODE"] = mode
                retriever.retrieve_queries(
                    (criterion.retrieval_query, *criterion.query_variants),
                    criterion_id=criterion.id,
                    top_k=top_k,
                )
            modes = ("vector", "hybrid") if len(records) % 2 == 0 else ("hybrid", "vector")
            for mode in modes:
                os.environ["RETRIEVAL_MODE"] = mode
                started = time.perf_counter()
                results = retriever.retrieve_queries(
                    (criterion.retrieval_query, *criterion.query_variants),
                    criterion_id=criterion.id,
                    top_k=top_k,
                )
                outputs[mode] = {
                    "ids": [item.chunk_id for item in results],
                    "latency": time.perf_counter() - started,
                    "criterion_hit": any(
                        item.metadata.get("criterion_id") == criterion.id for item in results
                    ),
                    "bm25_only": any(
                        item.metadata.get("bm25_rank") is not None
                        and item.metadata.get("vector_rank") is None
                        for item in results
                    ),
                    "bm25_included": any(
                        item.metadata.get("bm25_rank") is not None for item in results
                    ),
                }
            vector_ids = set(outputs["vector"]["ids"])
            hybrid_ids = set(outputs["hybrid"]["ids"])
            records.append(
                {
                    "profile_id": case.profile_id,
                    "criterion_id": criterion.id,
                    "vector": outputs["vector"],
                    "hybrid": outputs["hybrid"],
                    "top_k_overlap": _rate(len(vector_ids & hybrid_ids), top_k),
                }
            )
    return _summarize_records(records, "uscis")


def _compare_resume(embeddings, cases, top_k: int) -> dict:
    retriever = ResumeEvidenceRetriever(embeddings)
    records = []
    for case in cases:
        resume = ingest_resume(case.resume_path)
        for criterion in EB1A_CRITERIA:
            outputs = {}
            for mode in ("vector", "hybrid"):
                os.environ["RETRIEVAL_MODE"] = mode
                retriever.retrieve(
                    resume.passages,
                    " ".join((criterion.retrieval_query, *criterion.query_variants)),
                    top_k=top_k,
                    min_score=0.0,
                )
            modes = ("vector", "hybrid") if len(records) % 2 == 0 else ("hybrid", "vector")
            for mode in modes:
                os.environ["RETRIEVAL_MODE"] = mode
                started = time.perf_counter()
                results = retriever.retrieve(
                    resume.passages,
                    " ".join((criterion.retrieval_query, *criterion.query_variants)),
                    top_k=top_k,
                    min_score=0.0,
                )
                outputs[mode] = {
                    "ids": [item.passage_id for item in results],
                    "latency": time.perf_counter() - started,
                    "bm25_only": any(
                        item.retrieval_method == "bm25" for item in results
                    ),
                    "bm25_included": any(
                        "bm25" in item.retrieval_method for item in results
                    ),
                }
            vector_ids = set(outputs["vector"]["ids"])
            hybrid_ids = set(outputs["hybrid"]["ids"])
            records.append(
                {
                    "profile_id": case.profile_id,
                    "criterion_id": criterion.id,
                    "vector": outputs["vector"],
                    "hybrid": outputs["hybrid"],
                    "top_k_overlap": _rate(len(vector_ids & hybrid_ids), top_k),
                }
            )
    return _summarize_records(records, "resume")


def _summarize_records(records: list[dict], name: str) -> dict:
    total = len(records)
    summary = {"retrieval_target": name, "records": total, "modes": {}}
    for mode in ("vector", "hybrid"):
        entries = [record[mode] for record in records]
        summary["modes"][mode] = {
            "empty_rate": _rate(sum(not entry["ids"] for entry in entries), total),
            "criterion_hit_rate": (
                _rate(sum(entry.get("criterion_hit", False) for entry in entries), total)
                if name == "uscis"
                else None
            ),
            "bm25_only_rate": _rate(
                sum(entry.get("bm25_only", False) for entry in entries), total
            )
            if mode == "hybrid"
            else None,
            "bm25_included_rate": _rate(
                sum(entry.get("bm25_included", False) for entry in entries), total
            )
            if mode == "hybrid"
            else None,
            "average_latency_seconds": round(
                sum(entry["latency"] for entry in entries) / max(1, total), 4
            ),
        }
    summary["average_top_k_overlap"] = round(
        sum(record["top_k_overlap"] for record in records) / max(1, total), 4
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = BenchmarkConfig.from_file(root / "eval" / "benchmark_config.json")
    config.apply_environment()
    kb_config = default_config(root)
    if not kb_config.index_dir.exists():
        raise FileNotFoundError(f"USCIS index not found at {kb_config.index_dir}")
    cases = load_cases(root / "data" / "synthetic_profiles")
    embeddings = LocalSentenceTransformerEmbeddings(kb_config.embedding_model)
    report = {
        "mode": "offline_retrieval_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_chunks": len(load_chunks(kb_config)),
        "top_k": config.uscis_retrieval_top_k,
        "configuration": config.as_dict(),
        "uscis": _compare_uscis(
            USCISRetriever(kb_config, embeddings), cases, config.uscis_retrieval_top_k
        ),
        "resume": _compare_resume(embeddings, cases, config.resume_retrieval_top_k),
    }
    output_dir = root / "eval" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "retrieval_comparison.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
