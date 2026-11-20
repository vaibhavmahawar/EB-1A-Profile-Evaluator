"""Evaluation dataset loading and benchmark metrics."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .criteria import CRITERION_IDS
from .schemas import CriterionAssessment, CriterionRating


@dataclass(frozen=True)
class EvaluationCase:
    profile_id: str
    resume_path: Path
    expected: dict[str, CriterionRating]


def load_cases(data_dir: str | Path) -> list[EvaluationCase]:
    """Load all synthetic profiles and validate their expected labels."""

    root = Path(data_dir)
    labels = json.loads((root / "expected_labels.json").read_text(encoding="utf-8"))
    cases: list[EvaluationCase] = []
    for profile_id, expected in labels.items():
        resume_path = root / f"{profile_id}.txt"
        if not resume_path.exists():
            raise FileNotFoundError(f"Profile text not found: {resume_path}")
        if set(expected) != set(CRITERION_IDS):
            raise ValueError(f"Profile {profile_id} must label exactly all ten criteria")
        cases.append(
            EvaluationCase(
                profile_id=profile_id,
                resume_path=resume_path,
                expected={
                    criterion_id: CriterionRating(value)
                    for criterion_id, value in expected.items()
                },
            )
        )
    return cases


def rating_metrics(
    expected: dict[str, CriterionRating],
    predicted: dict[str, CriterionAssessment],
) -> dict[str, Any]:
    """Compute accuracy and confusion counts for one or more criteria."""

    criterion_ids = sorted(set(expected) & set(predicted))
    correct = sum(predicted[item].rating == expected[item] for item in criterion_ids)
    confusion: Counter[str] = Counter(
        f"{expected[item].value}->{predicted[item].rating.value}" for item in criterion_ids
    )
    by_criterion = {
        item: float(predicted[item].rating == expected[item]) for item in criterion_ids
    }
    return {
        "evaluated_count": len(criterion_ids),
        "correct_count": correct,
        "accuracy": round(correct / len(criterion_ids), 4) if criterion_ids else 0.0,
        "confusion_matrix": dict(confusion),
        "accuracy_by_criterion": by_criterion,
    }


def evidence_metrics(
    assessments: list[CriterionAssessment],
    *,
    valid_citations: set[str],
    valid_resume_text: str,
) -> dict[str, Any]:
    """Compute citation validity and unsupported-evidence rates."""

    citations = [citation for item in assessments for citation in item.citation_ids]
    evidence = [quote for item in assessments for quote in item.resume_evidence]
    valid_citation_count = sum(citation in valid_citations for citation in citations)
    supported_evidence_count = sum(quote in valid_resume_text for quote in evidence)
    return {
        "citation_count": len(citations),
        "valid_citation_count": valid_citation_count,
        "citation_validity": round(valid_citation_count / len(citations), 4)
        if citations
        else 1.0,
        "evidence_count": len(evidence),
        "supported_evidence_count": supported_evidence_count,
        "hallucinated_evidence_rate": round(
            (len(evidence) - supported_evidence_count) / len(evidence), 4
        )
        if evidence
        else 0.0,
    }


def combine_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-profile metric records."""

    total = sum(item["rating"]["evaluated_count"] for item in results)
    correct = sum(item["rating"]["correct_count"] for item in results)
    citation_count = sum(item["evidence"]["citation_count"] for item in results)
    valid_citation_count = sum(item["evidence"]["valid_citation_count"] for item in results)
    confusion: Counter[str] = Counter()
    for item in results:
        confusion.update(item["rating"]["confusion_matrix"])
    return {
        "profiles": len(results),
        "evaluated_count": total,
        "correct_count": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "confusion_matrix": dict(confusion),
        "citation_validity": round(valid_citation_count / citation_count, 4)
        if citation_count
        else 1.0,
        "hallucinated_evidence_rate": round(
            sum(
                item["evidence"]["evidence_count"]
                - item["evidence"]["supported_evidence_count"]
                for item in results
            )
            / max(1, sum(item["evidence"]["evidence_count"] for item in results)),
            4,
        ),
    }
