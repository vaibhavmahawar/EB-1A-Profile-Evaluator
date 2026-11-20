"""Deterministic scoring and summary helpers."""

from collections.abc import Iterable

from .schemas import CriterionAssessment, CriterionRating

RATING_WEIGHTS: dict[CriterionRating, float] = {
    CriterionRating.STRONG: 1.0,
    CriterionRating.PARTIAL: 0.5,
    CriterionRating.MISSING: 0.0,
    CriterionRating.INSUFFICIENT_INFO: 0.0,
}


def readiness_score(assessments: Iterable[CriterionAssessment]) -> float:
    """Return a 0–100 readiness indicator from criterion ratings."""

    items = list(assessments)
    if len(items) != 10:
        raise ValueError("readiness_score requires exactly ten assessments")
    return round(sum(RATING_WEIGHTS[item.rating] for item in items) / 10 * 100, 1)


def count_strong_or_partial(assessments: Iterable[CriterionAssessment]) -> int:
    """Count assessments with meaningful supporting evidence."""

    return sum(
        item.rating in {CriterionRating.STRONG, CriterionRating.PARTIAL}
        for item in assessments
    )


def strongest_criteria(assessments: Iterable[CriterionAssessment]) -> list[str]:
    """Return criterion IDs rated Strong, preserving input order."""

    return [item.criterion_id for item in assessments if item.rating == CriterionRating.STRONG]


def main_gaps(assessments: Iterable[CriterionAssessment]) -> list[str]:
    """Return criterion IDs needing evidence, preserving input order."""

    gap_ratings = {CriterionRating.MISSING, CriterionRating.INSUFFICIENT_INFO}
    return [item.criterion_id for item in assessments if item.rating in gap_ratings]
