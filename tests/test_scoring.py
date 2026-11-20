import pytest
from pydantic import ValidationError

from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.schemas import CriterionAssessment, CriterionRating
from eb1a_evaluator.scoring import (
    count_strong_or_partial,
    main_gaps,
    readiness_score,
    strongest_criteria,
)


def assessment(index: int, rating: CriterionRating) -> CriterionAssessment:
    criterion = EB1A_CRITERIA[index]
    return CriterionAssessment(
        criterion_id=criterion.id,
        criterion_name=criterion.name,
        rating=rating,
        reasoning="Test reasoning.",
    )


def test_all_strong_scores_100() -> None:
    items = [assessment(index, CriterionRating.STRONG) for index in range(10)]
    assert readiness_score(items) == 100.0
    assert count_strong_or_partial(items) == 10


def test_all_partial_scores_50() -> None:
    items = [assessment(index, CriterionRating.PARTIAL) for index in range(10)]
    assert readiness_score(items) == 50.0
    assert count_strong_or_partial(items) == 10


def test_missing_and_insufficient_info_score_zero() -> None:
    items = [
        assessment(
            index,
            CriterionRating.MISSING if index < 5 else CriterionRating.INSUFFICIENT_INFO,
        )
        for index in range(10)
    ]
    assert readiness_score(items) == 0.0
    assert count_strong_or_partial(items) == 0


def test_mixed_scores_and_summaries() -> None:
    ratings = [
        CriterionRating.STRONG,
        CriterionRating.PARTIAL,
        CriterionRating.MISSING,
        CriterionRating.INSUFFICIENT_INFO,
        CriterionRating.STRONG,
        CriterionRating.PARTIAL,
        CriterionRating.MISSING,
        CriterionRating.INSUFFICIENT_INFO,
        CriterionRating.MISSING,
        CriterionRating.PARTIAL,
    ]
    items = [assessment(index, rating) for index, rating in enumerate(ratings)]
    assert readiness_score(items) == 35.0
    assert count_strong_or_partial(items) == 5
    assert strongest_criteria(items) == ["awards_and_prizes", "original_contributions"]
    assert main_gaps(items) == [
        "published_material_about_candidate",
        "judging_work_of_others",
        "display_of_work",
        "leading_or_critical_role",
        "high_salary",
    ]


def test_scoring_requires_ten_assessments() -> None:
    with pytest.raises(ValueError, match="exactly ten"):
        readiness_score([assessment(0, CriterionRating.STRONG)])


def test_unknown_criterion_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown criterion_id"):
        CriterionAssessment(
            criterion_id="not_a_real_criterion",
            criterion_name="Awards and prizes",
            rating=CriterionRating.STRONG,
            reasoning="Test reasoning.",
        )


def test_only_two_next_steps_are_allowed() -> None:
    criterion = EB1A_CRITERIA[0]
    with pytest.raises(ValidationError):
        CriterionAssessment(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            rating=CriterionRating.PARTIAL,
            reasoning="Test reasoning.",
            next_steps=["one", "two", "three"],
        )
