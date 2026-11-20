import pytest
from pydantic import ValidationError

from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.schemas import CriterionAssessment, CriterionRating, ProfileEvaluation


def make_assessments() -> list[CriterionAssessment]:
    return [
        CriterionAssessment(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            rating=CriterionRating.MISSING,
            reasoning="No supporting information was found.",
        )
        for criterion in EB1A_CRITERIA
    ]


def test_complete_profile_evaluation_is_valid() -> None:
    evaluation = ProfileEvaluation(
        candidate_summary="Synthetic candidate.",
        assessments=make_assessments(),
        criteria_count_strong_or_partial=0,
        readiness_score=0,
    )
    assert len(evaluation.assessments) == 10
    assert evaluation.readiness_score == 0


def test_duplicate_criteria_are_rejected() -> None:
    assessments = make_assessments()
    assessments[-1] = assessments[0]
    with pytest.raises(ValidationError, match="duplicate"):
        ProfileEvaluation(
            candidate_summary="Synthetic candidate.",
            assessments=assessments,
            criteria_count_strong_or_partial=0,
            readiness_score=0,
        )


def test_incomplete_criteria_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProfileEvaluation(
            candidate_summary="Synthetic candidate.",
            assessments=make_assessments()[:9],
            criteria_count_strong_or_partial=0,
            readiness_score=0,
        )


def test_out_of_range_score_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProfileEvaluation(
            candidate_summary="Synthetic candidate.",
            assessments=make_assessments(),
            criteria_count_strong_or_partial=0,
            readiness_score=101,
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProfileEvaluation(
            candidate_summary="Synthetic candidate.",
            assessments=make_assessments(),
            criteria_count_strong_or_partial=0,
            readiness_score=0,
            unexpected_field="not allowed",
        )


def test_mismatched_criterion_id_and_name_are_rejected() -> None:
    with pytest.raises(ValidationError, match="same criterion"):
        CriterionAssessment(
            criterion_id=EB1A_CRITERIA[0].id,
            criterion_name=EB1A_CRITERIA[1].name,
            rating=CriterionRating.MISSING,
            reasoning="Test reasoning.",
        )


def test_inconsistent_summary_values_are_rejected() -> None:
    with pytest.raises(ValidationError, match="criteria_count_strong_or_partial"):
        ProfileEvaluation(
            candidate_summary="Synthetic candidate.",
            assessments=make_assessments(),
            criteria_count_strong_or_partial=1,
            readiness_score=0,
        )

    with pytest.raises(ValidationError, match="readiness_score"):
        ProfileEvaluation(
            candidate_summary="Synthetic candidate.",
            assessments=make_assessments(),
            criteria_count_strong_or_partial=0,
            readiness_score=10,
        )
