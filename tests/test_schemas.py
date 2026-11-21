import pytest
from pydantic import ValidationError

from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.recommendations import next_steps_for_criterion
from eb1a_evaluator.schemas import (
    CompactCriterionLLMOutput,
    CompactProfileLLMOutput,
    CriterionAssessment,
    CriterionRating,
    ProfileEvaluation,
    expand_compact_assessment,
)


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


def test_compact_assessment_validates_and_expands_to_public_schema() -> None:
    compact = CompactCriterionLLMOutput(
        criterion_id="awards_and_prizes",
        rating=CriterionRating.INSUFFICIENT_INFO,
        evidence_indexes=[0],
        citation_ids=["chunk-1"],
        reason="The award is listed, but recognition and selectivity are not documented.",
    )

    expanded = expand_compact_assessment(
        compact,
        evidence_spans=["She received the Early Career Award."],
        next_steps=["Collect official award selection documentation."],
    )

    assert isinstance(CompactProfileLLMOutput(assessments=[compact]), CompactProfileLLMOutput)
    assert expanded.criterion_name == "Awards and prizes"
    assert expanded.resume_evidence == ["She received the Early Career Award."]
    assert expanded.reasoning == compact.reason
    assert expanded.next_steps == ["Collect official award selection documentation."]


def test_compact_assessment_rejects_invalid_evidence_index() -> None:
    compact = CompactCriterionLLMOutput(
        criterion_id="awards_and_prizes",
        rating="Missing",
        evidence_indexes=[1],
        citation_ids=[],
        reason="No evidence was supplied.",
    )

    with pytest.raises(ValueError, match="outside evidence_spans"):
        expand_compact_assessment(compact, evidence_spans=["Only span"])


def test_compact_profile_rejects_duplicate_criteria() -> None:
    item = CompactCriterionLLMOutput(
        criterion_id="awards_and_prizes",
        rating="Missing",
        evidence_indexes=[],
        citation_ids=[],
        reason="No evidence was supplied.",
    )

    with pytest.raises(ValueError, match="duplicate"):
        CompactProfileLLMOutput(assessments=[item, item])


def test_next_steps_are_assembled_locally_by_criterion() -> None:
    steps = next_steps_for_criterion("leading_or_critical_role", CriterionRating.PARTIAL)
    assert len(steps) == 2
    assert "organization" in steps[1].lower()
