from pathlib import Path

from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.schemas import CriterionAssessment, CriterionRating, ProfileEvaluation
from eb1a_evaluator.ui import download_payload, format_score, rating_color


def make_evaluation() -> ProfileEvaluation:
    assessments = [
        CriterionAssessment(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            rating=CriterionRating.MISSING,
            reasoning="No evidence.",
        )
        for criterion in EB1A_CRITERIA
    ]
    return ProfileEvaluation(
        candidate_summary="Synthetic candidate.",
        assessments=assessments,
        criteria_count_strong_or_partial=0,
        readiness_score=0,
    )


def test_ui_helpers() -> None:
    assert format_score(35) == "35.0/100"
    assert rating_color(CriterionRating.STRONG).startswith("#")
    payload = download_payload(make_evaluation())
    assert len(payload["assessments"]) == 10
    assert payload["readiness_score"] == 0.0


def test_app_file_exists() -> None:
    assert (Path(__file__).resolve().parents[1] / "app.py").exists()
