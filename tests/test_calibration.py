from eb1a_evaluator.calibration import calibrate_assessment
from eb1a_evaluator.criteria import EB1A_CRITERIA
from eb1a_evaluator.schemas import CriterionAssessment, CriterionRating


def assessment(criterion_id: str, rating: CriterionRating, evidence=None):
    criterion = next(item for item in EB1A_CRITERIA if item.id == criterion_id)
    return CriterionAssessment(
        criterion_id=criterion_id,
        criterion_name=criterion.name,
        rating=rating,
        resume_evidence=evidence or [],
        reasoning="provider reasoning",
        citation_ids=[],
        next_steps=[],
    )


def test_no_evidence_preserves_provider_label(monkeypatch):
    monkeypatch.setenv("EVIDENCE_CALIBRATION", "1")
    result = calibrate_assessment(
        assessment("awards_and_prizes", CriterionRating.STRONG), has_evidence=False
    )
    assert result.rating is CriterionRating.STRONG


def test_strong_weak_criterion_is_capped_without_strong_signal(monkeypatch):
    monkeypatch.setenv("EVIDENCE_CALIBRATION", "1")
    result = calibrate_assessment(
        assessment(
            "high_salary", CriterionRating.STRONG,
            ["Salary is $200,000 per year."],
        ),
        has_evidence=True,
    )
    assert result.rating is CriterionRating.PARTIAL


def test_strong_signal_is_preserved(monkeypatch):
    monkeypatch.setenv("EVIDENCE_CALIBRATION", "1")
    result = calibrate_assessment(
        assessment(
            "original_contributions", CriterionRating.STRONG,
            ["The model was adopted by three hospitals."],
        ),
        has_evidence=True,
    )
    assert result.rating is CriterionRating.STRONG


def test_high_salary_non_strong_ratings_are_never_upgraded(monkeypatch):
    monkeypatch.setenv("EVIDENCE_CALIBRATION", "1")
    ratings = (
        CriterionRating.MISSING,
        CriterionRating.PARTIAL,
        CriterionRating.INSUFFICIENT_INFO,
    )
    for rating in ratings:
        result = calibrate_assessment(
            assessment("high_salary", rating, ["Salary is mentioned."]),
            has_evidence=True,
        )
        assert result.rating is rating


def test_high_salary_comparison_preserves_strong(monkeypatch):
    monkeypatch.setenv("EVIDENCE_CALIBRATION", "1")
    result = calibrate_assessment(
        assessment(
            "high_salary", CriterionRating.STRONG,
            ["Compensation is in the 95th percentile for the region."],
        ),
        has_evidence=True,
    )
    assert result.rating is CriterionRating.STRONG
