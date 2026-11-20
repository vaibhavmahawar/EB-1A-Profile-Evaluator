import json
from pathlib import Path

from eb1a_evaluator.evaluation import combine_metrics, evidence_metrics, load_cases, rating_metrics
from eb1a_evaluator.schemas import CriterionAssessment, CriterionRating

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic_profiles"


def test_evaluation_dataset_has_eight_profiles_and_all_criteria() -> None:
    cases = load_cases(DATA)
    assert len(cases) == 8
    assert all(len(case.expected) == 10 for case in cases)


def test_rating_metrics_calculate_accuracy_and_confusion() -> None:
    expected = {"awards_and_prizes": CriterionRating.STRONG}
    predicted = {
        "awards_and_prizes": CriterionAssessment(
            criterion_id="awards_and_prizes",
            criterion_name="Awards and prizes",
            rating=CriterionRating.STRONG,
            reasoning="test",
        )
    }
    metrics = rating_metrics(expected, predicted)
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == {"Strong->Strong": 1}


def test_evidence_metrics_detect_unsupported_evidence() -> None:
    assessment = CriterionAssessment(
        criterion_id="awards_and_prizes",
        criterion_name="Awards and prizes",
        rating=CriterionRating.PARTIAL,
        resume_evidence=["present quote", "invented quote"],
        reasoning="test",
        citation_ids=["valid", "invalid"],
    )
    metrics = evidence_metrics(
        [assessment],
        valid_citations={"valid"},
        valid_resume_text="present quote",
    )
    assert metrics["citation_validity"] == 0.5
    assert metrics["hallucinated_evidence_rate"] == 0.5


def test_combine_metrics_treats_absent_citations_as_not_applicable() -> None:
    assessment = CriterionAssessment(
        criterion_id="awards_and_prizes",
        criterion_name="Awards and prizes",
        rating=CriterionRating.MISSING,
        reasoning="test",
    )
    metrics = combine_metrics(
        [
            {
                "rating": rating_metrics(
                    {"awards_and_prizes": CriterionRating.MISSING},
                    {"awards_and_prizes": assessment},
                ),
                "evidence": evidence_metrics(
                    [assessment],
                    valid_citations=set(),
                    valid_resume_text="",
                ),
            }
        ]
    )
    assert metrics["citation_validity"] == 1.0


def test_expected_labels_file_is_valid_json() -> None:
    payload = json.loads((DATA / "expected_labels.json").read_text(encoding="utf-8"))
    assert "prompt_injection_candidate" in payload
