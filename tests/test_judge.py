import json

import pytest

from eb1a_evaluator.judge import summarize_judgments
from eb1a_evaluator.prompts import judge_prompt
from eb1a_evaluator.resume_retrieval import RetrievedResumePassage
from eb1a_evaluator.retrieval import RetrievedChunk
from eb1a_evaluator.schemas import JudgeDecision, JudgeLLMOutput, SupportAssessment


def _record(*, predicted="Strong", expected="Strong", decision="Correct"):
    return {
        "status": "completed",
        "predicted_rating": predicted,
        "expected_rating": expected,
        "manual_label_match": predicted == expected,
        "judgment": {
            "rating_decision": decision,
            "resume_evidence_support": "Supported",
            "citation_support": "Supported",
            "reasoning_grounded": True,
        },
    }


def test_judge_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        JudgeLLMOutput.model_validate(
            {
                "criterion_id": "awards_and_prizes",
                "rating_decision": JudgeDecision.CORRECT,
                "resume_evidence_support": SupportAssessment.SUPPORTED,
                "citation_support": SupportAssessment.SUPPORTED,
                "reasoning_grounded": True,
                "confidence": "High",
                "explanation": "The assessment agrees with the label.",
                "extra": "not allowed",
            }
        )


def test_judge_summary_measures_agreement_and_grounding() -> None:
    records = [_record(), _record(predicted="Missing", expected="Strong", decision="Incorrect")]
    summary = summarize_judgments(records)
    assert summary["completed_count"] == 2
    assert summary["judge_decision_accuracy"] == 1.0
    assert summary["manual_label_match_rate"] == 0.5
    assert summary["reasoning_grounded_rate"] == 1.0


def test_judge_summary_excludes_uncertain_from_decision_accuracy() -> None:
    summary = summarize_judgments([_record(decision="Uncertain")])
    assert summary["judge_decision_accuracy"] == 0.0
    assert summary["judge_uncertain_rate"] == 1.0


def test_judge_prompt_contains_grounding_context() -> None:
    system, user = judge_prompt(
        "awards_and_prizes",
        "Strong",
        {"criterion_id": "awards_and_prizes", "rating": "Strong"},
        "Won the Global Engineering Award.",
        [RetrievedResumePassage("resume_1", "Won the Global Engineering Award.", 1.0)],
        [RetrievedChunk("uscis_1", "Award guidance", {}, 0.1)],
    )
    assert "untrusted data" in system
    assert "Manual reference rating" in user
    assert "uscis_1" in user
    assert json.dumps({"criterion_id": "awards_and_prizes", "rating": "Strong"}, indent=2) in user
