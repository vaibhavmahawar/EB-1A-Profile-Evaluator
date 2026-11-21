"""Validated input and output contracts for profile evaluations."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .criteria import CRITERION_IDS, EB1A_CRITERIA


class CriterionRating(str, Enum):
    """Allowed conservative ratings for an individual criterion."""

    STRONG = "Strong"
    PARTIAL = "Partial"
    MISSING = "Missing"
    INSUFFICIENT_INFO = "Insufficient Info"


class CriterionAssessment(BaseModel):
    """Assessment of one EB-1A criterion."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    criterion_name: str
    rating: CriterionRating
    resume_evidence: list[str] = Field(default_factory=list)
    reasoning: str
    citation_ids: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str) -> str:
        if value not in CRITERION_IDS:
            raise ValueError(f"Unknown criterion_id: {value}")
        return value

    @field_validator("criterion_name")
    @classmethod
    def validate_criterion_name(cls, value: str) -> str:
        valid_names = {criterion.name for criterion in EB1A_CRITERIA}
        if value not in valid_names:
            raise ValueError(f"Unknown criterion_name: {value}")
        return value

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reasoning must not be blank")
        return value.strip()

    @field_validator("resume_evidence", "citation_ids", "next_steps")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("list items must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("list items must be unique")
        return value

    @model_validator(mode="after")
    def validate_criterion_pair(self) -> "CriterionAssessment":
        expected = next(
            criterion for criterion in EB1A_CRITERIA if criterion.id == self.criterion_id
        )
        if self.criterion_name != expected.name:
            raise ValueError("criterion_id and criterion_name must refer to the same criterion")
        return self


class CriterionLLMOutput(BaseModel):
    """Strict all-fields-required response contract sent to the LLM provider."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    criterion_name: str
    rating: CriterionRating
    resume_evidence: list[str]
    reasoning: str
    citation_ids: list[str]
    next_steps: list[str] = Field(max_length=2)

    @field_validator("criterion_id")
    @classmethod
    def validate_llm_criterion_id(cls, value: str) -> str:
        if value not in CRITERION_IDS:
            raise ValueError(f"Unknown criterion_id: {value}")
        return value

    @field_validator("reasoning")
    @classmethod
    def validate_llm_reasoning(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reasoning must not be blank")
        return value.strip()

    @field_validator("resume_evidence", "citation_ids", "next_steps")
    @classmethod
    def validate_llm_string_lists(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("list items must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("list items must be unique")
        return value


class ProfileLLMOutput(BaseModel):
    """Strict response containing one output for each supplied criterion batch."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[CriterionLLMOutput] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_unique_criteria(self) -> "ProfileLLMOutput":
        ids = [assessment.criterion_id for assessment in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("batch assessments must not contain duplicate criteria")
        return self


class CompactCriterionLLMOutput(BaseModel):
    """Small internal response contract used by the profile-batch evaluator."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    rating: CriterionRating
    evidence_indexes: list[int] = Field(max_length=3)
    citation_ids: list[str] = Field(max_length=3)
    reason: str

    @field_validator("criterion_id")
    @classmethod
    def validate_compact_criterion_id(cls, value: str) -> str:
        if value not in CRITERION_IDS:
            raise ValueError(f"Unknown criterion_id: {value}")
        return value

    @field_validator("evidence_indexes")
    @classmethod
    def validate_evidence_indexes(cls, value: list[int]) -> list[int]:
        if any(index < 0 for index in value):
            raise ValueError("evidence_indexes must contain non-negative integers")
        if len(value) != len(set(value)):
            raise ValueError("evidence_indexes must be unique")
        return value

    @field_validator("citation_ids")
    @classmethod
    def validate_compact_citation_ids(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("citation_ids must not contain blank values")
        if len(value) != len(set(value)):
            raise ValueError("citation_ids must be unique")
        return value

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value.strip()


class CompactProfileLLMOutput(BaseModel):
    """Compact response containing one or more criterion decisions."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[CompactCriterionLLMOutput] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_unique_compact_criteria(self) -> "CompactProfileLLMOutput":
        ids = [assessment.criterion_id for assessment in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("compact assessments must not contain duplicate criteria")
        return self


def expand_compact_assessment(
    compact: CompactCriterionLLMOutput,
    *,
    evidence_spans: list[str],
    next_steps: list[str] | None = None,
) -> CriterionAssessment:
    """Convert an indexed compact decision into the existing public assessment schema."""

    if any(index >= len(evidence_spans) for index in compact.evidence_indexes):
        raise ValueError("evidence_indexes contains an index outside evidence_spans")
    criterion = next(item for item in EB1A_CRITERIA if item.id == compact.criterion_id)
    return CriterionAssessment(
        criterion_id=compact.criterion_id,
        criterion_name=criterion.name,
        rating=compact.rating,
        resume_evidence=[evidence_spans[index] for index in compact.evidence_indexes],
        reasoning=compact.reason,
        citation_ids=compact.citation_ids,
        next_steps=(next_steps or [])[:2],
    )


class JudgeDecision(str, Enum):
    """How the judge assesses the original evaluator's rating."""

    CORRECT = "Correct"
    INCORRECT = "Incorrect"
    UNCERTAIN = "Uncertain"


class SupportAssessment(str, Enum):
    """Support level assigned by the judge to grounded evidence."""

    SUPPORTED = "Supported"
    PARTIALLY_SUPPORTED = "Partially Supported"
    UNSUPPORTED = "Unsupported"
    NOT_APPLICABLE = "Not Applicable"


class JudgeLLMOutput(BaseModel):
    """Strict structured response contract for the LLM-as-judge bonus."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    rating_decision: JudgeDecision
    resume_evidence_support: SupportAssessment
    citation_support: SupportAssessment
    reasoning_grounded: bool
    confidence: str
    explanation: str

    @field_validator("criterion_id")
    @classmethod
    def validate_judge_criterion_id(cls, value: str) -> str:
        if value not in CRITERION_IDS:
            raise ValueError(f"Unknown criterion_id: {value}")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        normalized = value.strip().title()
        if normalized not in {"Low", "Medium", "High"}:
            raise ValueError("confidence must be Low, Medium, or High")
        return normalized

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("explanation must not be blank")
        return value.strip()


class ProfileEvaluation(BaseModel):
    """Complete validated evaluation for one candidate profile."""

    model_config = ConfigDict(extra="forbid")

    candidate_summary: str
    assessments: list[CriterionAssessment] = Field(min_length=10, max_length=10)
    strongest_criteria: list[str] = Field(default_factory=list)
    main_gaps: list[str] = Field(default_factory=list)
    criteria_count_strong_or_partial: int = Field(ge=0, le=10)
    readiness_score: float = Field(ge=0, le=100)
    disclaimer: str = (
        "This is an informational prototype, not legal advice or a prediction of USCIS approval."
    )

    @model_validator(mode="after")
    def validate_complete_criteria(self) -> "ProfileEvaluation":
        ids = [assessment.criterion_id for assessment in self.assessments]
        expected_ids = [criterion.id for criterion in EB1A_CRITERIA]
        if len(set(ids)) != len(ids):
            raise ValueError("assessments must not contain duplicate criteria")
        if set(ids) != set(expected_ids):
            raise ValueError("assessments must contain exactly all ten EB-1A criteria")
        expected_count = sum(
            assessment.rating.value in {"Strong", "Partial"}
            for assessment in self.assessments
        )
        if self.criteria_count_strong_or_partial != expected_count:
            raise ValueError("criteria_count_strong_or_partial does not match assessments")
        rating_weights = {
            "Strong": 1.0,
            "Partial": 0.5,
            "Missing": 0.0,
            "Insufficient Info": 0.0,
        }
        expected_score = round(
            sum(rating_weights[assessment.rating.value] for assessment in self.assessments)
            / 10
            * 100,
            1,
        )
        if self.readiness_score != expected_score:
            raise ValueError("readiness_score does not match assessments")
        return self


__all__ = [
    "CriterionAssessment",
    "CriterionLLMOutput",
    "ProfileLLMOutput",
    "CompactCriterionLLMOutput",
    "CompactProfileLLMOutput",
    "expand_compact_assessment",
    "JudgeLLMOutput",
    "JudgeDecision",
    "SupportAssessment",
    "CriterionRating",
    "ProfileEvaluation",
]
