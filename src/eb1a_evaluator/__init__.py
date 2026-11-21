"""Core domain contracts for the EB-1A Profile Evaluator."""

from .criteria import EB1A_CRITERIA, CriterionDefinition
from .evaluator import CriterionEvaluator, ProfileEvaluator
from .schemas import (
    CompactCriterionLLMOutput,
    CompactProfileLLMOutput,
    CriterionAssessment,
    CriterionRating,
    ProfileEvaluation,
    expand_compact_assessment,
)

__all__ = [
    "EB1A_CRITERIA",
    "CriterionAssessment",
    "CompactCriterionLLMOutput",
    "CompactProfileLLMOutput",
    "CriterionDefinition",
    "CriterionEvaluator",
    "CriterionRating",
    "ProfileEvaluator",
    "ProfileEvaluation",
    "expand_compact_assessment",
]
