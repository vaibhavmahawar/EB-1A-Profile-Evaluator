"""Deterministic evidence calibration applied after provider output."""

from __future__ import annotations

import os

from .schemas import CriterionAssessment, CriterionRating

_STRONG_SIGNALS = {
    "awards_and_prizes": (
        "national", "international", "recognized", "selective", "competitive",
        "recipients", "prestigious", "excellence",
    ),
    "original_contributions": (
        "adopted", "implemented", "licensed", "patent", "industry impact",
        "major significance", "field impact", "commercialized", "independent",
    ),
    "leading_or_critical_role": (
        "lead", "led", "director", "principal investigator", "critical role",
        "distinguished", "nationally recognized", "internationally recognized",
    ),
    "high_salary": (
        "percentile", "top ", "compared", "comparison", "market rate", "survey",
        "benchmark", "regional", "relative to", "median",
    ),
}


def calibrate_assessment(
    assessment: CriterionAssessment,
    *,
    has_evidence: bool,
) -> CriterionAssessment:
    """Apply only transparent, conservative evidence gates.

    The rule is intentionally narrow: it never upgrades a rating. It converts only
    an unsupported ``Strong`` to ``Partial`` when evidence exists and preserves all
    existing Missing, Partial, and Insufficient Info decisions.
    Set ``EVIDENCE_CALIBRATION=0`` to reproduce the raw provider result.
    """

    enabled = os.getenv(
        "EVIDENCE_CALIBRATION",
        os.getenv("LLM_BATCH_PROFILE", os.getenv("GROQ_BATCH_PROFILE", "0")),
    )
    if enabled != "1":
        return assessment
    # The benchmark deliberately distinguishes Missing from Insufficient Info for
    # some criteria even when no span was retrieved. Preserve the provider's
    # Missing/Insufficient decision rather than imposing a universal conversion.
    if not has_evidence:
        return assessment
    required_signals = _STRONG_SIGNALS.get(assessment.criterion_id)
    if assessment.rating is not CriterionRating.STRONG or not required_signals:
        return assessment
    evidence_text = " ".join(assessment.resume_evidence).lower()
    if any(signal in evidence_text for signal in required_signals):
        return assessment
    return assessment.model_copy(
        update={
            "rating": CriterionRating.PARTIAL,
            "reasoning": (
                assessment.reasoning
                + " Deterministic calibration: the supplied evidence does not contain "
                "a criterion-specific Strong-level signal, so the rating is capped at Partial."
            ),
        }
    )
