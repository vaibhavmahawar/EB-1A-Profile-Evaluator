"""Pure presentation helpers used by the Streamlit interface."""

from .schemas import CriterionRating, ProfileEvaluation

RATING_COLORS = {
    CriterionRating.STRONG: "#198754",
    CriterionRating.PARTIAL: "#b58105",
    CriterionRating.MISSING: "#dc3545",
    CriterionRating.INSUFFICIENT_INFO: "#6c757d",
}


def rating_color(rating: CriterionRating) -> str:
    """Return a stable display color for a rating."""

    return RATING_COLORS[rating]


def format_score(score: float) -> str:
    """Format the readiness score for display."""

    return f"{score:.1f}/100"


def download_payload(evaluation: ProfileEvaluation) -> dict:
    """Return the validated JSON payload used by the download button."""

    return evaluation.model_dump(mode="json")
