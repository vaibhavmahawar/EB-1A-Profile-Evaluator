"""Deterministic next-step assembly for compact evaluator responses."""

from .schemas import CriterionRating

_MISSING_STEPS = {
    "awards_and_prizes": [
        "Provide documentation of the award's recognition, scope, and selectivity.",
        "Include the award certificate, selection criteria, and number of recipients.",
    ],
    "outstanding_membership": [
        "Provide the association's fellowship or membership selection criteria.",
        "Document the role of recognized experts in selecting members or fellows.",
    ],
    "published_material_about_candidate": [
        "Provide independent articles or media coverage about the candidate and work.",
        "Include the publication outlet, date, title, and evidence of its significance.",
    ],
    "judging_work_of_others": [
        "Provide invitations, acknowledgments, or letters confirming judging assignments.",
        "Document the field, scope, and completed nature of the reviews or judging.",
    ],
    "original_contributions": [
        "Provide independent evidence of the contribution's significance and field impact.",
        "Include adoption, licensing, patent, citation, or expert-letter documentation.",
    ],
    "scholarly_articles": [
        "Provide representative publications showing authorship and publication venue.",
        "Document the professional or peer-reviewed standing of the venues.",
    ],
    "display_of_work": [
        "Provide exhibition catalogs, invitations, or public display documentation.",
        "Document that the display was an artistic exhibition or qualifying showcase.",
    ],
    "leading_or_critical_role": [
        "Provide an employer or organization letter describing responsibilities and impact.",
        "Document the organization's distinguished reputation with independent evidence.",
    ],
    "high_salary": [
        "Provide contracts, pay statements, or other compensation documentation.",
        "Include reliable comparative salary data for the field, region, or role.",
    ],
    "commercial_success_performing_arts": [
        "Provide sales, revenue, ranking, or box-office documentation.",
        "Document the candidate's participation in the performing-arts production.",
    ],
}


def next_steps_for_criterion(criterion_id: str, rating: CriterionRating) -> list[str]:
    """Return deterministic documentation steps without asking the model to format them."""

    steps = _MISSING_STEPS[criterion_id]
    if rating is CriterionRating.STRONG:
        return [f"Retain official supporting documentation for {criterion_id.replace('_', ' ')}."]
    if rating is CriterionRating.PARTIAL:
        return steps[:2]
    return steps[:2]


__all__ = ["next_steps_for_criterion"]
