"""Offline heuristic baseline for the currently implemented awards criterion."""

from .criteria import EB1A_CRITERIA
from .schemas import CriterionAssessment, CriterionRating


class HeuristicAwardsProvider:
    """Conservative keyword baseline used without an API key."""

    def evaluate(self, profile_id: str, criterion_id: str, resume_path) -> CriterionAssessment:
        if criterion_id != "awards_and_prizes":
            raise ValueError("HeuristicAwardsProvider supports only awards_and_prizes")
        text = resume_path.read_text(encoding="utf-8")
        criterion = next(item for item in EB1A_CRITERIA if item.id == criterion_id)
        lowered = text.lower()
        award_terms = ("award", "prize", "nobel", "medal")
        has_award = any(term in lowered for term in award_terms)
        ambiguous = any(term in lowered for term in ("regional", "unclear", "not stated"))
        if not has_award:
            rating = CriterionRating.MISSING
            evidence: list[str] = []
            reasoning = "No award or prize evidence was found in the synthetic profile."
        elif ambiguous:
            rating = CriterionRating.INSUFFICIENT_INFO
            evidence = [line.strip() for line in text.splitlines() if "award" in line.lower()][:1]
            reasoning = "An award is mentioned, but recognition or scope details are incomplete."
        else:
            rating = CriterionRating.STRONG
            evidence = [line.strip() for line in text.splitlines() if "award" in line.lower()][:1]
            reasoning = "The profile explicitly describes an award or prize."
        return CriterionAssessment(
            criterion_id=criterion.id,
            criterion_name=criterion.name,
            rating=rating,
            resume_evidence=evidence,
            reasoning=reasoning,
            citation_ids=[],
            next_steps=["Collect official award documentation and recognition details."]
            if rating != CriterionRating.MISSING
            else ["Identify qualifying awards or prizes and document their recognition."],
        )
