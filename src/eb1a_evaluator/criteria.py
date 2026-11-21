"""Stable definitions for the ten EB-1A criteria."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CriterionDefinition:
    """A stable identifier and human-readable name for one criterion."""

    id: str
    name: str
    retrieval_query: str
    evidence_guidance: str
    query_variants: tuple[str, ...] = ()
    rating_guidance: str = ""
    calibration_examples: tuple[str, ...] = ()


EB1A_CRITERIA: tuple[CriterionDefinition, ...] = (
    CriterionDefinition(
        "awards_and_prizes",
        "Awards and prizes",
        "nationally or internationally recognized awards or prizes for excellence in the field",
        "Look for the candidate's receipt of a qualifying award, its recognition, scope, "
        "and basis.",
        ("selective award competition excellence prize nationally internationally recognized",),
        "A listed award is not Strong without recognition, selectivity, scope, or "
        "comparable proof.",
        ("Strong: nationally recognized and highly selective award with recipient counts. "
         "Partial: award is listed but prestige or selectivity is not documented. "
         "Missing: no award claim appears.",),
    ),
    CriterionDefinition(
        "outstanding_membership",
        "Membership in associations requiring outstanding achievement",
        "membership in associations requiring outstanding achievement judged by recognized experts",
        "Look for selective membership or fellowship where recognized experts judge "
        "outstanding achievement.",
        ("elected fellow selective fellowship expert judged outstanding achievement",),
        "Membership must be selective and based on expert judgment of achievement.",
    ),
    CriterionDefinition(
        "published_material_about_candidate",
        "Published material about the candidate",
        "published material about the candidate in professional or major trade publications",
        "Look for published material about the candidate, including title, outlet, "
        "date, and authorship context.",
        ("profile article news coverage interview trade publication about candidate",),
        "The candidate's own scholarly articles are not published material about the candidate.",
    ),
    CriterionDefinition(
        "judging_work_of_others",
        "Judging the work of others",
        "participation as a judge of the work of others in the field",
        "Look for judging, peer review, selection panels, or comparable evaluation of "
        "others' work.",
        (
            "peer reviewer technical program committee editorial board selection panel "
            "evaluated others",
        ),
        "Reviewing or selecting others' work supports the criterion; the role and "
        "assignments should be explicit.",
    ),
    CriterionDefinition(
        "original_contributions",
        "Original contributions of major significance",
        "original scientific scholarly artistic or business contributions of major significance",
        "Look for original work and specific evidence of significance or impact in the field.",
        ("adopted implementation licensed patent industry impact citations major significance",),
        "Original work alone is insufficient; Strong requires concrete field impact or "
        "independent significance evidence.",
        ("Strong: original work plus adoption, licensing, patents, or independent evidence "
         "of major field impact. Partial: original work or citations are described without "
         "major-significance proof. Missing: no contribution claim appears.",),
    ),
    CriterionDefinition(
        "scholarly_articles",
        "Authorship of scholarly articles",
        "authorship of scholarly articles in professional or major trade publications",
        "Look for authorship, publication venue, peer review, and scholarly or "
        "professional context.",
        ("published peer reviewed journal conference scholarly authorship",),
        "Authorship and publication venue are the core requirements; do not require "
        "impact evidence for this criterion.",
    ),
    CriterionDefinition(
        "display_of_work",
        "Display of work at exhibitions",
        "display of the candidate's work at artistic exhibitions or showcases",
        "Look for the candidate's work being displayed at qualifying artistic exhibitions.",
        ("artistic exhibition showcase gallery public display",),
        "Require an artistic exhibition or showcase, not an ordinary technical presentation.",
    ),
    CriterionDefinition(
        "leading_or_critical_role",
        "Leading or critical role in distinguished organizations",
        "leading or critical role for organizations or establishments with a "
        "distinguished reputation",
        "Look for the candidate's role, responsibilities, and the distinguished "
        "reputation of the organization.",
        (
            "research group lead principal investigator senior leadership critical "
            "responsibilities distinguished organization",
        ),
        "Strong requires both a leading or critical role and evidence that the "
        "organization is distinguished.",
        ("Strong: a clearly leading or critical role and independent evidence the organization "
         "is distinguished. Partial: the role or organization reputation is asserted but one "
         "side lacks support. Missing: no qualifying role claim appears.",),
    ),
    CriterionDefinition(
        "high_salary",
        "High salary or remuneration",
        "high salary or significantly high remuneration compared with others in the field",
        "Look for compensation and reliable comparative evidence for the field, region, or role.",
        ("compensation remuneration equity salary survey wage comparison field region",),
        "A salary amount alone is insufficient; Strong requires reliable comparison "
        "with the field, region, or role.",
        (
            "Strong: compensation plus a reliable percentile, market, regional, or role "
            "comparison. Partial: compensation is stated without comparison. Missing: no "
            "remuneration claim appears. Insufficient Info: remuneration is mentioned but "
            "the amount or comparability cannot be determined.",
        ),
    ),
    CriterionDefinition(
        "commercial_success_performing_arts",
        "Commercial success in the performing arts",
        "commercial successes in the performing arts demonstrated by box office or sales receipts",
        "Look for box office receipts, sales, rankings, revenue, or comparable "
        "commercial evidence.",
        ("box office receipts ticket sales revenue rankings performing arts commercial success",),
        "Require performing-arts participation plus sales, revenue, box-office, "
        "ranking, or comparable evidence.",
    ),
)

CRITERION_IDS = frozenset(criterion.id for criterion in EB1A_CRITERIA)
