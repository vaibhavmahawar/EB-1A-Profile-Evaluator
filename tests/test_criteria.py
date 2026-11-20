from eb1a_evaluator.criteria import CRITERION_IDS, EB1A_CRITERIA


def test_all_ten_criteria_are_defined_once() -> None:
    assert len(EB1A_CRITERIA) == 10
    assert len(CRITERION_IDS) == 10
    assert [criterion.id for criterion in EB1A_CRITERIA] == [
        "awards_and_prizes",
        "outstanding_membership",
        "published_material_about_candidate",
        "judging_work_of_others",
        "original_contributions",
        "scholarly_articles",
        "display_of_work",
        "leading_or_critical_role",
        "high_salary",
        "commercial_success_performing_arts",
    ]
