from pathlib import Path

import pytest

from eb1a_evaluator.evaluator import OpenAICriterionClient
from eb1a_evaluator.json_recovery import parse_json_payload
from eb1a_evaluator.schemas import CompactProfileLLMOutput

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "provider_outputs"


@pytest.mark.parametrize(
    "fixture",
    [
        "fenced_compact_batch.txt",
        "trailing_comma_object.txt",
        "array_batch.txt",
        "results_envelope.txt",
    ],
)
def test_provider_fixtures_recover_compact_payloads(fixture: str) -> None:
    content = (FIXTURE_DIR / fixture).read_text(encoding="utf-8")
    assert parse_json_payload(content) is not None
    normalized = OpenAICriterionClient._normalize_provider_payload(
        content,
        CompactProfileLLMOutput,
    )
    parsed = CompactProfileLLMOutput.model_validate(normalized)
    assert parsed.assessments[0].criterion_id == "awards_and_prizes"


def test_provider_fixture_rejects_non_json() -> None:
    with pytest.raises(ValueError):
        parse_json_payload("The provider returned no JSON at all.")
