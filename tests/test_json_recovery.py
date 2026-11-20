import pytest

from eb1a_evaluator.json_recovery import parse_json_object, parse_json_payload


@pytest.mark.parametrize(
    ("content", "key"),
    [
        ('{"rating": "Strong"}', "rating"),
        ('```json\n{"rating": "Strong"}\n```', "rating"),
        ('Here is the result:\n{"rating": "Strong"}\nDone.', "rating"),
        ('{"rating": "Strong",}', "rating"),
        ('{"reasoning": "A comma, inside text",}', "reasoning"),
    ],
)
def test_parse_json_object_recovers_safe_wrappers(content: str, key: str) -> None:
    assert key in parse_json_object(content)


def test_parse_json_object_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_json_object("")


def test_parse_json_object_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="object"):
        parse_json_object('["not", "an", "object"]')


def test_parse_json_payload_accepts_array_for_batch_normalization() -> None:
    assert parse_json_payload('[{"criterion_id": "awards_and_prizes"}]')[0][
        "criterion_id"
    ] == "awards_and_prizes"
