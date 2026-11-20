import pytest

from eb1a_evaluator.bm25 import BM25Index, tokenize


def test_tokenize_normalizes_unicode_and_case() -> None:
    assert tokenize("Nationally Recognized Awards") == [
        "nationally",
        "recognized",
        "awards",
    ]


def test_bm25_ranks_keyword_match_first() -> None:
    index = BM25Index(
        [
            ("award", "nationally recognized awards for excellence", {"criterion_id": "awards"}),
            ("salary", "high salary compared with peers", {"criterion_id": "salary"}),
        ]
    )
    results = index.retrieve("recognized awards", top_k=2)
    assert results[0].document_id == "award"


def test_bm25_applies_metadata_filter() -> None:
    index = BM25Index(
        [
            ("award", "recognized awards", {"criterion_id": "awards"}),
            ("salary", "high salary", {"criterion_id": "salary"}),
        ]
    )
    results = index.retrieve(
        "recognized awards high salary",
        predicate=lambda metadata: metadata["criterion_id"] == "salary",
    )
    assert [result.document_id for result in results] == ["salary"]


def test_bm25_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="empty document"):
        BM25Index([])
    with pytest.raises(ValueError, match="k1"):
        BM25Index([("one", "text", {})], k1=0)
