import pytest

from eb1a_evaluator.resume import make_passages
from eb1a_evaluator.resume_retrieval import ResumeEvidenceRetriever


class FakeEmbeddings:
    model_name = "fake"

    def embed_documents(self, texts):
        return [[1.0, 0.0] if "award" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def test_resume_retrieval_ranks_relevant_passage_first() -> None:
    passages = make_passages(
        "Published an award-winning paper. Managed a team.",
        source_name="candidate.txt",
        target_words=4,
        overlap_words=1,
    )
    results = ResumeEvidenceRetriever(FakeEmbeddings()).retrieve(passages, "awards")
    assert "award" in results[0].text.lower()


def test_resume_retrieval_can_filter_low_similarity_passages() -> None:
    passages = make_passages(
        "Published an award-winning paper. Managed a team.",
        source_name="candidate.txt",
        target_words=4,
        overlap_words=1,
    )
    results = ResumeEvidenceRetriever(FakeEmbeddings()).retrieve(
        passages,
        "awards",
        min_score=0.9,
    )
    assert results
    assert all(item.score >= 0.9 for item in results)


def test_hybrid_resume_retrieval_records_combined_source(monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    passages = make_passages(
        "Nationally recognized award for computer vision. Managed a research team.",
        source_name="candidate.txt",
        target_words=5,
        overlap_words=1,
    )
    results = ResumeEvidenceRetriever(FakeEmbeddings()).retrieve(passages, "award")
    assert results
    assert "vector" in results[0].retrieval_method


def test_resume_retrieval_rejects_empty_inputs() -> None:
    retriever = ResumeEvidenceRetriever(FakeEmbeddings())
    with pytest.raises(ValueError, match="passages"):
        retriever.retrieve([], "awards")
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve(make_passages("award", source_name="x.txt"), " ")
