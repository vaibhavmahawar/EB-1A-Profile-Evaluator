from pathlib import Path

import pytest

from eb1a_evaluator.config import KnowledgeBaseConfig
from eb1a_evaluator.retrieval import RetrievedChunk, USCISRetriever


class FakeEmbeddings:
    model_name = "fake"

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def test_retriever_rejects_invalid_inputs(tmp_path: Path) -> None:
    # Collection opening is intentionally not reached for these validation cases.
    retriever = object.__new__(USCISRetriever)
    retriever.config = KnowledgeBaseConfig(project_root=tmp_path)
    retriever.embeddings = FakeEmbeddings()
    with pytest.raises(ValueError, match="must not be empty"):
        retriever.retrieve(" ")
    with pytest.raises(ValueError, match="Unknown criterion_id"):
        retriever.retrieve("awards", criterion_id="unknown")


def test_hybrid_fusion_preserves_both_retrieval_sources(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_RRF_CONSTANT", "60")
    vector = [RetrievedChunk("vector", "vector result", {"criterion_id": "awards"}, 0.1)]
    keyword = [RetrievedChunk("keyword", "keyword result", {"criterion_id": "awards"}, 0.2)]
    results = USCISRetriever._fuse(vector, keyword, top_k=2)
    assert {item.chunk_id for item in results} == {"vector", "keyword"}
    assert all(item.metadata["retrieval_method"] == "hybrid" for item in results)
