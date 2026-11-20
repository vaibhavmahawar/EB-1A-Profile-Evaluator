import pytest

from eb1a_evaluator.embeddings import LocalSentenceTransformerEmbeddings


def test_empty_embedding_inputs_are_rejected() -> None:
    provider = LocalSentenceTransformerEmbeddings("test-model")
    with pytest.raises(ValueError, match="empty query"):
        provider.embed_query("   ")
    with pytest.raises(ValueError, match="empty document"):
        provider.embed_documents(["valid", ""])
