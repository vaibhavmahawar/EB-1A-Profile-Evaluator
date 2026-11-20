"""Embedding adapters for local semantic retrieval."""

from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Minimal embedding interface used by the vector store."""

    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalSentenceTransformerEmbeddings:
    """Lazy local sentence-transformers embedding provider."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Cannot embed an empty document or an empty document batch")
        return self.model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed an empty query")
        return self.model.encode(text, normalize_embeddings=True).tolist()
