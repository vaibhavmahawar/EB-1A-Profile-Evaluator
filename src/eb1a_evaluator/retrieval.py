"""Criterion-aware retrieval over the USCIS knowledge base."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Index
from .config import KnowledgeBaseConfig
from .corpus import load_chunks
from .criteria import CRITERION_IDS
from .embeddings import EmbeddingProvider
from .vector_store import open_collection


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved USCIS chunk ready for evaluator prompting."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float


class USCISRetriever:
    """Retrieve relevant USCIS guidance with optional criterion filtering."""

    def __init__(self, config: KnowledgeBaseConfig, embeddings: EmbeddingProvider) -> None:
        self.config = config
        self.embeddings = embeddings
        self.collection = open_collection(config)
        chunks = load_chunks(config)
        self.bm25 = BM25Index(
            (chunk["chunk_id"], chunk["text"], chunk) for chunk in chunks
        )

    def retrieve(
        self,
        query: str,
        *,
        criterion_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if criterion_id is not None and criterion_id not in CRITERION_IDS:
            raise ValueError(f"Unknown criterion_id: {criterion_id}")
        requested_k = top_k or self.config.top_k
        if requested_k < 1:
            raise ValueError("top_k must be at least 1")
        mode = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
        if mode not in {"vector", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be 'vector' or 'hybrid'")
        candidate_k = max(
            requested_k,
            int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "10")),
        )
        where = {"criterion_id": criterion_id} if criterion_id else None
        vector_chunks = self._retrieve_vector(query, where, candidate_k, criterion_id)
        if mode == "vector":
            return vector_chunks[:requested_k]
        keyword_chunks = self.bm25.retrieve(
            query,
            top_k=candidate_k,
            predicate=(
                (lambda metadata: metadata.get("criterion_id") == criterion_id)
                if criterion_id
                else None
            ),
        )
        if criterion_id and not keyword_chunks:
            keyword_chunks = self.bm25.retrieve(query, top_k=candidate_k)
        return self._fuse(
            vector_chunks,
            keyword_chunks,
            query=query,
            criterion_id=criterion_id,
            top_k=requested_k,
        )

    def retrieve_queries(
        self,
        queries: tuple[str, ...] | list[str],
        *,
        criterion_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Fuse results from criterion-specific paraphrase queries."""

        unique: dict[str, RetrievedChunk] = {}
        hit_counts: dict[str, int] = {}
        for query in dict.fromkeys(query for query in queries if query.strip()):
            for chunk in self.retrieve(query, criterion_id=criterion_id, top_k=top_k):
                unique.setdefault(chunk.chunk_id, chunk)
                hit_counts[chunk.chunk_id] = hit_counts.get(chunk.chunk_id, 0) + 1
        ranked = sorted(
            unique.values(),
            key=lambda chunk: (-hit_counts[chunk.chunk_id], chunk.distance, chunk.chunk_id),
        )
        return ranked[: top_k or self.config.top_k]

    def _retrieve_vector(
        self,
        query: str,
        where: dict[str, str] | None,
        candidate_k: int,
        criterion_id: str | None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embeddings.embed_query(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=where,
        )
        chunks = self._to_chunks(result)
        if criterion_id and not chunks:
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
            )
            chunks = self._to_chunks(result)
        return chunks

    @staticmethod
    def _fuse(
        vector_chunks,
        keyword_chunks,
        *,
        query: str = "",
        criterion_id: str | None = None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        rrf_constant = int(os.getenv("HYBRID_RRF_CONSTANT", "60"))
        vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.5"))
        keyword_weight = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
        combined: dict[str, dict[str, Any]] = {}
        for rank, chunk in enumerate(vector_chunks, start=1):
            combined[chunk.chunk_id] = {
                "text": chunk.text,
                "metadata": dict(chunk.metadata),
                "distance": chunk.distance,
                "vector_rank": rank,
                "keyword_rank": None,
                "fused_score": vector_weight / (rrf_constant + rank),
            }
        for rank, chunk in enumerate(keyword_chunks, start=1):
            chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.document_id
            entry = combined.setdefault(
                chunk_id,
                {
                    "text": chunk.text,
                    "metadata": dict(getattr(chunk, "metadata", {})),
                    "distance": 1.0,
                    "vector_rank": None,
                    "keyword_rank": rank,
                    "fused_score": 0.0,
                },
            )
            entry["keyword_rank"] = rank
            entry["fused_score"] += keyword_weight / (rrf_constant + rank)
        query_terms = {
            token.lower()
            for token in query.split()
            if len(token) >= 4
        }
        for entry in combined.values():
            text_terms = set(entry["text"].lower().split())
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            metadata = entry["metadata"]
            criterion_boost = (
                0.003
                if criterion_id and metadata.get("criterion_id") == criterion_id
                else 0.0
            )
            entry["rerank_score"] = entry["fused_score"] + (0.01 * overlap) + criterion_boost
        ranked = sorted(
            combined.items(),
            key=lambda item: (-item[1]["rerank_score"], item[0]),
        )[:top_k]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=entry["text"],
                metadata={
                    **entry["metadata"],
                    "retrieval_method": "hybrid",
                    "vector_rank": entry["vector_rank"],
                    "bm25_rank": entry["keyword_rank"],
                    "fused_score": entry["fused_score"],
                    "rerank_score": entry["rerank_score"],
                },
                distance=entry["distance"],
            )
            for chunk_id, entry in ranked
        ]

    @staticmethod
    def _to_chunks(result: dict[str, Any]) -> list[RetrievedChunk]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(chunk_id=chunk_id, text=text, metadata=metadata, distance=distance)
            for chunk_id, text, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            )
        ]
