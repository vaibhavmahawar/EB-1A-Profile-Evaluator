"""Local semantic retrieval over extracted resume passages."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from .bm25 import BM25Index
from .embeddings import EmbeddingProvider
from .resume import ResumePassage


@dataclass(frozen=True)
class RetrievedResumePassage:
    passage_id: str
    text: str
    score: float
    retrieval_method: str = "vector"


class ResumeEvidenceRetriever:
    """Rank resume passages against a criterion query."""

    def __init__(self, embeddings: EmbeddingProvider) -> None:
        self.embeddings = embeddings
        self._bm25_key: tuple[str, ...] | None = None
        self._bm25: BM25Index | None = None

    def retrieve(
        self,
        passages: tuple[ResumePassage, ...] | list[ResumePassage],
        query: str,
        *,
        top_k: int = 5,
        min_score: float | None = None,
    ) -> list[RetrievedResumePassage]:
        if not passages:
            raise ValueError("passages must not be empty")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        mode = os.getenv("RETRIEVAL_MODE", "hybrid").lower()
        if mode not in {"vector", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be 'vector' or 'hybrid'")
        vectors = np.asarray(self.embeddings.embed_documents([item.text for item in passages]))
        query_vector = np.asarray(self.embeddings.embed_query(query))
        scores = vectors @ query_vector
        candidate_k = max(top_k, int(os.getenv("RETRIEVAL_CANDIDATE_POOL", "10")))
        order = np.argsort(-scores)[:candidate_k]
        vector_results = [
            RetrievedResumePassage(
                passage_id=passages[index].passage_id,
                text=passages[index].text,
                score=float(scores[index]),
                retrieval_method="vector",
            )
            for index in order
        ]
        if mode == "vector":
            results = vector_results[:top_k]
        else:
            index = self._get_bm25(passages)
            keyword_results = index.retrieve(query, top_k=candidate_k)
            results = self._fuse(vector_results, keyword_results, top_k=top_k)
        if min_score is not None:
            results = [item for item in results if item.score >= min_score]
        return results

    def retrieve_evidence_spans(
        self,
        passages: tuple[ResumePassage, ...] | list[ResumePassage],
        query: str,
        *,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[RetrievedResumePassage]:
        """Retrieve sentence-sized, auditable evidence spans.

        Passage-level retrieval is useful for context, but sentence-level spans
        give the evaluator a smaller and more reliable set of exact quotes.
        Stable IDs retain the originating passage and sentence position.
        """

        spans: list[ResumePassage] = []
        for passage in passages:
            sentences = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+|\n+", passage.text)
                if sentence.strip()
            ]
            for index, sentence in enumerate(sentences):
                digest = passage.content_hash[:8]
                spans.append(
                    ResumePassage(
                        passage_id=f"{passage.passage_id}:span:{index}",
                        text=sentence,
                        source_name=passage.source_name,
                        passage_index=passage.passage_index,
                        content_hash=f"{digest}-{index}",
                    )
                )
        if not spans:
            return []
        results = self.retrieve(spans, query, top_k=top_k, min_score=min_score)
        return results

    def _get_bm25(self, passages: tuple[ResumePassage, ...] | list[ResumePassage]) -> BM25Index:
        key = tuple(item.content_hash for item in passages)
        if self._bm25 is None or self._bm25_key != key:
            self._bm25 = BM25Index(
                (item.passage_id, item.text, {}) for item in passages
            )
            self._bm25_key = key
        return self._bm25

    @staticmethod
    def _fuse(vector_results, keyword_results, *, top_k: int) -> list[RetrievedResumePassage]:
        rrf_constant = int(os.getenv("HYBRID_RRF_CONSTANT", "60"))
        vector_weight = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.5"))
        keyword_weight = float(os.getenv("HYBRID_BM25_WEIGHT", "0.5"))
        combined: dict[str, dict] = {}
        for rank, item in enumerate(vector_results, start=1):
            combined[item.passage_id] = {
                "text": item.text,
                "vector_score": item.score,
                "fused_score": vector_weight / (rrf_constant + rank),
                "methods": ["vector"],
            }
        for rank, item in enumerate(keyword_results, start=1):
            entry = combined.setdefault(
                item.document_id,
                {
                    "text": item.text,
                    "vector_score": None,
                    "fused_score": 0.0,
                    "methods": [],
                },
            )
            entry["fused_score"] += keyword_weight / (rrf_constant + rank)
            entry["methods"].append("bm25")
        for entry in combined.values():
            vector_score = entry["vector_score"] or 0.0
            entry["rerank_score"] = entry["fused_score"] + (0.01 * vector_score)
        ranked = sorted(
            combined.items(),
            key=lambda item: (-item[1]["rerank_score"], item[0]),
        )[:top_k]
        return [
            RetrievedResumePassage(
                passage_id=passage_id,
                text=entry["text"],
                score=(
                    entry["vector_score"]
                    if entry["vector_score"] is not None
                    else entry["fused_score"]
                ),
                retrieval_method="+".join(entry["methods"]),
            )
            for passage_id, entry in ranked
        ]
