"""Small deterministic BM25 index for local hybrid retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable

_TOKEN_PATTERN = re.compile(r"[\w][\w'-]*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for keyword retrieval."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_PATTERN.findall(normalized)


@dataclass(frozen=True)
class BM25Result:
    """A ranked BM25 document."""

    document_id: str
    text: str
    score: float
    metadata: dict[str, Any]


class BM25Index:
    """In-memory BM25 index with deterministic filtering and ordering."""

    def __init__(
        self,
        documents: Iterable[tuple[str, str, dict[str, Any] | None]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")
        self.k1 = k1
        self.b = b
        self._documents = [
            (document_id, text, metadata or {}, tokenize(text))
            for document_id, text, metadata in documents
        ]
        if not self._documents:
            raise ValueError("BM25 cannot index an empty document collection")
        self._document_frequency: dict[str, int] = {}
        for _, _, _, tokens in self._documents:
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1
        self._average_length = sum(len(item[3]) for item in self._documents) / len(self._documents)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[BM25Result]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        candidates = []
        for document_id, text, metadata, tokens in self._documents:
            if predicate is not None and not predicate(metadata):
                continue
            score = self._score(query_tokens, tokens)
            if score > 0:
                candidates.append(BM25Result(document_id, text, score, metadata))
        candidates.sort(key=lambda item: (-item.score, item.document_id))
        return candidates[:top_k]

    def _score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        term_frequency: dict[str, int] = {}
        for token in document_tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        document_length = len(document_tokens)
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency.get(token, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequency.get(token, 0)
            idf = math.log(
                1 + (len(self._documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(1, self._average_length)
            )
            score += idf * frequency * (self.k1 + 1) / denominator
        return score
