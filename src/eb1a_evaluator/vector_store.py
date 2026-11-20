"""Chroma-backed local index for USCIS chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb

from .config import KnowledgeBaseConfig
from .embeddings import EmbeddingProvider

COLLECTION_NAME = "uscis_eb1a_guidance"


def build_index(
    config: KnowledgeBaseConfig,
    chunks: list[dict[str, Any]],
    embeddings: EmbeddingProvider,
    *,
    replace: bool = True,
) -> Path:
    """Build or replace the persistent Chroma collection."""

    if not chunks:
        raise ValueError("Cannot index an empty chunk list")
    config.index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.index_dir))
    if replace:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME)
    texts = [chunk["text"] for chunk in chunks]
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings.embed_documents(texts),
    metadatas=[
        {key: value for key, value in chunk.items() if key != "text"} for chunk in chunks
    ],
    )
    manifest = config.index_dir / "index_manifest.json"
    manifest.write_text(
        json.dumps(
            {"collection": COLLECTION_NAME, "embedding_model": embeddings.model_name},
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def open_collection(config: KnowledgeBaseConfig):
    """Open the persistent USCIS collection."""

    client = chromadb.PersistentClient(path=str(config.index_dir))
    return client.get_collection(COLLECTION_NAME)
