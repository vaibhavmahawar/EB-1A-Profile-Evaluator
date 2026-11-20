"""Build the persistent Chroma index for the local USCIS corpus."""

from pathlib import Path

from eb1a_evaluator.config import default_config
from eb1a_evaluator.corpus import build_processed_corpus, load_chunks
from eb1a_evaluator.embeddings import LocalSentenceTransformerEmbeddings
from eb1a_evaluator.vector_store import build_index

if __name__ == "__main__":
    config = default_config(Path(__file__).resolve().parents[1])
    if not config.processed_json.exists():
        build_processed_corpus(config)
    chunks = load_chunks(config)
    embeddings = LocalSentenceTransformerEmbeddings(config.embedding_model)
    manifest = build_index(config, chunks, embeddings)
    print(f"Built Chroma index: {manifest}")
