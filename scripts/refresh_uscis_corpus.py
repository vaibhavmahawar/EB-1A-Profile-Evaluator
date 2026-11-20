"""Refresh and process the official USCIS EB-1A source snapshot."""

from pathlib import Path

from eb1a_evaluator.config import default_config
from eb1a_evaluator.corpus import build_processed_corpus, download_source

if __name__ == "__main__":
    config = default_config(Path(__file__).resolve().parents[1])
    download_source(config, replace=True)
    chunks = build_processed_corpus(config)
    print(f"Processed {len(chunks)} USCIS chunks")
