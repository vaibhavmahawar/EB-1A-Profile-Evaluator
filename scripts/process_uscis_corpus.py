"""Process the existing local USCIS source snapshot without downloading it."""

from pathlib import Path

from eb1a_evaluator.config import default_config
from eb1a_evaluator.corpus import build_processed_corpus

if __name__ == "__main__":
    config = default_config(Path(__file__).resolve().parents[1])
    if not config.source_snapshot.exists():
        raise FileNotFoundError(f"USCIS source snapshot not found: {config.source_snapshot}")
    chunks = build_processed_corpus(config)
    print(f"Processed {len(chunks)} USCIS chunks")
