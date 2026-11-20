"""Evaluate all ten EB-1A criteria for one local resume."""

import json
import sys
from pathlib import Path

from eb1a_evaluator.config import default_config
from eb1a_evaluator.embeddings import LocalSentenceTransformerEmbeddings
from eb1a_evaluator.evaluator import CriterionEvaluator, OpenAICriterionClient, ProfileEvaluator
from eb1a_evaluator.resume import ingest_resume
from eb1a_evaluator.retrieval import USCISRetriever

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/evaluate_profile.py path/to/resume.txt")
    config = default_config(Path(__file__).resolve().parents[1])
    embeddings = LocalSentenceTransformerEmbeddings(config.embedding_model)
    criterion_evaluator = CriterionEvaluator(
        USCISRetriever(config, embeddings),
        embeddings,
        OpenAICriterionClient(),
    )
    evaluation = ProfileEvaluator(criterion_evaluator).evaluate(ingest_resume(sys.argv[1]))
    print(json.dumps(evaluation.model_dump(mode="json"), indent=2))
