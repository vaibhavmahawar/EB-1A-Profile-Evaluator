"""Compare configured model and prompt strategies using identical benchmark settings."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from run_eval import run_live

from eb1a_evaluator.benchmark_config import BenchmarkConfig
from eb1a_evaluator.response_cache import failure_category


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def _winner(strategies: list[dict], metric: str, *, reverse: bool) -> str | None:
    valid = [item for item in strategies if item.get("status") == "completed"]
    values = [item["aggregate"].get(metric) for item in valid]
    if not valid or any(value is None for value in values) or len(set(values)) == 1:
        return None
    best = max(values) if reverse else min(values)
    return next(item["name"] for item in valid if item["aggregate"].get(metric) == best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("eval/comparison_config.json"),
        help="comparison strategy JSON configuration",
    )
    parser.add_argument("--limit", type=int, help="evaluate only the first N profiles")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    comparison_path = args.config if args.config.is_absolute() else root / args.config
    base_config = BenchmarkConfig.from_file(root / "eval" / "benchmark_config.json")
    definitions = json.loads(comparison_path.read_text(encoding="utf-8"))["strategies"]
    results = []

    for index, definition in enumerate(definitions):
        name = str(definition["name"])
        strategy_config = replace(
            base_config,
            model=str(definition.get("model", base_config.model)),
            prompt_version=str(
                definition.get("prompt_version", base_config.prompt_version)
            ),
        )
        try:
            report = run_live(
                root / "data" / "synthetic_profiles",
                root / "eval" / "results",
                limit=args.limit,
                benchmark_config=strategy_config,
                no_cache=args.no_cache,
                refresh_cache=args.refresh_cache and index == 0,
                report_name=f"strategy_{_safe_name(name)}.json",
                strategy_name=name,
            )
            results.append(
                {
                    "name": name,
                    "model": strategy_config.model,
                    "prompt_version": strategy_config.prompt_version,
                    "status": "completed",
                    "aggregate": report["aggregate"],
                    "run_metadata": report["run_metadata"],
                }
            )
        except Exception as error:
            results.append(
                {
                    "name": name,
                    "model": strategy_config.model,
                    "prompt_version": strategy_config.prompt_version,
                    "status": "failed",
                    "failure_category": failure_category(error),
                    "failure_type": type(error).__name__,
                }
            )

    comparison = {
        "mode": "strategy_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": results,
        "winners": {
            "accuracy": _winner(results, "accuracy", reverse=True),
            "structured_output_validity": _winner(
                results, "structured_output_validity", reverse=True
            ),
            "hallucinated_evidence_rate": _winner(
                results, "hallucinated_evidence_rate", reverse=False
            ),
            "evaluation_failure_count": _winner(
                results, "evaluation_failure_count", reverse=False
            ),
        },
    }
    output_path = root / "eval" / "results" / "strategy_comparison.json"
    output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
