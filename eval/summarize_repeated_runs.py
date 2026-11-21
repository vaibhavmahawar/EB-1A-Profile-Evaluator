"""Summarize repeated benchmark reports without making provider calls."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    def values(key: str) -> list[float]:
        return [float(report["aggregate"].get(key, 0)) for report in reports]

    metrics = {}
    for key in (
        "accuracy",
        "valid_output_accuracy",
        "structured_output_validity",
        "evaluation_failure_count",
    ):
        samples = values(key)
        metrics[key] = {
            "runs": samples,
            "mean": round(statistics.mean(samples), 4),
            "min": min(samples),
            "max": max(samples),
            "stdev": round(statistics.stdev(samples), 4) if len(samples) > 1 else 0.0,
        }
    metrics["provider_calls"] = [
        report.get("run_metadata", {}).get("provider", {}).get("provider_calls", 0)
        for report in reports
    ]
    metrics["logical_calls"] = [
        report.get("run_metadata", {}).get("provider", {}).get("logical_calls", 0)
        for report in reports
    ]
    return {"run_count": len(reports), "reports": [str(path) for path in paths], "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.reports)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
