"""Compare two saved benchmark reports without making provider calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = (
    ("accuracy", "aggregate.accuracy"),
    ("valid_output_accuracy", "aggregate.valid_output_accuracy"),
    ("structured_output_validity", "aggregate.structured_output_validity"),
    ("evaluation_failure_count", "aggregate.evaluation_failure_count"),
    ("citation_validity", "aggregate.citation_validity"),
    ("hallucinated_evidence_rate", "aggregate.hallucinated_evidence_rate"),
    ("provider_calls", "run_metadata.provider.provider_calls"),
)


def _get_path(payload: dict, path: str):
    value = payload
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def compare_reports(baseline_path: Path, current_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    comparison = {}
    for name, path in METRICS:
        before = _get_path(baseline, path)
        after = _get_path(current, path)
        delta = None if before is None or after is None else round(after - before, 4)
        comparison[name] = {"baseline": before, "current": after, "delta": delta}
    return {
        "baseline": str(baseline_path),
        "current": str(current_path),
        "metrics": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_reports(args.baseline, args.current)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Comparison written to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
