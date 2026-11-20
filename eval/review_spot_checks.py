"""Summarize completed human spot-check annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

RATINGS = {"Strong", "Partial", "Missing", "Insufficient Info"}
SUPPORT = {"Supported", "Partially Supported", "Unsupported", "Not Applicable"}


def summarize(input_path: Path, output_path: Path) -> dict:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    checks = payload.get("checks", [])
    if not checks:
        raise ValueError("Spot-check template contains no checks")
    missing = []
    for check in checks:
        if check.get("human_rating") not in RATINGS:
            missing.append(f"{check.get('review_id')}: human_rating")
        for field in ("human_evidence_support", "human_citation_support"):
            if check.get(field) not in SUPPORT:
                missing.append(f"{check.get('review_id')}: {field}")
        if not isinstance(check.get("human_reasoning_grounded"), bool):
            missing.append(f"{check.get('review_id')}: human_reasoning_grounded")
    if missing:
        raise ValueError("Incomplete spot checks: " + ", ".join(missing))

    human_matches = [check["human_rating"] == check["expected_rating"] for check in checks]
    judge_matches = [
        check["automated_judgment"]["rating_decision"]
        == ("Correct" if check["predicted_rating"] == check["expected_rating"] else "Incorrect")
        for check in checks
    ]
    result = {
        "checked_count": len(checks),
        "human_label_accuracy": round(sum(human_matches) / len(checks), 4),
        "automated_judge_agreement": round(sum(judge_matches) / len(checks), 4),
        "human_evidence_support": dict(
            Counter(check["human_evidence_support"] for check in checks)
        ),
        "human_citation_support": dict(
            Counter(check["human_citation_support"] for check in checks)
        ),
        "human_reasoning_grounded_rate": round(
            sum(check["human_reasoning_grounded"] for check in checks) / len(checks), 4
        ),
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Spot-check summary written to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/results/manual_spot_check_template.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/manual_spot_check_summary.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    summarize(input_path, output_path)


if __name__ == "__main__":
    main()
