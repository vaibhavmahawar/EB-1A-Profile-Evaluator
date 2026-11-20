"""Create a reproducible manual spot-check template from a judge report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def create_template(input_path: Path, output_path: Path, per_rating: int = 3) -> dict:
    report = json.loads(input_path.read_text(encoding="utf-8"))
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in report.get("records", []):
        if record.get("status") != "completed":
            continue
        buckets[record["expected_rating"]].append(record)

    selected = []
    for rating in ("Strong", "Partial", "Missing", "Insufficient Info"):
        selected.extend(buckets[rating][:per_rating])
    injection_record = next(
        (
            record
            for record in report.get("records", [])
            if record.get("profile_id") == "prompt_injection_candidate"
            and record.get("status") == "completed"
        ),
        None,
    )
    if injection_record and injection_record not in selected:
        selected[-1:] = [injection_record]

    checks = []
    for index, record in enumerate(selected, start=1):
        checks.append(
            {
                "review_id": f"spot-{index:03d}",
                "profile_id": record["profile_id"],
                "criterion": record["criterion"],
                "expected_rating": record["expected_rating"],
                "predicted_rating": record["predicted_rating"],
                "assessment": record["assessment"],
                "automated_judgment": record["judgment"],
                "human_rating": None,
                "human_evidence_support": None,
                "human_citation_support": None,
                "human_reasoning_grounded": None,
                "notes": "",
            }
        )
    result = {
        "instructions": (
            "Fill every human_* field after reviewing the supplied assessment and source material. "
            "Use the same rating and support labels as the automated judge."
        ),
        "source_report": str(input_path),
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Spot-check template written to {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/results/judge_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/manual_spot_check_template.json"),
    )
    parser.add_argument("--per-rating", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    create_template(input_path, output_path, per_rating=args.per_rating)


if __name__ == "__main__":
    main()
