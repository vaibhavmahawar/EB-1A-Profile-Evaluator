"""Create a transparent per-criterion error audit from a saved live report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def audit(report_path: Path, labels_path: Path, output_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    records = []
    error_types = Counter()
    for profile in report.get("profiles", []):
        expected_profile = labels.get(profile["profile_id"], {})
        for item in profile.get("criteria", []):
            criterion = item["criterion"]
            expected = expected_profile.get(criterion)
            predicted = item["assessment"].get("rating")
            if expected == predicted:
                continue
            category = "rating_mismatch"
            if not item.get("structured_output_valid", True):
                category = "structured_failure"
            elif predicted == "Insufficient Info":
                category = "conservative_or_missing_support"
            elif expected == "Strong" and predicted in {"Partial", "Missing"}:
                category = "strong_underrating"
            elif expected == "Missing" and predicted in {"Partial", "Strong"}:
                category = "unsupported_positive"
            error_types[category] += 1
            records.append({
                "profile_id": profile["profile_id"],
                "criterion": criterion,
                "expected": expected,
                "predicted": predicted,
                "category": category,
                "resume_evidence": item["assessment"].get("resume_evidence", []),
                "reasoning": item["assessment"].get("reasoning", ""),
            })
    result = {
        "source_report": str(report_path),
        "mismatch_count": len(records),
        "error_categories": dict(error_types),
        "mismatches_by_criterion": dict(Counter(item["criterion"] for item in records)),
        "records": records,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/synthetic_profiles/expected_labels.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("eval/results/error_audit.json"))
    args = parser.parse_args()
    result = audit(args.report, args.labels, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
