"""Run the opt-in LLM-as-judge evaluation over a completed live report."""

from __future__ import annotations

import argparse
from pathlib import Path

from eb1a_evaluator.judge import run_judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval/results/live_full_profile_report.json"),
        help="completed live evaluator report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/judge_report.json"),
        help="judge report output path",
    )
    parser.add_argument("--limit", type=int, help="judge only the first N profiles")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = args.input if args.input.is_absolute() else root / args.input
    output_path = args.output if args.output.is_absolute() else root / args.output
    run_judge(
        input_path,
        output_path,
        limit=args.limit,
        no_cache=args.no_cache,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
