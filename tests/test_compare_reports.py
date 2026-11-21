import json

from eval.compare_reports import compare_reports


def test_compare_reports_calculates_metric_deltas(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(
        json.dumps(
            {
                "aggregate": {
                    "accuracy": 0.6,
                    "valid_output_accuracy": 0.7,
                    "structured_output_validity": 0.8,
                    "evaluation_failure_count": 4,
                    "citation_validity": 1.0,
                    "hallucinated_evidence_rate": 0.1,
                },
                "run_metadata": {"provider": {"provider_calls": 10}},
            }
        ),
        encoding="utf-8",
    )
    current.write_text(
        json.dumps(
            {
                "aggregate": {
                    "accuracy": 0.7,
                    "valid_output_accuracy": 0.8,
                    "structured_output_validity": 0.95,
                    "evaluation_failure_count": 1,
                    "citation_validity": 1.0,
                    "hallucinated_evidence_rate": 0.0,
                },
                "run_metadata": {"provider": {"provider_calls": 3}},
            }
        ),
        encoding="utf-8",
    )

    result = compare_reports(baseline, current)

    assert result["metrics"]["accuracy"]["delta"] == 0.1
    assert result["metrics"]["evaluation_failure_count"]["delta"] == -3
    assert result["metrics"]["provider_calls"]["delta"] == -7
