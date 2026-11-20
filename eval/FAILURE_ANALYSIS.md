# Evaluation Failure Analysis

The benchmark contains eight synthetic profiles and manually assigned labels for all ten EB-1A criteria. The live evaluation command runs the same retrieval, prompting, verification, and scoring path used by the application:

```powershell
$env:PYTHONPATH = "src"
python eval/run_eval.py --mode live
```

The generated report is written to `eval/results/live_full_profile_report.json`. It reports exact rating accuracy overall and by criterion, citation-ID validity, unsupported resume evidence rate, structured-output validity, and retrieval diagnostics.

## Completed benchmark result

The completed run used `openai/gpt-oss-20b` with five seconds between requests and eight synthetic profiles. All 80 profile-criterion slots were processed.

| Metric | Result |
|---|---:|
| Rating accuracy | 39/80 (48.75%) |
| Structured-output validity | 51/80 (63.75%) |
| Citation-ID validity | 100% |
| Hallucinated evidence rate | 8.57% |
| Empty retrieval rate | 0% |
| Rate-limit failures | 3 |
| Validation failures | 18 |
| Other verification/provider failures | 8 |

Accuracy by criterion was highest for high salary (87.5%), followed by commercial success in the performing arts and display of work (62.5% each). The weakest criteria were awards and prizes and leading or critical role (25% each), followed by original contributions, judging work of others, and scholarly articles (37.5% each).

The benchmark completed technically, but the rating accuracy is affected by 29 criterion-level failures that the application conservatively converted to `Insufficient Info`. The results should therefore be presented as an honest prototype baseline rather than a model-quality claim.

## Known failure modes

- Exact rating accuracy is strict. A conservative `Insufficient Info` result can be reasonable even when it does not match a manually assigned `Partial` label.
- Citation validity currently verifies that citation IDs exist in the processed USCIS corpus. It does not independently prove that every cited paragraph semantically supports every sentence in the reasoning.
- Resume evidence verification is deterministic substring matching. Faithful paraphrases that are not exact excerpts may be rejected, while a short matching phrase can pass without proving the entire claim.
- The profile evaluator converts an individual criterion failure into `Insufficient Info` so one provider or retrieval error does not prevent the remaining criteria from being returned.
- The smaller `openai/gpt-oss-20b` model produced some schema or verification failures under the full 80-call run. Request pacing reduced rate-limit failures but did not eliminate all provider limitations.
- Eight synthetic profiles are useful for regression testing but are too small to establish general-world accuracy. Labels should be expanded and manually reviewed before making benchmark claims.
- The default offline `heuristic` mode is intentionally limited to the awards criterion. The `live` mode is the complete evaluation path and requires the configured Groq API key.

## Current implementation status

The evaluator now uses criterion query variants, sentence-level resume evidence spans, compact prompts, hybrid BM25 plus vector retrieval, deterministic reranking, bounded provider retry, and separate valid-output accuracy reporting. The public assessment schema is unchanged. The Groq baseline remains the primary comparison target because the current Gemini run has substantially lower structured-output validity.

## Improvement priorities

1. Add semantic citation-support checks with human spot review.
2. Expand the dataset with more borderline and document-quality cases.
3. Compare hybrid retrieval strategies and prompt variants using the same labels.
4. Use the Phase 0 response cache to track latency, provider failures, token cost, and replayability across repeated runs.
