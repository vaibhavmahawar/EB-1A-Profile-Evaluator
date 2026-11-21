# EB 1A Profile Evaluator

## Docker deployment

Docker is an optional deployment path for running the Streamlit interface in an isolated environment. It does not change the evaluator, retrieval logic, or output schema.

Prerequisites:

- Docker Desktop installed and running
- A local `.env` file containing the selected provider configuration

Build and start the application from PowerShell:

```powershell
Set-Location -LiteralPath "C:\Users\ASUS\Downloads\EB-1A Profile Evaluator"
docker compose up --build
```

Open `http://localhost:8501` after the container starts. Compose loads API credentials from `.env`; credentials are not copied into the image or committed to Git.

The Compose setup persists these generated artifacts in named Docker volumes:

- `eb1a-index`: USCIS vector index at `/app/data/index`
- `eb1a-eval-cache`: provider response cache at `/app/eval/cache`
- `eb1a-eval-results`: benchmark reports at `/app/eval/results`

To build the USCIS index inside the container before starting the UI:

```powershell
docker compose run --rm evaluator python scripts/build_uscis_index.py
docker compose up
```

To run the offline benchmark inside the container:

```powershell
docker compose run --rm evaluator python eval/run_eval.py --mode heuristic
```

Stop the service with `Ctrl+C`, or run:

```powershell
docker compose down
```

This repository contains a prototype that evaluates synthetic candidate profiles against the 10 EB-1A extraordinary-ability criteria using grounded USCIS guidance.

## Milestone 1

Milestone 1 establishes the project foundation:

- A Python package using a `src/` layout.
- Strict Pydantic models for evaluation output.
- Stable identifiers for all 10 criteria.
- Deterministic readiness scoring.
- Pytest coverage for the domain contract and scoring behavior.

## Setup

```text
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

Run tests:

```text
pytest
```

For automatic CLI and benchmark runs, create a local ignored `.env` file from `.env.example` and configure the selected provider. The file is ignored by Git and the key should never be committed or shared.

The evaluator supports Groq and Gemini through the OpenAI Python client. For Gemini, use the OpenAI-compatible endpoint and Chat Completions structured parsing:

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
LLM_MODEL=gemini-3.5-flash-lite
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

For Groq, use `LLM_PROVIDER=groq`, `GROQ_API_KEY`, and a Groq model. The provider-neutral `LLM_*` variables take precedence while the older `GROQ_*` variables remain supported for compatibility. Gemini’s compatibility endpoint and structured-output support are documented by Google in the [Gemini OpenAI compatibility guide](https://aistudio.google.com/docs/openai).

## USCIS knowledge base

The baseline USCIS Policy Manual PDF is stored under `data/uscis/raw/`. Process the existing local snapshot with:

```text
$env:PYTHONPATH = "src"
python scripts/process_uscis_corpus.py
```

To explicitly refresh the snapshot from the official USCIS source and process it:

```text
$env:PYTHONPATH = "src"
python scripts/refresh_uscis_corpus.py
```

The refresh command replaces the local snapshot only when explicitly run. The processed chunks are written to `data/uscis/processed/`.

## First criterion evaluator

The first end-to-end evaluator handles the `awards_and_prizes` criterion. Set the API key in the active PowerShell session, then run:

```text
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:LLM_MODEL = "openai/gpt-oss-20b"
$env:PYTHONPATH = "src"
python scripts/evaluate_awards.py path/to/synthetic_profile.txt
```

The evaluator returns validated JSON containing the rating, resume evidence, USCIS citation IDs, reasoning, and next steps. Resume text is treated as untrusted data and is never treated as an instruction source.

To evaluate all ten criteria:

```text
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:LLM_MODEL = "openai/gpt-oss-20b"
$env:PYTHONPATH = "src"
python scripts/evaluate_profile.py path/to/synthetic_profile.txt
```

The complete evaluator runs criteria sequentially and continues with `Insufficient Info` if an individual criterion fails.

## Streamlit interface

Start the local interface from PowerShell:

```text
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-groq-key"
$env:LLM_MODEL = "openai/gpt-oss-20b"
$env:PYTHONPATH = "C:\Users\ASUS\Downloads\EB-1A Profile Evaluator\src"
python -m streamlit run "C:\Users\ASUS\Downloads\EB-1A Profile Evaluator\app.py"
```

Upload a synthetic PDF or TXT resume, run the evaluation, inspect all ten criteria, and download the validated JSON result.

## Evaluation benchmark

The repository includes eight synthetic profiles with expected labels for all ten criteria. Run the offline awards baseline without an API key:

```powershell
$env:PYTHONPATH = "src"
python eval/run_eval.py --mode heuristic
```

Run the complete retrieval and provider evaluation across all eight profiles. Groq is the primary acceptance baseline:

```powershell
$env:LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-groq-key"
$env:LLM_MODEL = "openai/gpt-oss-20b"
$env:GROQ_REQUEST_DELAY_SECONDS = "5"
$env:GROQ_RETRY_BACKOFF_SECONDS = "20"
$env:GROQ_BATCH_PROFILE = "0"
$env:GROQ_BATCH_SIZE = "2"
$env:RESUME_RETRIEVAL_TOP_K = "3"
$env:RESUME_RETRIEVAL_MIN_SCORE = "0.15"
$env:USCIS_RETRIEVAL_TOP_K = "3"
$env:PYTHONPATH = "src"
python eval/run_eval.py --mode live
```

Preserve a run instead of overwriting the default report:

```powershell
python eval/run_eval.py --mode live --no-cache --report-name live_groq_accuracy_run.json
```

For a lower-cost smoke test, evaluate one profile first:

```powershell
python eval/run_eval.py --mode live --limit 1
```

The benchmark uses the versioned settings in `eval/benchmark_config.json` by default. Successful structured provider responses are cached locally in `eval/cache/` so repeated runs do not consume additional provider tokens. Use `--no-cache` to bypass replay or `--refresh-cache` to clear the cache before a run:

```powershell
python eval/run_eval.py --mode live --limit 1 --no-cache
python eval/run_eval.py --mode live --refresh-cache
```

The cache stores parsed responses and diagnostics, never API keys or raw prompts. Reports include the configuration hash, cache hits and misses, provider-call count, latency metadata, and normalized failure categories such as `quota_exhausted`, `rate_limit`, `structured_output`, and `evidence_validation`.

Provider-output recovery is bounded and schema-preserving. If structured parsing is rejected, the client uses JSON mode, extracts a JSON object from harmless Markdown or surrounding text, removes only safe trailing commas, validates the result with Pydantic, and makes at most one repair request. Missing fields, invalid ratings, unknown fields, and provider quota errors are never silently invented or rewritten. Offline fixtures cover valid JSON, wrapped JSON, malformed JSON, schema failures, and one-repair behavior.

Resume guardrails run before embedding or provider calls. They reject empty, too-short, unreadable, or garbled input, detect instruction-like text in uploaded resumes, preserve the original text as evidence, and treat detected instructions as untrusted data. Evidence quotes and USCIS citation IDs are validated before scoring, while provider and validation failures are reported separately from model accuracy.

Retrieval defaults to a local hybrid of Chroma vector search and an in-memory BM25 keyword index. Reciprocal Rank Fusion combines both lists without adding FAISS or another vector database. Use `RETRIEVAL_MODE=vector` for the vector-only baseline. Compare both modes offline, without an API call:

```powershell
$env:PYTHONPATH = "src"
python eval/compare_retrieval.py
```

The comparison report is written to `eval/results/retrieval_comparison.json` and includes empty-retrieval rate, criterion-hit rate, latency, and top-k overlap. Hybrid retrieval is the default: BM25 and Chroma vector results are fused with Reciprocal Rank Fusion, then ranked using criterion metadata and lexical overlap. Criterion query variants are evaluated through the same hybrid path.

The live report evaluates all 80 profile-criterion pairs and reports rating accuracy, valid-output accuracy, accuracy by criterion, citation-ID validity, unsupported resume evidence rate, structured-output validity, and retrieval diagnostics. The default criterion-by-criterion mode uses the configured provider and model. Set `LLM_BATCH_PROFILE=1` to opt into grouped structured requests, with `LLM_BATCH_SIZE` controlling the group size; grouped mode retains criterion-specific retrieval and verification but may be slower for reasoning-heavy models. Reports are written to `eval/results/`. See `eval/FAILURE_ANALYSIS.md` for the completed benchmark results and limitations.

### Prompt and model strategy comparison

Phase 3 adds versioned evaluation strategies so prompt changes and model changes can be measured under the same benchmark settings. The current strategies are:

- `criterion-v3`: the existing conservative baseline.
- `criterion-v4-evidence-first`: explicitly separates exact resume evidence from requirement matching and is stricter about unsupported inferences.

The production default remains `criterion-v3` until a live comparison demonstrates a clear improvement. Configure strategies in `eval/comparison_config.json`, then run the comparison when Groq quota is available:

```powershell
$env:GROQ_API_KEY = "your-groq-key"
$env:PYTHONPATH = "src"
python eval/compare_strategies.py --limit 1
```

Each strategy receives an isolated report under `eval/results/strategy_*.json`; the summary is written to `eval/results/strategy_comparison.json`. It compares accuracy, structured-output validity, hallucinated-evidence rate, and provider/evaluation failures. The response-cache key includes the model and prompt version, so cached results cannot silently cross-contaminate strategies. Reports also include provider-call latency and token-usage diagnostics. Use `--no-cache` only when intentionally measuring fresh provider calls.

### LLM-as-judge and manual spot checks

Phase 4 is an opt-in quality-control layer. It reviews an existing live report against the manually assigned labels in `data/synthetic_profiles/expected_labels.json`; it does not rerun or change the original evaluator. The judge checks rating agreement, resume-evidence support, USCIS citation support, and reasoning grounding using the same local retrieval context.

Run it after a live report exists:

```powershell
$env:PYTHONPATH = "src"
python eval/run_judge.py --input eval/results/live_full_profile_report.json
```

Use `--limit 1` for a low-cost smoke test. The judge uses `judge-v1` cache keys, so its responses remain separate from evaluator and prompt-strategy responses. The report is written to `eval/results/judge_report.json`. It includes judge decision accuracy, manual-label match rate, grounding distributions, failures, and per-criterion metrics.

Create a reproducible human review sample from the judge report:

```powershell
python eval/create_spot_checks.py
```

Fill the `human_*` fields in `eval/results/manual_spot_check_template.json`, then summarize the review:

```powershell
python eval/review_spot_checks.py
```

The manual review summary is written to `eval/results/manual_spot_check_summary.json`. The judge and spot-check outputs are diagnostic evidence for quality analysis, not legal advice and not production eligibility decisions.

## Architecture and design decisions

The pipeline extracts PDF or TXT resume text into traceable passages and sentence-level evidence spans, retrieves criterion-specific USCIS chunks with hybrid Chroma vector plus local BM25 retrieval, and sends only compact selected evidence and USCIS guidance to the configured structured-output provider. Pydantic validation, citation-ID checks, and exact resume-evidence checks run before deterministic scoring produces the final summary. The Streamlit app calls this same pipeline and exposes the validated JSON result.

The prototype uses local embeddings and Chroma to keep retrieval inspectable and inexpensive, an OpenAI-compatible provider client to keep the Groq/Gemini boundary small, and an explicit `Insufficient Info` fallback to avoid overconfident ratings when a criterion call fails. Synthetic profiles are used for evaluation and no real personal data is required.

### Best recorded benchmark result

The table below reports the best single completed Groq run, rather than selecting
different metrics from different runs. It used eight synthetic profiles, all ten
criteria, compact profile batching, `openai/gpt-oss-20b`, and temperature `0`.
The result is a prototype measurement, not a legal conclusion or a general-world
accuracy claim.

| Metric | Best single run |
| --- | ---: |
| Evaluation scope | 8 profiles / 80 criterion ratings |
| Rating accuracy | 58/80 (72.5%) |
| Valid-output accuracy | 58/79 (73.42%) |
| Structured-output validity | 79/80 (98.75%) |
| Evaluation failures | 1/80 |
| Citation validity | 100% |
| Grounded/unsupported evidence | 0% hallucinated evidence |
| Retrieval metrics | Empty retrieval rate: 0%; latency: 0.436s average |
| Prompt/model comparison | Supported by `eval/compare_strategies.py` |
| Variance reporting | 3-run mean: 70.83%, range: 68.75%–72.5% |

These are prototype measurements from synthetic profiles, not legal conclusions
or general-world accuracy claims. The benchmark counts all 80 criterion pairs and
conservatively reports failed calls as failures with a fallback rating.

The best-run report is `eval/results/variance_run_1.json`. Repeated-run stability is
reported separately in `eval/results/repeated_run_summary_latest.json`; it should
not be mixed into the best-run comparison table.

## Improvements with more time

- Add semantic citation-support verification and human spot checks.
- Compare hybrid BM25 plus vector retrieval or reranking against the current baseline.
- Expand the synthetic dataset with more borderline cases and document-quality failures.
- Compare multiple models or prompt variants for quality, latency, and cost.

The system is an informational prototype, not legal advice or a prediction of USCIS approval.
