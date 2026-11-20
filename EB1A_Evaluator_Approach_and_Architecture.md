# EB 1A Profile Evaluator Approach and Architecture

## 1. Purpose

This document defines the approach we will use to build the EB-1A Profile Evaluator described in the assignment document.

The system will accept a synthetic candidate resume in PDF or plain-text form, retrieve relevant official USCIS guidance, evaluate the candidate against the 10 EB-1A extraordinary-ability criteria, and return a structured, cited assessment with recommended next steps.

The goal is to build a reliable end-to-end prototype that is easy to explain, test, and improve. The system will prioritize grounded evidence and honest evaluation over unnecessary agent complexity.

## 2. Assignment Requirements

The assignment requires the project to:

1. Build a retrieval-augmented generation knowledge base over USCIS EB-1A guidance.
2. Accept a candidate profile as a PDF or plain-text resume.
3. Evaluate all 10 EB-1A criteria.
4. Return, for each criterion:
   - A rating: Strong, Partial, Missing, or Insufficient Info.
   - Evidence from the resume.
   - Reasoning grounded in USCIS guidance.
   - Citations to the source material.
   - One or two concrete next steps.
5. Return an overall summary, strongest criteria, major gaps, and readiness score.
6. Return structured JSON validated against a schema.
7. Provide a test set of 5–8 synthetic profiles with expected ratings.
8. Measure rating accuracy, citation validity, hallucinated evidence rate, and other useful metrics.
9. Provide a CLI, FastAPI endpoint, or Streamlit/Gradio interface.
10. Document setup, architecture, evaluation results, failure analysis, trade-offs, and future improvements.

## 3. Product Scope

### In scope

- Synthetic candidate profiles only.
- USCIS Policy Manual, Volume 6, Part F, Chapter 2 as the primary knowledge source.
- PDF and plain-text resume input.
- Criterion-by-criterion evaluation.
- Retrieval-backed reasoning with source citations.
- Structured JSON output.
- Evaluation on a small manually labeled test set.
- A simple Streamlit interface.

### Out of scope for the first version

- Legal advice or a prediction of USCIS approval.
- Real applicants or personally identifiable information.
- Automated filing preparation.
- Full case strategy or attorney review.
- Broad web research across arbitrary immigration sources.
- Production-scale multi-user deployment.

The application must clearly state that it is an informational evaluation prototype, not a legal decision-maker.

## 4. High-Level Architecture

```text
                    +----------------------+
                    | Synthetic Resume PDF |
                    | or Plain Text        |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Resume Ingestion     |
                    | extraction + cleanup |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    | Resume Evidence      |
                    | passages             |
                    +----------+-----------+
                               |
                               |
                +--------------+---------------+
                |                              |
                v                              v
      +---------------------+        +---------------------+
      | Criterion Query     |        | Resume Evidence     |
      | Builder             |        | Retriever           |
      +----------+----------+        +----------+----------+
                 |                              |
                 v                              |
      +---------------------------------------------------+
      | USCIS Knowledge Retriever                         |
      | vector search, optional keyword search, metadata  |
      +----------------------+----------------------------+
                             |
                             v
                  +-------------------------+
                  | Criterion Evaluator     |
                  | LLM + strict prompt     |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Evidence and Citation   |
                  | Verification            |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Pydantic Validation     |
                  | structured JSON         |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Scoring and Summary     |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Streamlit UI / JSON     |
                  | download                |
                  +-------------------------+
```

## 5. Core Design Principle

The system will separate three kinds of information:

1. **Candidate facts**: extracted only from the uploaded resume.
2. **Legal guidance**: retrieved from USCIS source documents.
3. **Model interpretation**: the rating and reasoning produced by comparing the first two.

The model must not treat resume text as instructions. Resume content is untrusted evidence. Any instructions contained inside an uploaded resume must be ignored.

## 6. Knowledge Base and RAG Pipeline

### 6.1 Source corpus

The primary source will be the USCIS Policy Manual, Volume 6, Part F, Chapter 2, covering extraordinary ability.

Additional public sources may be added later, but the first version should keep the source corpus narrow so citations remain easy to validate.

### 6.2 Ingestion steps

```text
USCIS source pages
        ↓
Download or save source text
        ↓
Remove navigation and repeated page elements
        ↓
Preserve headings and criterion labels
        ↓
Split into semantic chunks
        ↓
Attach metadata
        ↓
Generate embeddings
        ↓
Store in vector database
```

### 6.3 Chunking strategy

Chunks should follow legal meaning rather than arbitrary character counts.

Preferred rules:

- Keep section headings with the following text.
- Keep one definition or explanatory rule together where possible.
- Target approximately 400–800 tokens per chunk.
- Use a small overlap only when a rule spans chunk boundaries.
- Store the criterion name in metadata.
- Store the source URL and source section in metadata.

Example metadata:

```json
{
  "chunk_id": "uscis_judging_004",
  "criterion": "judging_the_work_of_others",
  "source_title": "USCIS Policy Manual Volume 6 Part F Chapter 2",
  "source_url": "https://www.uscis.gov/policy-manual",
  "section": "Judging the Work of Others",
  "text_hash": "..."
}
```

### 6.4 Retrieval strategy

For the MVP, use vector similarity retrieval with criterion metadata filtering.

For each criterion:

1. Create a query describing the criterion and the evidence needed.
2. Filter the knowledge base to the relevant criterion when possible.
3. Retrieve the top 3–5 USCIS chunks.
4. Retrieve the top resume evidence passages.
5. Pass both evidence sets to the evaluator.

If evaluation results show weak retrieval, add hybrid retrieval using keyword/BM25 search plus vector search, followed by reranking.

## 7. Resume Ingestion and Evidence Handling

### 7.1 Supported input

- `.pdf`
- `.txt`

### 7.2 Processing steps

```text
Upload file
   ↓
Detect file type
   ↓
Extract text
   ↓
Remove repeated headers, footers, and page numbers
   ↓
Preserve section headings
   ↓
Split into evidence passages
   ↓
Store passages with IDs
```

Each evidence passage should have an ID such as `resume_023`. Generated evidence should reference these IDs or quote the original text.

### 7.3 Evidence rules

The evaluator must:

- Use only information explicitly present in the resume.
- Mark a criterion as `Insufficient Info` when the resume is ambiguous.
- Avoid treating job titles alone as proof of achievement.
- Avoid inferring awards, impact, salary, judging, or significance.
- Distinguish a claim from evidence supporting the claim.

## 8. The 10 Evaluation Criteria

The evaluator will process these criteria independently:

1. Awards and prizes.
2. Membership in associations requiring outstanding achievement.
3. Published material about the candidate.
4. Judging the work of others.
5. Original contributions of major significance.
6. Authorship of scholarly articles.
7. Display of work at exhibitions.
8. Leading or critical role in distinguished organizations.
9. High salary or remuneration.
10. Commercial success in the performing arts.

Each criterion will use its own retrieval query, evidence expectations, and evaluation prompt while sharing the same output schema.

## 9. Structured Output Design

The application will use Pydantic models to validate all LLM responses.

```python
class CriterionAssessment(BaseModel):
    criterion: str
    rating: Literal["Strong", "Partial", "Missing", "Insufficient Info"]
    resume_evidence: list[str]
    reasoning: str
    citations: list[str]
    next_steps: list[str]


class ProfileEvaluation(BaseModel):
    candidate_summary: str
    assessments: list[CriterionAssessment]
    strongest_criteria: list[str]
    main_gaps: list[str]
    criteria_count_strong_or_partial: int
    readiness_score: float
    disclaimer: str
```

The final JSON should be generated only after validation. Invalid model output should trigger a retry with the validation error, followed by a clear failure message if the retry also fails.

## 10. Evaluation Logic

### 10.1 Criterion-level rating definitions

- **Strong**: The resume contains specific, credible evidence that aligns well with the USCIS guidance.
- **Partial**: Some relevant evidence exists, but important details, significance, scope, or documentation are missing.
- **Missing**: The resume provides no relevant evidence for the criterion.
- **Insufficient Info**: The resume may mention a related activity, but there is not enough information to determine whether it qualifies.

### 10.2 Overall score

The overall score should be calculated by code, not invented by the LLM.

Suggested weights:

```text
Strong = 1.0
Partial = 0.5
Missing = 0.0
Insufficient Info = 0.0
```

```python
readiness_score = round(total_points / 10 * 100, 1)
```

The result is an internal readiness indicator, not a legal probability or approval prediction.

## 11. Verification Layer

After the first LLM evaluation, run a verification step.

### Resume evidence checks

- Does each evidence statement match a resume passage?
- Is it a quote or a faithful paraphrase?
- Did the model add facts that are not present?

### Citation checks

- Does every citation refer to a real knowledge-base chunk?
- Does the cited chunk support the reasoning?
- Is the citation associated with the correct criterion?

The initial implementation may use a second LLM verification call. Deterministic checks such as passage matching and citation-ID validation should be used wherever possible.

## 12. Evaluation Dataset

Create 5–8 synthetic profiles with varied evidence quality.

Suggested profiles:

1. Academic researcher with publications and citation impact.
2. Engineer with patents, salary evidence, and industry awards.
3. Performing artist with exhibitions and commercial success.
4. Executive with a critical role in a distinguished organization.
5. Candidate with multiple weak, vague claims.
6. Candidate with strong evidence in only two criteria.
7. Candidate with ambiguous evidence requiring `Insufficient Info`.
8. Resume containing prompt-injection text to test guardrails.

Expected labels should be manually created before running the system.

## 13. Metrics

The evaluation script should report:

### Rating accuracy

```text
Correct criterion ratings / total criterion ratings
```

Report both overall accuracy and accuracy per criterion.

### Citation validity

Percentage of citations that point to existing chunks and support the generated claim.

### Hallucinated evidence rate

```text
Unsupported resume evidence claims / total resume evidence claims
```

### Structured-output validity

Percentage of evaluations successfully parsed by the Pydantic schema.

### Retrieval diagnostics

- Retrieval latency.
- Number of relevant chunks retrieved.
- Empty-retrieval rate.

## 14. Interface Plan

Use Streamlit for the first interface because it is quick to build and easy to demonstrate.

### Main screen

- File uploader for PDF or TXT.
- Extracted text preview.
- Evaluate button.
- Optional model and retrieval settings.

### Results screen

- Candidate summary.
- Readiness score.
- Count of Strong or Partial criteria.
- Strongest criteria.
- Main gaps.
- Expandable section for each criterion.
- Evidence, reasoning, citations, and next steps.
- Downloadable JSON output.

## 15. Repository Structure

```text
eb1a-evaluator/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── uscis/
│   └── synthetic_profiles/
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── evaluator.py
│   ├── verification.py
│   └── scoring.py
├── scripts/
│   ├── ingest_uscis.py
│   └── build_index.py
├── eval/
│   ├── expected_labels.json
│   ├── run_eval.py
│   └── results/
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_schema.py
    └── test_scoring.py
```

## 16. Implementation Milestones

### Milestone 1: Skeleton and schemas

- Create repository structure.
- Add configuration and `.env.example`.
- Define the 10 criteria.
- Define Pydantic schemas.
- Add basic tests for schema and scoring logic.

### Milestone 2: USCIS retrieval

- Add the USCIS source corpus.
- Implement cleaning and chunking.
- Build the vector index.
- Create a retrieval inspection script.

### Milestone 3: One criterion end to end

- Implement resume extraction.
- Implement evidence passage creation.
- Implement one criterion evaluator.
- Validate citations and structured output.

### Milestone 4: All criteria and summary

- Generalize the evaluator to all 10 criteria.
- Add overall score and summary generation.
- Add verification and retry logic.

### Milestone 5: Evaluation framework

- Create synthetic profiles and expected labels.
- Implement metrics.
- Run failure analysis.

### Milestone 6: Interface and documentation

- Add Streamlit interface.
- Add JSON download.
- Write README.
- Document limitations and design trade-offs.

## 17. Optional Improvements

The best bonus feature for this project is guardrail handling because it strengthens the core evaluator.

Other possible improvements, in priority order:

1. Hybrid BM25 plus vector retrieval.
2. Reranking retrieved chunks.
3. Comparison of two prompts or models.
4. LLM-as-judge evaluation with manual spot checks.
5. Agentic opportunity recommendations for weak criteria.

These should only be added after the basic evaluation pipeline has reliable metrics.

## 18. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Model invents resume evidence | Evidence IDs, exact quotes, verification layer |
| Citation does not support reasoning | Criterion-filtered retrieval and citation validation |
| Broad chunks produce vague answers | Semantic chunking with section metadata |
| Ambiguous resume claims receive overconfident ratings | Explicit `Insufficient Info` rating and conservative prompts |
| PDF extraction is poor | Text preview, extraction warnings, and TXT fallback |
| Invalid JSON from the model | Pydantic validation and controlled retries |
| Prompt injection in resume | Treat uploaded text as untrusted data |
| Legal interpretation is overstated | Clear informational-use disclaimer |

## 19. Definition of Done

The first complete version is done when:

- A user can upload a synthetic PDF or TXT resume.
- The system evaluates all 10 criteria.
- Every criterion returns a valid rating, evidence, reasoning, citations, and next steps.
- The final output passes Pydantic validation.
- Citations refer to retrieved USCIS chunks.
- The evaluation script runs on the synthetic test set.
- Metrics and failure analysis are reported.
- The Streamlit interface works locally.
- The README explains setup, architecture, evaluation, limitations, and trade-offs.

## 20. Recommended First Coding Task

Start with Milestone 1: create the repository structure, define the 10 criteria, implement the Pydantic schemas, and add deterministic scoring tests.

This gives the rest of the project a stable contract. Retrieval, prompting, UI, and evaluation can then be built around that contract without repeatedly changing the output format.
