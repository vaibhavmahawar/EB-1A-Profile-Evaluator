"""Streamlit interface for the EB-1A Profile Evaluator."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from eb1a_evaluator.config import default_config
from eb1a_evaluator.embeddings import LocalSentenceTransformerEmbeddings
from eb1a_evaluator.evaluator import (
    CriterionEvaluator,
    OpenAICriterionClient,
    ProfileEvaluator,
    _load_project_env,
)
from eb1a_evaluator.resume import ingest_resume
from eb1a_evaluator.retrieval import USCISRetriever
from eb1a_evaluator.ui import download_payload, format_score, rating_color

USCIS_URL = "https://www.uscis.gov/policy-manual/volume-6-part-f-chapter-2"


@st.cache_resource(show_spinner="Loading the USCIS knowledge base...")
def get_profile_evaluator() -> ProfileEvaluator:
    """Build and cache the local retrieval and configured evaluator stack."""

    _load_project_env()
    provider = os.getenv("LLM_PROVIDER", "groq")
    key_names = {
        "groq": ("LLM_API_KEY", "GROQ_API_KEY"),
        "gemini": ("LLM_API_KEY", "GEMINI_API_KEY"),
    }
    if not any(os.getenv(name) for name in key_names.get(provider, ())):
        raise RuntimeError(f"No API key is configured for the {provider} provider.")
    project_root = Path(__file__).resolve().parent
    config = default_config(project_root)
    if not config.index_dir.exists():
        raise RuntimeError(
            "The USCIS Chroma index is missing. Run scripts/build_uscis_index.py first."
        )
    embeddings = LocalSentenceTransformerEmbeddings(config.embedding_model)
    criterion_evaluator = CriterionEvaluator(
        USCISRetriever(config, embeddings),
        embeddings,
        OpenAICriterionClient(),
    )
    return ProfileEvaluator(criterion_evaluator)


def save_uploaded_resume(uploaded_file) -> Path:
    """Write an uploaded file to a temporary path with its original suffix."""

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise ValueError("Please upload a PDF or TXT resume.")
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        temporary.write(uploaded_file.getvalue())
        return Path(temporary.name)
    finally:
        temporary.close()


def render_assessment(assessment) -> None:
    """Render one criterion assessment."""

    color = rating_color(assessment.rating)
    st.markdown(
        f'<span style="color:{color};font-weight:700;font-size:1.1rem">'
        f"{assessment.rating.value}</span>",
        unsafe_allow_html=True,
    )
    if assessment.resume_evidence:
        st.markdown("**Resume evidence**")
        for evidence in assessment.resume_evidence:
            st.markdown(f"> {evidence}")
    else:
        st.markdown("**Resume evidence**: None found")
    st.markdown(f"**Reasoning**: {assessment.reasoning}")
    if assessment.citation_ids:
        st.markdown("**USCIS citations**")
        for citation in assessment.citation_ids:
            st.code(citation, language="text")
    if assessment.next_steps:
        st.markdown("**Next steps**")
        for step in assessment.next_steps:
            st.markdown(f"- {step}")


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(page_title="EB-1A Profile Evaluator", page_icon="📄", layout="wide")
    st.title("EB-1A Profile Evaluator")
    st.caption(
        "Evidence-grounded prototype using USCIS Policy Manual guidance. "
        "This is informational analysis, not legal advice or a prediction of USCIS approval."
    )
    st.link_button("Open USCIS Policy Manual source", USCIS_URL)

    uploaded_file = st.file_uploader("Upload a synthetic resume", type=["pdf", "txt"])
    if uploaded_file is None:
        st.info("Upload a synthetic PDF or TXT resume to begin.")
        return

    st.write(f"Selected file: `{uploaded_file.name}`")
    if uploaded_file.type == "text/plain":
        preview = uploaded_file.getvalue().decode("utf-8", errors="replace")
        with st.expander("Preview extracted text"):
            st.text(preview[:5000])

    if not st.button("Evaluate profile", type="primary"):
        return

    temporary_path: Path | None = None
    try:
        temporary_path = save_uploaded_resume(uploaded_file)
        resume = ingest_resume(temporary_path)
        if resume.prompt_injection_findings:
            categories = ", ".join(
                finding.category for finding in resume.prompt_injection_findings
            )
            st.warning(
                "Instruction-like text was detected in the uploaded resume and will be "
                f"treated only as untrusted evidence ({categories})."
            )
        evaluator = get_profile_evaluator()
        with st.spinner("Evaluating all 10 EB-1A criteria..."):
            evaluation = evaluator.evaluate(resume)
    except Exception as error:
        st.error(f"Evaluation could not be completed: {error}")
        return
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    st.success("Evaluation completed.")
    st.subheader("Overall summary")
    summary_columns = st.columns(4)
    summary_columns[0].metric("Readiness score", format_score(evaluation.readiness_score))
    summary_columns[1].metric(
        "Strong or Partial",
        str(evaluation.criteria_count_strong_or_partial),
    )
    summary_columns[2].metric("Strong criteria", str(len(evaluation.strongest_criteria)))
    summary_columns[3].metric("Main gaps", str(len(evaluation.main_gaps)))
    st.write(evaluation.candidate_summary)

    if evaluation.strongest_criteria:
        st.markdown(f"**Strongest criteria:** {', '.join(evaluation.strongest_criteria)}")
    if evaluation.main_gaps:
        st.markdown(f"**Main gaps:** {', '.join(evaluation.main_gaps)}")

    st.subheader("Criterion assessments")
    for assessment in evaluation.assessments:
        with st.expander(assessment.criterion_name):
            render_assessment(assessment)

    st.download_button(
        "Download validated JSON",
        data=json.dumps(download_payload(evaluation), indent=2),
        file_name="eb1a_profile_evaluation.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
