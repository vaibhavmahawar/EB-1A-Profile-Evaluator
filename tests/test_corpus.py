from pathlib import Path

from eb1a_evaluator.config import KnowledgeBaseConfig
from eb1a_evaluator.corpus import chunk_text, clean_html, infer_criterion


def test_clean_html_removes_navigation_and_scripts() -> None:
    html = "<nav>Menu</nav><main><h1>Guidance</h1><p>Useful text.</p></main><script>bad()</script>"
    cleaned = clean_html(html)
    assert "Menu" not in cleaned
    assert "bad()" not in cleaned
    assert "Useful text." in cleaned


def test_chunking_is_deterministic_and_has_metadata(tmp_path: Path) -> None:
    config = KnowledgeBaseConfig(project_root=tmp_path, chunk_size_words=8, chunk_overlap_words=2)
    text = "Awards and prizes\n\n" + "A useful paragraph with several words.\n\n" * 5
    first = chunk_text(text, config)
    second = chunk_text(text, config)
    assert first == second
    assert first
    assert all(chunk["chunk_id"].startswith("uscis_") for chunk in first)
    assert all(chunk["content_hash"] for chunk in first)


def test_criterion_inference_handles_known_and_general_text() -> None:
    assert infer_criterion("The judging the work of others criterion") == "judging_work_of_others"
    assert infer_criterion("General policy information") == "general"
