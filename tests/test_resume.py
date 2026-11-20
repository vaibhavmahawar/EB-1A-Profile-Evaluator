from pathlib import Path

import pytest

from eb1a_evaluator.resume import extract_resume_text, ingest_resume, make_passages


def test_txt_resume_ingestion(tmp_path: Path) -> None:
    path = tmp_path / "candidate.txt"
    path.write_text("Won a nationally recognized engineering award in 2024.", encoding="utf-8")
    document = ingest_resume(path)
    assert document.source_type == "txt"
    assert document.passages[0].passage_id.startswith("resume_0000_")
    assert "nationally recognized" in document.raw_text


def test_unsupported_resume_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "candidate.docx"
    path.write_text("not supported", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported resume type"):
        extract_resume_text(path)


def test_empty_resume_is_rejected() -> None:
    with pytest.raises(ValueError, match="no extractable text"):
        make_passages("   ", source_name="empty.txt")


def test_passages_are_deterministic_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(20))
    passages = make_passages(text, source_name="candidate.txt", target_words=10, overlap_words=2)
    assert passages[0].text.split()[-2:] == passages[1].text.split()[:2]
    assert (
        make_passages(text, source_name="candidate.txt", target_words=10, overlap_words=2)
        == passages
    )
