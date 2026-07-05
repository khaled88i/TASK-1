"""
Unit tests for the ingestion module. These tests do not require a Gemini API
key since they only exercise local file parsing and chunking logic.

Run with:  python -m pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import build_all_chunks, chunk_text, discover_files  # noqa: E402


def test_chunk_text_short_returns_single_chunk():
    text = "This is a short piece of text."
    chunks = chunk_text(text, chunk_size=1000, overlap=150)
    assert len(chunks) == 1
    assert "short piece of text" in chunks[0]


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("", chunk_size=1000) == []
    assert chunk_text("   ", chunk_size=1000) == []


def test_chunk_text_long_text_splits_into_multiple_chunks():
    lines = [f"This is paragraph number {i} with some extra filler words." for i in range(30)]
    text = "\n".join(lines)
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # every chunk should respect roughly the requested size (with overlap slack)
    assert all(len(c) <= 200 + 50 + 20 for c in chunks)


def test_discover_files_finds_supported_types(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.csv").write_text("col1,col2\nv1,v2")
    (tmp_path / "d.unsupported").write_text("ignored")

    found = discover_files(tmp_path)
    names = {f.name for f in found}
    assert names == {"a.txt", "b.md", "c.csv"}


def test_build_all_chunks_txt_and_csv(tmp_path):
    (tmp_path / "notes.txt").write_text("Amman is the capital of Jordan.")
    (tmp_path / "data.csv").write_text("name,role\nKhaled,Student")

    chunks = build_all_chunks(tmp_path)
    sources = {c.source for c in chunks}
    assert "notes.txt" in sources
    assert "data.csv" in sources
    joined_text = " ".join(c.text for c in chunks)
    assert "Amman" in joined_text
    assert "Khaled" in joined_text
