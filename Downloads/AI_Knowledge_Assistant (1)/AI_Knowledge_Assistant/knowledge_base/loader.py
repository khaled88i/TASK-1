"""
knowledge_base/loader.py

Loads raw text out of PDF, DOCX, TXT, Markdown, and CSV files, page by page
where the format supports pages (PDF), and attaches DocumentMetadata to
every file discovered in the knowledge base folder (Functional Requirement #1
and #2: automatic indexing + metadata extraction).
"""

import os
from typing import List, Dict, Any

import pandas as pd
from pypdf import PdfReader
from docx import Document as DocxDocument

from knowledge_base.metadata import (
    DocumentMetadata,
    compute_file_hash,
    make_doc_id,
    SUPPORTED_EXTENSIONS,
)


def discover_files(folder_path: str) -> List[str]:
    """Returns the list of all supported files inside a folder (recursive)."""
    found = []
    for root, _dirs, files in os.walk(folder_path):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found)


def build_document_metadata(file_path: str) -> DocumentMetadata:
    """Builds the DocumentMetadata for a single file (without loading its text)."""
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    num_pages = None
    if ext == "pdf":
        try:
            num_pages = len(PdfReader(file_path).pages)
        except Exception:
            num_pages = None

    return DocumentMetadata(
        doc_id=make_doc_id(file_path),
        file_name=os.path.basename(file_path),
        file_path=os.path.abspath(file_path),
        file_type=ext,
        file_hash=compute_file_hash(file_path),
        num_pages=num_pages,
    )


def _load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """Returns one entry per page: {'text': ..., 'page_number': int}.

    Robust against corrupted PDFs: if the file can't be opened at all, an
    empty page list is returned (the caller/retriever logs and skips it
    instead of crashing the whole indexing run). If an individual page
    fails to extract (common with malformed/scanned pages), that single
    page is skipped and the rest of the document is still indexed.
    """
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        print(f"[loader] Could not open PDF '{file_path}': {exc}")
        return []

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"[loader] Skipping page {i} of '{file_path}' (extraction error: {exc})")
            continue
        if text.strip():
            pages.append({"text": text, "page_number": i})
    return pages


def _load_docx(file_path: str) -> List[Dict[str, Any]]:
    """DOCX has no native page concept, so the whole document is returned
    as a single unit with page_number=None."""
    try:
        doc = DocxDocument(file_path)
    except Exception as exc:
        print(f"[loader] Could not open DOCX '{file_path}': {exc}")
        return []
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "page_number": None}]


def _read_text_with_fallback(file_path: str) -> str:
    """Reads a text file as UTF-8, falling back to latin-1 (which never
    raises) if the file isn't valid UTF-8, instead of silently dropping
    bytes with errors='ignore'."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"[loader] '{file_path}' is not valid UTF-8, reading as latin-1 instead.")
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def _load_txt(file_path: str) -> List[Dict[str, Any]]:
    try:
        text = _read_text_with_fallback(file_path)
    except Exception as exc:
        print(f"[loader] Could not read TXT '{file_path}': {exc}")
        return []
    return [{"text": text, "page_number": None}]


def _load_markdown(file_path: str) -> List[Dict[str, Any]]:
    try:
        text = _read_text_with_fallback(file_path)
    except Exception as exc:
        print(f"[loader] Could not read Markdown '{file_path}': {exc}")
        return []
    return [{"text": text, "page_number": None}]


def _load_csv(file_path: str) -> List[Dict[str, Any]]:
    """Each CSV row becomes its own 'page' unit, rendered as readable text
    so the chunker and the LLM can reason about it in natural language."""
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"[loader] Could not read CSV '{file_path}': {exc}")
        return []

    pages = []
    for _, row in df.iterrows():
        row_text = "; ".join(f"{col}: {row[col]}" for col in df.columns)
        pages.append({"text": row_text, "page_number": None})
    return pages


_LOADERS = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "txt": _load_txt,
    "md": _load_markdown,
    "csv": _load_csv,
}


def load_document(file_path: str) -> Dict[str, Any]:
    """Loads a single file and returns its metadata plus a list of page units.

    Returns:
        {
            "metadata": DocumentMetadata,
            "pages": [{"text": str, "page_number": int|None}, ...]
        }
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")

    metadata = build_document_metadata(file_path)
    pages = _LOADERS[ext](file_path)
    return {"metadata": metadata, "pages": pages}
