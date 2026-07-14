"""
knowledge_base/metadata.py

Defines the metadata structures attached to every document and every chunk,
plus the hashing utility used to detect new / changed files for incremental
re-indexing (Functional Requirement #1: "detect newly added documents and
allow re-indexing without rebuilding the entire project").
"""

import hashlib
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DocumentMetadata:
    """Metadata describing one source file in the knowledge base."""
    doc_id: str            # stable id derived from the file path
    file_name: str         # e.g. "HR_Policy.md"
    file_path: str         # absolute path on disk
    file_type: str         # "pdf" | "docx" | "txt" | "md" | "csv"
    file_hash: str         # sha256 of the file content (used for change detection)
    indexed_at: float = field(default_factory=time.time)
    num_pages: Optional[int] = None  # only meaningful for PDFs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkMetadata:
    """Metadata attached to a single retrievable chunk."""
    chunk_id: str                  # unique id: f"{doc_id}_{chunk_number}"
    doc_id: str
    document_name: str             # required by spec: "Document Name"
    chunk_number: int              # required by spec: "Chunk Number"
    page_number: Optional[int]     # required by spec: "Page Number (if available)"
    file_type: str

    def to_dict(self) -> dict:
        return asdict(self)


def compute_file_hash(file_path: str, block_size: int = 65536) -> str:
    """Returns the sha256 hex digest of a file's content.

    This is the core mechanism that lets the indexer tell whether a file is
    brand new, unchanged, or has been modified since the last indexing run,
    without needing to compare full document content.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def make_doc_id(file_path: str) -> str:
    """Deterministic, filesystem-independent id for a document."""
    return hashlib.md5(os.path.abspath(file_path).encode("utf-8")).hexdigest()[:16]


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
