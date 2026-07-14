"""
knowledge_base/chunker.py

Splits the raw text of each loaded document into meaningful, overlapping
chunks and attaches ChunkMetadata (document name, page number, chunk number)
to every chunk, as required by Functional Requirement #2.
"""

from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge_base.metadata import ChunkMetadata, DocumentMetadata

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def chunk_document(
    metadata: DocumentMetadata,
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Splits every page of a document into chunks.

    Splitting happens per page so that the resulting page_number metadata
    stays accurate even for multi-page PDFs. Chunk numbers are sequential
    across the whole document (not reset per page), which makes citations
    like "chunk 7 of HR_Policy.md" unambiguous.

    Returns a flat list of:
        {"text": str, "metadata": ChunkMetadata}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    running_chunk_number = 0

    for page in pages:
        page_text = page["text"]
        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(page_text)
        for piece in page_chunks:
            running_chunk_number += 1
            chunk_meta = ChunkMetadata(
                chunk_id=f"{metadata.doc_id}_{running_chunk_number}",
                doc_id=metadata.doc_id,
                document_name=metadata.file_name,
                chunk_number=running_chunk_number,
                page_number=page.get("page_number"),
                file_type=metadata.file_type,
            )
            chunks.append({"text": piece, "metadata": chunk_meta})

    return chunks
