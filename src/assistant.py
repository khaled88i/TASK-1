"""
assistant.py
------------
Top-level orchestration: keeps the vector store in sync with the Knowledge
Base directory, and answers user questions by retrieving relevant chunks
and asking Gemini to generate a grounded answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .gemini_client import GeminiClient
from .ingestion import build_chunks_for_file, discover_files, file_hash
from .vector_store import VectorStore

NOT_FOUND_MESSAGE = "I don't have information about this in the Knowledge Base."


class KnowledgeBaseAssistant:
    def __init__(
        self,
        kb_dir: Path,
        index_path: Path,
        gemini_client: GeminiClient,
        top_k: int = 5,
        min_score: float = 0.55,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self.kb_dir = kb_dir
        self.gemini = gemini_client
        self.store = VectorStore(index_path)
        self.top_k = top_k
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def sync(self, force: bool = False, verbose: bool = True) -> dict:
        """
        Scan the knowledge base directory and update the vector store so it
        reflects the current set of files: new/changed files are embedded,
        and files removed from disk are dropped from the index.
        """
        if force:
            self.store.clear()

        current_files = discover_files(self.kb_dir)
        current_sources = set()
        added, updated, unchanged = 0, 0, 0

        for path in current_files:
            rel_source = str(path.relative_to(self.kb_dir))
            current_sources.add(rel_source)
            digest = file_hash(path)

            if self.store.get_file_hash(rel_source) == digest:
                unchanged += 1
                continue

            is_update = rel_source in self.store.known_sources()
            self.store.remove_source(rel_source)

            chunks = build_chunks_for_file(
                path, self.kb_dir, chunk_size=self.chunk_size, overlap=self.chunk_overlap
            )
            if chunks:
                embeddings = self.gemini.embed_texts([c.text for c in chunks])
                self.store.add_chunks(chunks, embeddings)
            self.store.set_file_hash(rel_source, digest)

            if is_update:
                updated += 1
                if verbose:
                    print(f"[sync] Updated: {rel_source} ({len(chunks)} chunks)")
            else:
                added += 1
                if verbose:
                    print(f"[sync] Added: {rel_source} ({len(chunks)} chunks)")

        # Remove files that no longer exist on disk
        removed = 0
        for known_source in list(self.store.known_sources()):
            if known_source not in current_sources:
                self.store.remove_source(known_source)
                del self.store.file_hashes[known_source]
                removed += 1
                if verbose:
                    print(f"[sync] Removed (deleted from disk): {known_source}")

        self.store.save()
        summary = {
            "added": added,
            "updated": updated,
            "unchanged": unchanged,
            "removed": removed,
            "total_chunks": len(self.store.records),
        }
        if verbose:
            print(
                f"[sync] Done. Added={added} Updated={updated} "
                f"Unchanged={unchanged} Removed={removed} "
                f"TotalChunks={summary['total_chunks']}"
            )
        return summary

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #
    def retrieve(self, question: str) -> List[Tuple[dict, float]]:
        query_embedding = self.gemini.embed_query(question)
        return self.store.search(query_embedding, top_k=self.top_k, min_score=self.min_score)

    def answer(self, question: str) -> dict:
        """
        Return a dict with the answer text and the source chunks used, so the
        caller (CLI/web UI) can display citations alongside the answer.
        """
        if self.store.is_empty():
            return {
                "answer": (
                    "The Knowledge Base is empty. Add PDF, DOCX, TXT, Markdown, "
                    "or CSV files to the 'knowledge_base' directory and run a sync."
                ),
                "sources": [],
            }

        results = self.retrieve(question)
        if not results:
            return {"answer": NOT_FOUND_MESSAGE, "sources": []}

        context = "\n\n---\n\n".join(
            f"[Source: {record['source']} | chunk {record['chunk_index']}]\n{record['text']}"
            for record, _score in results
        )
        answer_text = self.gemini.generate_answer(question, context)

        # Defense in depth: if the model claims it has no info, don't show
        # sources, since they weren't actually used to ground an answer.
        sources = [] if NOT_FOUND_MESSAGE.lower() in answer_text.lower() else [
            {"source": r["source"], "chunk_index": r["chunk_index"], "score": round(score, 3)}
            for r, score in results
        ]
        return {"answer": answer_text or NOT_FOUND_MESSAGE, "sources": sources}
