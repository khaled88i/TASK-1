"""
vector_store.py
----------------
A lightweight, dependency-free vector store used to persist chunk embeddings
to disk and perform cosine-similarity search at query time.

No external vector database is required — embeddings are stored as a JSON
file next to a compact index of which source files have already been
processed (via file_hash), so re-running ingestion only re-embeds files that
are new or have changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ingestion import Chunk


class VectorStore:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.records: List[dict] = []          # [{id, text, source, chunk_index, embedding}, ...]
        self.file_hashes: Dict[str, str] = {}   # relative source path -> hash
        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.records = data.get("records", [])
            self.file_hashes = data.get("file_hashes", {})

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"records": self.records, "file_hashes": self.file_hashes},
                f,
            )

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        self.records = []
        self.file_hashes = {}

    def remove_source(self, source: str) -> None:
        self.records = [r for r in self.records if r["source"] != source]

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self.records.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "embedding": embedding,
                }
            )

    def set_file_hash(self, source: str, digest: str) -> None:
        self.file_hashes[source] = digest

    def get_file_hash(self, source: str) -> Optional[str]:
        return self.file_hashes.get(source)

    def known_sources(self) -> List[str]:
        return list(self.file_hashes.keys())

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query_embedding: List[float], top_k: int = 5,
               min_score: float = 0.0) -> List[Tuple[dict, float]]:
        scored = [
            (record, self._cosine_similarity(query_embedding, record["embedding"]))
            for record in self.records
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        results = [pair for pair in scored if pair[1] >= min_score]
        return results[:top_k]

    def is_empty(self) -> bool:
        return len(self.records) == 0
