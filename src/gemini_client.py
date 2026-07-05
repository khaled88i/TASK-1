"""
gemini_client.py
----------------
Thin wrapper around the Google Gemini API using the official `google-genai`
SDK (the current, actively-maintained client library; the older
`google-generativeai` package is deprecated). Centralizes model
configuration, embedding calls, and answer generation so the rest of the
application never talks to the SDK directly.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "google-genai is not installed. Run `pip install -r requirements.txt`."
    ) from exc


DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

SYSTEM_INSTRUCTION = (
    "You are a Knowledge Base Assistant. You must answer the user's question "
    "using ONLY the information contained in the CONTEXT section below. "
    "Do not use any outside knowledge and do not guess or make assumptions.\n\n"
    "Rules:\n"
    "1. If the answer is fully or partially contained in the context, answer clearly "
    "and cite which source file(s) the information came from.\n"
    "2. If the context does not contain the information needed to answer the question, "
    "respond EXACTLY with: \"I don't have information about this in the Knowledge Base.\" "
    "Do not attempt to answer from general knowledge.\n"
    "3. Never fabricate facts, sources, or details that are not present in the context."
)


class GeminiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. Add it to your .env file or export it "
                "as an environment variable before running the assistant."
            )
        self.client = genai.Client(api_key=api_key)
        self.generation_model_name = generation_model
        self.embedding_model_name = embedding_model

    def embed_texts(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT",
                     batch_size: int = 32, max_retries: int = 3) -> List[List[float]]:
        """Embed a list of texts, batching requests and retrying on transient errors."""
        embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(1, max_retries + 1):
                try:
                    result = self.client.models.embed_content(
                        model=self.embedding_model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task_type),
                    )
                    embeddings.extend([e.values for e in result.embeddings])
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == max_retries:
                        raise RuntimeError(
                            f"Failed to embed batch after {max_retries} attempts: {exc}"
                        ) from exc
                    time.sleep(2 ** attempt)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model=self.embedding_model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    def generate_answer(self, question: str, context: str, max_retries: int = 3) -> str:
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "Answer using only the CONTEXT above, following the rules in your system instruction."
        )
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.generation_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
                )
                return (response.text or "").strip()
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries:
                    raise RuntimeError(f"Gemini generation failed: {exc}") from exc
                time.sleep(2 ** attempt)
        return ""
