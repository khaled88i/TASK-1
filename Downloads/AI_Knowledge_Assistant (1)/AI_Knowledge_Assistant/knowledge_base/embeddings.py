"""
embeddings.py

Provides the embedding function used to turn chunk text into vectors.

Default provider: HuggingFace sentence-transformers (free, runs locally,
no API key required) -- this is what the automated tests in this project
run against.

Optional provider: Google's Gemini embedding model, selected by setting
EMBEDDING_PROVIDER=gemini in the .env file (requires GOOGLE_API_KEY).
Both providers implement the same LangChain Embeddings interface, so the
rest of the codebase (retriever.py) never needs to know which one is active.
"""

import config


def get_embedding_function():
    """Returns a LangChain-compatible embedding function based on config."""
    if config.EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # default: local, free, offline-capable HuggingFace embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
