# Knowledge Base Assistant (Google Gemini)

A command-line assistant that answers questions **strictly** from the documents
you place in a local `knowledge_base/` folder. It uses Google Gemini for both
embeddings (retrieval) and answer generation, and is explicitly designed to
avoid hallucinations: if the answer isn't in your documents, it tells you so
instead of guessing.

Built as Task 1 for Shamsieh Technology Services Co.

---

## Project Overview

The assistant follows a classic **Retrieval-Augmented Generation (RAG)**
pipeline:

1. **Ingest** — every supported file inside `knowledge_base/` is parsed and
   split into overlapping text chunks.
2. **Embed** — each chunk is converted into a vector using Gemini's
   `gemini-embedding-001` model and stored in a local JSON-based vector index.
3. **Retrieve** — when you ask a question, the question itself is embedded
   and compared (cosine similarity) against every stored chunk to find the
   most relevant ones.
4. **Generate** — the top matching chunks are inserted into a strict prompt
   sent to `gemini-2.5-flash`, which is instructed to answer **only** from
   that context, and to say so explicitly if the answer isn't there.

## Features

- 📂 **Automatic ingestion** of PDF, DOCX, TXT, Markdown, and CSV files placed
  in the `knowledge_base/` directory — no manual registration needed.
- 🔄 **Incremental sync** — files are hashed, so only new or changed files are
  re-embedded on subsequent runs; unchanged files are skipped and deleted
  files are automatically removed from the index.
- 🧠 **Grounded answers only** — a strict system prompt plus a relevance-score
  threshold at retrieval time work together to stop the model from answering
  from general knowledge.
- 🚫 **Explicit "not found" handling** — if no sufficiently relevant chunk is
  found, or the model determines the context doesn't answer the question, the
  assistant clearly says: *"I don't have information about this in the
  Knowledge Base."*
- 📎 **Source citations** — every grounded answer lists which file(s) and
  chunk(s) it came from, with a similarity score.
- 💬 **Two usage modes** — ask a single question (`ask`) or start a continuous
  interactive session (`chat`).
- ⚠️ **Error handling** — missing API keys, unreadable files, and transient
  API errors are caught and reported clearly instead of crashing.

## Technologies Used

| Purpose                     | Technology                              |
|------------------------------|------------------------------------------|
| LLM (generation + embeddings)| Google Gemini via the `google-genai` SDK |
| PDF parsing                  | `pypdf`                                  |
| DOCX parsing                 | `python-docx`                            |
| Vector storage / search      | Custom lightweight JSON store + cosine similarity (no external DB needed) |
| Config / secrets             | `python-dotenv`                          |
| Language                     | Python 3.10+                             |
| Testing                      | `pytest`                                 |

## Project Structure

```
kb-assistant/
├── knowledge_base/         # Put your PDF/DOCX/TXT/MD/CSV files here
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # File discovery, parsing, and chunking
│   ├── vector_store.py     # JSON-backed embedding store + cosine similarity search
│   ├── gemini_client.py    # Wrapper around the Gemini API (embeddings + generation)
│   ├── assistant.py        # Orchestrates sync + retrieval + grounded generation
│   └── cli.py               # Command-line interface (entry point)
├── tests/
│   └── test_ingestion.py   # Unit tests for parsing/chunking (no API key required)
├── .env.example             # Template for required environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/kb-assistant.git
   cd kb-assistant
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration Instructions

The assistant needs a Gemini API key.

1. Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and set your key:
   ```
   GEMINI_API_KEY=your_actual_key_here
   ```

The `.env` file is already excluded via `.gitignore` — never commit real API keys.

## How to Run the Project

1. **Add your documents** to the `knowledge_base/` folder (PDF, DOCX, TXT,
   Markdown, or CSV).

2. **Build/refresh the index:**
   ```bash
   python -m src.cli sync
   ```
   Use `--force` to rebuild the entire index from scratch:
   ```bash
   python -m src.cli sync --force
   ```

3. **Ask a single question:**
   ```bash
   python -m src.cli ask "What is the company's leave policy?"
   ```

4. **Or start an interactive chat session** (automatically syncs first):
   ```bash
   python -m src.cli chat
   ```
   Type `exit` or `quit` to leave the session.

## Usage Examples

```bash
$ python -m src.cli sync
[sync] Added: policies/leave_policy.pdf (4 chunks)
[sync] Added: onboarding/faq.md (2 chunks)
[sync] Done. Added=2 Updated=0 Unchanged=0 Removed=0 TotalChunks=6

$ python -m src.cli ask "How many annual leave days do employees get?"

======================================================================
ANSWER:
According to policies/leave_policy.pdf, employees are entitled to 21 paid
annual leave days per year.

SOURCES:
  - policies/leave_policy.pdf (chunk 1, relevance 0.812)
======================================================================

$ python -m src.cli ask "What is the CEO's home address?"

======================================================================
ANSWER:
I don't have information about this in the Knowledge Base.
======================================================================
```

## Assumptions and Limitations

- **Local, single-user use.** The vector index is a single JSON file on disk
  (`.index/kb_index.json`); it is not designed for concurrent multi-user
  access or very large corpora (tens of thousands of chunks).
- **No external vector database.** Cosine similarity search is done in-memory
  in pure Python for simplicity and to avoid extra infrastructure. This is
  fast enough for a knowledge base of a few hundred documents, but would need
  to be swapped for a real vector DB (e.g., FAISS, Chroma, pgvector) at scale.
- **Relevance threshold.** Answers are only generated when at least one chunk
  clears a minimum cosine-similarity score (default `0.55`). This threshold
  can be tuned in `assistant.py` if it's too strict or too lenient for your
  documents.
- **Text-based formats only.** Scanned/image-only PDFs (no extractable text
  layer) will yield little or no usable text — OCR is not included.
- **CSV handling.** CSV files are flattened into "column: value" sentences
  per row for embedding; very wide or deeply nested CSVs may not chunk ideally.
- **API costs/rate limits.** Every sync and every question consumes Gemini
  API quota; large knowledge bases will require more embedding calls.
