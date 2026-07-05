"""
cli.py
------
Command line interface for the Knowledge Base Assistant.

Usage:
    python -m src.cli sync                 Ingest/refresh the knowledge base index
    python -m src.cli ask "your question"  Ask a single question and exit
    python -m src.cli chat                 Start an interactive chat session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .assistant import KnowledgeBaseAssistant
from .gemini_client import GeminiClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".index" / "kb_index.json"


def build_assistant(kb_dir: Path) -> KnowledgeBaseAssistant:
    load_dotenv()
    gemini = GeminiClient()
    return KnowledgeBaseAssistant(kb_dir=kb_dir, index_path=DEFAULT_INDEX_PATH, gemini_client=gemini)


def print_answer(result: dict) -> None:
    print("\n" + "=" * 70)
    print("ANSWER:")
    print(result["answer"])
    if result["sources"]:
        print("\nSOURCES:")
        for src in result["sources"]:
            print(f"  - {src['source']} (chunk {src['chunk_index']}, relevance {src['score']})")
    print("=" * 70 + "\n")


def cmd_sync(args: argparse.Namespace) -> None:
    kb_dir = Path(args.kb_dir).resolve()
    kb_dir.mkdir(parents=True, exist_ok=True)
    assistant = build_assistant(kb_dir)
    assistant.sync(force=args.force)


def cmd_ask(args: argparse.Namespace) -> None:
    kb_dir = Path(args.kb_dir).resolve()
    assistant = build_assistant(kb_dir)
    assistant.sync(verbose=False)  # keep the index fresh before answering
    result = assistant.answer(args.question)
    print_answer(result)


def cmd_chat(args: argparse.Namespace) -> None:
    kb_dir = Path(args.kb_dir).resolve()
    assistant = build_assistant(kb_dir)
    print("Syncing knowledge base...")
    assistant.sync(verbose=True)
    print("\nKnowledge Base Assistant ready. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = assistant.answer(question)
        print_answer(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="kb-assistant",
        description="Knowledge Base Assistant powered by Google Gemini",
    )
    parser.add_argument(
        "--kb-dir",
        default=str(DEFAULT_KB_DIR),
        help="Path to the Knowledge Base directory (default: ./knowledge_base)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Ingest/refresh the knowledge base index")
    sync_parser.add_argument(
        "--force", action="store_true", help="Rebuild the entire index from scratch"
    )
    sync_parser.set_defaults(func=cmd_sync)

    ask_parser = subparsers.add_parser("ask", help="Ask a single question and exit")
    ask_parser.add_argument("question", type=str, help="The question to ask")
    ask_parser.set_defaults(func=cmd_ask)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    try:
        args.func(args)
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
