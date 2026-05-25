"""
main.py — Production RAG System CLI
-------------------------------------
Ties all three phases together:
  Phase 1: ingest → chunk → embed → store
  Phase 2: hybrid retrieval + rerank → cited answer
  Phase 3: evaluate against golden dataset

Usage:
  python main.py ingest docs/paper.pdf docs/guide.md
  python main.py ask "What is the main conclusion of the paper?"
  python main.py evaluate --dataset eval/golden_dataset.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

load_dotenv()
console = Console()
import os
os.environ.setdefault("USER_AGENT", "rag-system/1.0")


def cmd_ingest(args):
    from src.ingest import ingest
    sources = args.sources
    if not sources:
        # Auto-discover docs/ folder
        sources = (
            [str(p) for p in Path("docs").glob("*.pdf")] +
            [str(p) for p in Path("docs").glob("*.md")] +
            [str(p) for p in Path("docs").glob("*.txt")]
        )
    if not sources:
        console.print("[red]No sources found. Pass file paths or URLs.[/]")
        sys.exit(1)
    ingest(sources)
    console.print("[bold green]Ingestion complete.[/]")


def cmd_ask(args):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    from src.ingest import CHROMA_DIR, COLLECTION_NAME
    from src.retriever import retrieve
    from src.chain import answer, check_faithfulness

    question = " ".join(args.question)

    # Load vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Reconstruct chunks for BM25
    raw = vectorstore.get()
    all_chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    # Retrieve
    chunks = retrieve(question, vectorstore, all_chunks)
    if not chunks:
        console.print("[red]No relevant chunks found in the document corpus.[/]")
        return

    # Generate answer
    result = answer(question, chunks)

    # Display
    console.print()
    console.print(Panel(
        Markdown(result["answer"]),
        title=f"[bold cyan]Answer[/] — Q: {question[:70]}",
        border_style="cyan",
    ))

    if not result["refused"]:
        console.print("\n[bold]Sources:[/]")
        for src in result["sources"]:
            console.print(f"  {src}")

        # Optional faithfulness check
        if args.check_faithfulness:
            faith = check_faithfulness(result["answer"], chunks)
            if faith.get("faithful"):
                console.print("\n[green]✓ Faithfulness check passed[/]")
            else:
                console.print("\n[yellow]⚠ Unsupported claims detected:[/]")
                for claim in faith.get("unsupported_claims", []):
                    console.print(f"  - {claim}")
    else:
        console.print("[yellow]⚠ System declined to answer: insufficient context[/]")


def cmd_evaluate(args):
    # Delegate to evaluate module (Phase 3)
    # This also sets sys.exit based on threshold, powering CI/CD
    from src.evaluate import main as eval_main
    sys.argv = ["evaluate", "--dataset", args.dataset, "--threshold", str(args.threshold)]
    eval_main()


# ─── CLI routing ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Production RAG System (LangChain + ChromaDB + Claude)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the vector store")
    p_ingest.add_argument("sources", nargs="*", help="File paths or URLs")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question against ingested documents")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--check-faithfulness", action="store_true",
                       help="Run faithfulness check on the answer")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Run offline evaluation (Phase 3)")
    p_eval.add_argument("--dataset", default="eval/golden_dataset.json")
    p_eval.add_argument("--threshold", type=float, default=0.70)

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
