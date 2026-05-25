"""
Phase 3: Offline Evaluation Pipeline — Golden Dataset + RAGAS
--------------------------------------------------------------
Runs evaluation against a manually curated golden Q&A dataset.
Measures faithfulness, answer relevancy, context precision.
Designed to plug into CI/CD: exits with code 1 if quality drops below threshold.

LangChain integrations used:
  - ragas: RAG evaluation framework (faithfulness, answer_relevancy, context_precision)
  - langchain-groq: ChatGroq (judge model)
  - langchain-huggingface: HuggingFaceEmbeddings (for RAGAS semantic scoring)

Usage:
  python -m src.evaluate --dataset eval/golden_dataset.json --threshold 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# RAGAS evaluation framework
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

# LangChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.ingest import CHROMA_DIR, COLLECTION_NAME, load_documents, chunk_documents
from src.retriever import retrieve
from src.chain import answer

load_dotenv()
console = Console()

DEFAULT_THRESHOLD = 0.70   # Build fails if avg score drops below this


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_golden_dataset(path: str) -> List[dict]:
    """
    Load golden Q&A pairs.
    Expected JSON format:
      [{"question": "...", "ground_truth": "..."}, ...]
    """
    with open(path) as f:
        data = json.load(f)
    console.print(f"[green]✓[/] Loaded [bold]{len(data)}[/] golden Q&A pairs")
    return data


def run_rag_on_dataset(golden: List[dict], vectorstore: Chroma, all_chunks) -> dict:
    """
    Run the full RAG pipeline on each question in the golden dataset.
    Collects inputs needed by RAGAS.
    """
    questions, answers_list, contexts_list, ground_truths = [], [], [], []

    for item in golden:
        q = item["question"]
        gt = item["ground_truth"]

        chunks = retrieve(q, vectorstore, all_chunks)
        result = answer(q, chunks)

        questions.append(q)
        answers_list.append(result["answer"])
        contexts_list.append([c.content for c in chunks])
        ground_truths.append(gt)

        status = "[red]REFUSED[/]" if result["refused"] else "[green]answered[/]"
        console.print(f"  Q: {q[:60]}... → {status}")

    return {
        "question": questions,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }


# ─── RAGAS evaluation ─────────────────────────────────────────────────────────

def run_ragas_eval(dataset_dict: dict) -> dict:
    """
    Run RAGAS metrics:
      - faithfulness: are answer claims supported by context?
      - answer_relevancy: does the answer address the question?
      - context_precision: are top-ranked chunks relevant?
    """
    hf_dataset = Dataset.from_dict(dataset_dict)

    # Wrap LangChain models for RAGAS
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))

    results = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=embeddings,
    )
    return results


def print_results_table(results: dict, threshold: float) -> float:
    """Pretty-print results and return average score."""
    table = Table(title="RAG Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Threshold", style="dim")
    table.add_column("Status")

    scores = []
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        val = results.get(metric, 0.0)
        scores.append(float(val))
        status = "[green]PASS ✓[/]" if val >= threshold else "[red]FAIL ✗[/]"
        table.add_row(metric, f"{val:.3f}", f"{threshold:.2f}", status)

    avg = sum(scores) / len(scores) if scores else 0.0
    avg_status = "[green]PASS ✓[/]" if avg >= threshold else "[red]FAIL ✗[/]"
    table.add_row("[bold]Average[/]", f"[bold]{avg:.3f}[/]", f"{threshold:.2f}", avg_status)

    console.print(table)
    return avg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    # Load golden dataset
    golden = load_golden_dataset(args.dataset)

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

    # Get all chunks for BM25 (needed by hybrid retriever)
    raw_collection = vectorstore.get()
    from langchain_core.documents import Document
    all_chunks = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(
            raw_collection["documents"],
            raw_collection["metadatas"],
        )
    ]
    console.print(f"[green]✓[/] Loaded [bold]{len(all_chunks)}[/] chunks from vector store")

    # Run RAG on golden dataset
    console.print("\n[bold]Running RAG pipeline on golden dataset...[/]")
    dataset_dict = run_rag_on_dataset(golden, vectorstore, all_chunks)

    # Save raw results for audit
    Path("eval").mkdir(exist_ok=True)
    with open("eval/last_run_results.json", "w") as f:
        serializable = {k: v for k, v in dataset_dict.items()}
        json.dump(serializable, f, indent=2)

    # RAGAS evaluation
    console.print("\n[bold]Running RAGAS evaluation...[/]")
    results = run_ragas_eval(dataset_dict)

    # Print and decide pass/fail
    console.print()
    avg = print_results_table(results, args.threshold)

    if avg < args.threshold:
        console.print(f"\n[red bold]✗ BUILD FAILED:[/] Average score {avg:.3f} "
                      f"below threshold {args.threshold:.2f}")
        sys.exit(1)  # CI/CD integration: non-zero exit fails the build
    else:
        console.print(f"\n[green bold]✓ BUILD PASSED:[/] Average score {avg:.3f} "
                      f">= threshold {args.threshold:.2f}")
        sys.exit(0)


if __name__ == "__main__":
    main()
