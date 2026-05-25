"""
Phase 1: Document Ingestion & Chunking Pipeline
-----------------------------------------------
Loads PDF / Markdown / web pages, chunks them with overlap,
embeds via HuggingFace, stores in ChromaDB.

LangChain integrations used:
  - langchain-community: PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader
  - langchain-huggingface: HuggingFaceEmbeddings
  - langchain-chroma: Chroma vector store
  - langchain.text_splitter: RecursiveCharacterTextSplitter
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

# LangChain document loaders (langchain-community)
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)

# Text splitter (langchain-core / langchain)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings (langchain-huggingface)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store (langchain-chroma)
from langchain_chroma import Chroma

load_dotenv()
console = Console()

# ─── Constants ────────────────────────────────────────────────────────────────
CHUNK_SIZE = 700          # ~700 tokens per chunk (500–800 range from spec)
CHUNK_OVERLAP = 100       # 100-token overlap so boundary sentences aren't lost
import os
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"   # Free HuggingFace model, 768-dim


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_documents(source: str):
    """
    Load a document from a file path or URL.
    Supports: .pdf, .md, .txt, http/https URLs, and directories.
    """
    path = Path(source)

    # Handle directories — recurse into supported file types
    if path.is_dir():
        docs = []
        for ext in ("*.pdf", "*.md", "*.txt"):
            for file in sorted(path.glob(ext)):
                docs.extend(load_documents(str(file)))
        if not docs:
            console.print(f"[yellow]Warning: No supported files (.pdf/.md/.txt) found in {source}[/]")
        return docs

    elif source.startswith("http://") or source.startswith("https://"):
        console.print(f"[cyan]Loading web page:[/] {source}")
        loader = WebBaseLoader(source)

    elif source.endswith(".pdf"):
        console.print(f"[cyan]Loading PDF:[/] {source}")
        loader = PyPDFLoader(source)

    elif source.endswith(".md") or source.endswith(".txt"):
        console.print(f"[cyan]Loading Markdown/Text:[/] {source}")
        loader = UnstructuredMarkdownLoader(source)

    else:
        raise ValueError(f"Unsupported source type: {source}")

    return loader.load()


def chunk_documents(docs):
    """
    Split documents into 500–800 token chunks with 100-token overlap.
    RecursiveCharacterTextSplitter respects paragraph/sentence boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    console.print(f"[green]✓[/] Split into [bold]{len(chunks)}[/] chunks "
                  f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks, persist: bool = True) -> Chroma:
    """
    Embed chunks with HuggingFace BAAI/bge-base-en-v1.5 and store in ChromaDB.
    This model is free, runs locally (CPU), and is excellent for retrieval tasks.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity
    )

    if persist and Path(CHROMA_DIR).exists():
        console.print("[yellow]Loading existing ChromaDB...[/]")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        vectorstore.add_documents(chunks)
    else:
        console.print("[yellow]Creating new ChromaDB...[/]")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR if persist else None,
        )

    console.print(f"[green]✓[/] Vector store ready at [bold]{CHROMA_DIR}[/]")
    return vectorstore


def ingest(sources: List[str], persist: bool = True) -> Chroma:
    """
    Full Phase 1 ingestion pipeline:
      load → chunk → embed → store
    """
    all_docs = []
    for source in track(sources, description="Loading sources..."):
        all_docs.extend(load_documents(source))

    console.print(f"[green]✓[/] Loaded [bold]{len(all_docs)}[/] raw document pages")

    chunks = chunk_documents(all_docs)
    vectorstore = build_vector_store(chunks, persist=persist)
    return vectorstore


# ─── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sources = sys.argv[1:] if len(sys.argv) > 1 else [
        # Default: drop sample docs in ./docs/ folder
        *[str(p) for p in Path("./docs").glob("*.pdf")],
        *[str(p) for p in Path("./docs").glob("*.md")],
    ]

    if not sources:
        console.print("[red]No sources found. Pass file paths or URLs as arguments.[/]")
        sys.exit(1)

    ingest(sources)
