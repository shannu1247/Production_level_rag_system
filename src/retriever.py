"""
Phase 2: Hybrid Retrieval + Cross-Encoder Reranking
----------------------------------------------------
Combines BM25 keyword search with semantic vector search, then
reranks results with a cross-encoder model for precision.

LangChain integrations used:
  - langchain-community: BM25Retriever, EnsembleRetriever
  - langchain-chroma: Chroma (as base retriever)
  - langchain-huggingface: HuggingFaceEmbeddings
  - sentence-transformers: CrossEncoder (for reranking)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from rich.console import Console

# LangChain retrievers
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Cross-encoder reranker (sentence-transformers)
from sentence_transformers import CrossEncoder

from src.ingest import CHROMA_DIR, COLLECTION_NAME

load_dotenv()
console = Console()

# ─── Config ───────────────────────────────────────────────────────────────────
TOP_K_INITIAL = 10       # Retrieve this many candidates before reranking
TOP_K_FINAL = 4          # Return this many after reranking
BM25_WEIGHT = 0.4        # Weight for keyword search in ensemble
VECTOR_WEIGHT = 0.6      # Weight for semantic search in ensemble
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    content: str
    metadata: dict
    score: float
    source_index: int  # 1-based index for citation: [Source N]

    @property
    def citation(self) -> str:
        src = self.metadata.get("source", "unknown")
        page = self.metadata.get("page", "")
        page_str = f", p.{page}" if page else ""
        return f"[Source {self.source_index}] {src}{page_str}"


# ─── Retriever factory ────────────────────────────────────────────────────────

def load_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )




def build_hybrid_retriever(vectorstore, all_chunks):
    """
    Manual hybrid retrieval: run BM25 and vector search separately,
    merge results by combining unique docs (replaces EnsembleRetriever).
    """
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = TOP_K_INITIAL

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_INITIAL},
    )

    def hybrid_invoke(query):
        bm25_docs = bm25_retriever.invoke(query)
        vector_docs = vector_retriever.invoke(query)
        # Merge, deduplicate by content
        seen = set()
        merged = []
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return merged[:TOP_K_INITIAL]

    return hybrid_invoke


def rerank(query: str, candidates: List[Document]) -> List[Tuple[Document, float]]:
    """
    Cross-encoder reranker: scores (query, chunk) pairs together.
    Dramatically improves precision over initial retrieval ranking.
    """
    model = CrossEncoder(RERANKER_MODEL)
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:TOP_K_FINAL]


def retrieve(query: str, vectorstore: Chroma, all_chunks: List[Document]) -> List[RetrievedChunk]:
    """
    Full hybrid retrieval pipeline:
      1. Ensemble retrieval (BM25 + vector)
      2. Cross-encoder reranking
      3. Return annotated RetrievedChunk objects with source indices
    """
    hybrid = build_hybrid_retriever(vectorstore, all_chunks)

    console.print(f"[cyan]Retrieving for:[/] {query!r}")
    candidates = hybrid(query)
    console.print(f"  Candidates from hybrid retrieval: [bold]{len(candidates)}[/]")

    ranked = rerank(query, candidates)
    console.print(f"  After reranking, top-k kept: [bold]{len(ranked)}[/]")

    results = []
    for i, (doc, score) in enumerate(ranked, start=1):
        results.append(RetrievedChunk(
            content=doc.page_content,
            metadata=doc.metadata,
            score=float(score),
            source_index=i,
        ))

    return results
