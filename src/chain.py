"""
Phase 2: RAG Answer Chain — Citation Enforcement + Prompt Versioning
---------------------------------------------------------------------
Generates answers grounded in retrieved chunks.
Declines to answer if context doesn't support a response (no hallucination).
Prompts loaded from version-controlled config/prompts.yaml.

LangChain integrations used:
  - langchain-groq: ChatGroq (llama-3.3-70b-versatile as LLM)
  - langchain-core: ChatPromptTemplate, StrOutputParser, RunnablePassthrough
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from rich.console import Console

# LangChain Groq integration
from langchain_groq import ChatGroq

# LangChain core LCEL primitives
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.retriever import RetrievedChunk

load_dotenv()
console = Console()

# ─── Prompt loader ────────────────────────────────────────────────────────────
_PROMPTS_PATH = Path(__file__).parent.parent / "config" / "prompts.yaml"

def load_prompts() -> dict:
    """Load version-controlled prompts from YAML config."""
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


# ─── LLM ──────────────────────────────────────────────────────────────────────

def get_llm() -> ChatGroq:
    """Groq-hosted LLaMA 3.3 70B via langchain-groq integration."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0,       # Deterministic for citations
        max_tokens=1024,
    )


# ─── Context builder ──────────────────────────────────────────────────────────

def build_context_string(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block.
    Each chunk is labelled [Source N] so the LLM can cite it.
    """
    parts = []
    for chunk in chunks:
        parts.append(
            f"[Source {chunk.source_index}] "
            f"({chunk.metadata.get('source', 'unknown')})\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)


# ─── RAG Chain ────────────────────────────────────────────────────────────────

def build_rag_chain():
    """
    LCEL chain: context + question → cited answer.
    Uses prompts from version-controlled YAML.
    """
    prompts = load_prompts()
    rag_cfg = prompts["rag_answer"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", rag_cfg["system"]),
        ("human",  rag_cfg["human"]),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    return chain


def answer(question: str, chunks: List[RetrievedChunk]) -> dict:
    """
    Generate a cited answer from retrieved chunks.

    Returns:
        {
          "answer": str,           # The LLM answer with [Source N] citations
          "sources": [str, ...],   # Human-readable citation strings
          "refused": bool,         # True if context was insufficient
        }
    """
    if not chunks:
        return {
            "answer": "I cannot answer this question based on the available documents.",
            "sources": [],
            "refused": True,
        }

    context = build_context_string(chunks)
    chain = build_rag_chain()

    raw_answer = chain.invoke({
        "context": context,
        "question": question,
    })

    # Citation enforcement: check if LLM declined to answer
    refused = "cannot answer" in raw_answer.lower() or "not" in raw_answer[:60].lower()

    sources = [chunk.citation for chunk in chunks]

    return {
        "answer": raw_answer,
        "sources": sources,
        "refused": refused,
        "chunks": chunks,   # kept for eval pipeline
    }


# ─── Faithfulness judge ───────────────────────────────────────────────────────

def check_faithfulness(answer_text: str, chunks: List[RetrievedChunk]) -> dict:
    """
    Use Claude to judge if every claim in the answer is
    actually supported by the retrieved chunks (Phase 2 eval guard).
    """
    prompts = load_prompts()
    faith_cfg = prompts["faithfulness_check"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", faith_cfg["system"]),
        ("human",  faith_cfg["human"]),
    ])

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    context = build_context_string(chunks)
    raw = chain.invoke({"context": context, "answer": answer_text})

    try:
        # Strip markdown fences if present
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"faithful": None, "unsupported_claims": [], "raw": raw}
