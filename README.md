# Production RAG System

A production-grade **Retrieval-Augmented Generation** pipeline built entirely with LangChain, implementing all three phases from demo → production → shippable.

## Stack

| Component | Package | Purpose |
|---|---|---|
| LLM | `langchain-groq` | LLaMA 3.3 70B via Groq for generation |
| Embeddings | `langchain-huggingface>=0.1.0     # HuggingFace embeddings (BAAI/bge-base-en-v1.5)` |
| Vector store | `langchain-chroma` | ChromaDB persistent store |
| Document loaders | `langchain-community` | PDF, Markdown, Web pages |
| Keyword search | `langchain-community` | BM25Retriever |
| Hybrid retrieval | `langchain` | EnsembleRetriever (BM25 + vector) |
| Reranker | `sentence-transformers` | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Evaluation | `ragas` | Faithfulness, answer relevancy, context precision |

---

## Three Phases

### Phase 1 — Fundamentals
- Load PDFs, Markdown files, and web pages via `langchain-community` loaders
- Chunk with `RecursiveCharacterTextSplitter`: 700 tokens, 100-token overlap
- Embed with `BAAI/bge-base-en-v1.5` (free, local) and store in ChromaDB
- Retrieve top-k chunks and generate cited answers

### Phase 2 — Production Quality
- **Hybrid retrieval**: BM25 (keyword) + vector (semantic) via `EnsembleRetriever`
- **Cross-encoder reranking**: rescores (query, chunk) pairs together for precision
- **Citation enforcement**: system declines to answer when context is insufficient
- **Prompt versioning**: all prompts live in `config/prompts.yaml`

### Phase 3 — Shippable
- **Golden dataset**: `eval/golden_dataset.json` — manually curated Q&A pairs
- **RAGAS evaluation**: measures faithfulness, answer relevancy, context precision
- **CI/CD integration**: GitHub Actions runs eval on every PR; fails build if quality drops

---

## Setup

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY and HF_TOKEN

# 3. Drop your documents in docs/
mkdir docs
cp my_paper.pdf docs/
```

## Usage

```bash
# Phase 1: Ingest documents
python main.py ingest docs/paper.pdf docs/guide.md
python main.py ingest https://example.com/docs

# Phase 2: Ask questions with citations
python main.py ask "What is the main conclusion of the paper?"
python main.py ask "What methods were used?" --check-faithfulness

# Phase 3: Evaluate against golden dataset
python main.py evaluate --dataset eval/golden_dataset.json --threshold 0.70
```

## Project Structure

```
rag_system/
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example
├── config/
│   └── prompts.yaml           # Version-controlled prompts
├── src/
│   ├── ingest.py              # Phase 1: load → chunk → embed → store
│   ├── retriever.py           # Phase 2: hybrid BM25 + vector + reranker
│   ├── chain.py               # Phase 2: RAG chain with citation enforcement
│   └── evaluate.py            # Phase 3: RAGAS offline evaluation
├── eval/
│   └── golden_dataset.json    # Curated Q&A pairs (expand to 50–200)
├── docs/                      # Drop your documents here
└── .github/
    └── workflows/
        └── eval.yml           # CI/CD: eval runs on every PR
```

## Expanding the Golden Dataset

Add Q&A pairs to `eval/golden_dataset.json`:

```json
[
  {
    "question": "Your question here",
    "ground_truth": "The correct answer, verified manually"
  }
]
```

Aim for **50–200 pairs** covering edge cases, refusal cases (questions the docs can't answer), and domain-specific terminology.

## How CI/CD Works

The `evaluate` command exits with code `1` if the average RAGAS score drops below the threshold. GitHub Actions treats this as a build failure, blocking the PR merge. This is exactly how production AI teams gate quality.
