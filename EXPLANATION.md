# Production RAG System — Complete Explanation

This document explains every file in the project: what it does, why it exists, how it works internally, and which LangChain packages power it. Read this alongside the code to understand the full picture.

---

## Big Picture: What This System Does

This is a **domain-specific "ask my docs" system**. You feed it a corpus of documents (PDFs, Markdown files, web pages), and it lets you ask natural-language questions. Instead of just guessing an answer, it **retrieves the most relevant passages** from your documents and generates a response that **cites exactly where the information came from**. If the documents don't contain enough information to answer, the system says so rather than making something up.

The system is built in three phases, going from working demo all the way to production-shippable:

- **Phase 1** — Ingest documents and store them searchably
- **Phase 2** — Retrieve precisely and generate cited, grounded answers
- **Phase 3** — Continuously evaluate quality so regressions get caught automatically

---

## Project Structure

```
rag_system/
├── main.py                        ← Single CLI entry point for all commands
├── requirements.txt               ← All Python package dependencies
├── .env.example                   ← API key template
├── README.md                      ← Quick-start guide
├── EXPLANATION.md                 ← This file
│
├── config/
│   └── prompts.yaml               ← Version-controlled prompt templates
│
├── src/
│   ├── __init__.py                ← Makes src/ a Python package
│   ├── ingest.py                  ← Phase 1: load → chunk → embed → store
│   ├── retriever.py               ← Phase 2: hybrid retrieval + reranking
│   ├── chain.py                   ← Phase 2: answer generation + citation enforcement
│   └── evaluate.py                ← Phase 3: offline evaluation with RAGAS
│
├── eval/
│   └── golden_dataset.json        ← Curated Q&A pairs for evaluation
│
├── docs/                          ← Drop your own documents here
│
└── .github/
    └── workflows/
        └── eval.yml               ← GitHub Actions CI/CD pipeline
```

---

## File-by-File Explanation

---

### `requirements.txt`

**What it is:** The list of every Python package the system needs, with version pins.

**Why these packages:**

| Package | Role |
|---|---|
| `langchain` | Core framework: text splitters, retriever abstractions, LCEL chain primitives |
| `langchain-core` | Base interfaces: `Document`, `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough` |
| `langchain-community` | Community integrations: `PyPDFLoader`, `UnstructuredMarkdownLoader`, `WebBaseLoader`, `BM25Retriever`, `EnsembleRetriever` |
| `langchain-groq` | Official Anthropic integration: `ChatGroq` for Claude |
| `langchain-huggingface | `BAAI/bge-base-en-v1.5` — free, local, 768-dim` |
| `langchain-chroma` | Official ChromaDB integration: persistent local vector store |
| `pypdf` | PDF parsing backend used by `PyPDFLoader` |
| `unstructured` | Document parsing backend used by `UnstructuredMarkdownLoader` |
| `rank-bm25` | BM25 algorithm implementation used by `BM25Retriever` |
| `sentence-transformers` | Provides the `CrossEncoder` model for reranking |
| `torch` | Required backend for sentence-transformers |
| `ragas` | RAG-specific evaluation framework with faithfulness/relevancy metrics |
| `pyyaml` | Loads the version-controlled `prompts.yaml` config |
| `python-dotenv` | Loads API keys from `.env` file |
| `rich` | Pretty terminal output: colored text, progress bars, tables |

---

### `.env.example`

**What it is:** A template showing which environment variables you need to set. You copy this to `.env` and fill in your actual keys.

**Variables:**

- `GROQ_API_KEY` — Required. Used by `langchain-groq` (`ChatGroq`) to call Claude for answer generation and faithfulness checking.
- `HF_TOKEN` — Required. Used by `langchain-huggingface | `BAAI/bge-base-en-v1.5` — free, local, 768-dim`.
- `LANGSMITH_*` — Optional. If you set these, LangChain automatically traces every chain call through LangSmith, giving you a visual debugger for your retrieval and generation steps.

**Why separate keys for two providers:** Claude is used as the LLM (it's excellent at instruction-following and citation) while OpenAI's embedding model is used for semantic search (it's cost-effective and high quality for retrieval). This is a common production pattern.

---

### `config/prompts.yaml`

**What it is:** A YAML file that stores all prompt templates used by the system. This is the "prompt versioning" described as a marker of engineering maturity.

**Why this matters:** Prompts are part of your system's architecture. If you hardcode them inside Python files, changing a prompt means changing code, which means your git history mixes prompt experiments with code changes and you can't easily roll back a prompt without rolling back code. Storing them here means you can version them, review them, and A/B test them independently.

**What's inside:**

`rag_answer` — The main RAG prompt, with two parts:
- `system`: Gives Claude strict rules — answer ONLY from context, cite every claim as `[Source N]`, refuse if context is insufficient, never use outside knowledge.
- `human`: The template with `{context}` (the retrieved chunks) and `{question}` (the user's query) injected at runtime.

`faithfulness_check` — A secondary prompt used when you run `--check-faithfulness`:
- Asks Claude to act as a judge and determine whether each claim in the generated answer is actually backed by the context.
- Returns structured JSON: `{"faithful": true/false, "unsupported_claims": [...]}`.

---

### `src/__init__.py`

**What it is:** An empty file that marks `src/` as a Python package, allowing imports like `from src.ingest import ingest`.

---

### `src/ingest.py` — Phase 1: Document Ingestion

**Purpose:** Takes raw documents (PDFs, Markdown files, web pages) and converts them into searchable vector embeddings stored in ChromaDB. This is the "offline" step you run once before the system can answer questions.

**The pipeline it implements:**

```
Source files/URLs
      ↓
  load_documents()     ← Picks the right LangChain loader per file type
      ↓
  chunk_documents()    ← Splits into overlapping chunks
      ↓
  build_vector_store() ← Embeds chunks and stores in ChromaDB
```

**Key functions:**

**`load_documents(source)`**
Inspects the source string and routes it to the correct LangChain document loader:
- URLs starting with `http://` or `https://` → `WebBaseLoader` (from `langchain-community`). Fetches the page, strips HTML boilerplate, returns plain text.
- `.pdf` files → `PyPDFLoader` (from `langchain-community`). Extracts text page by page; each page becomes a separate `Document` object with metadata including `page` number.
- `.md` or `.txt` files → `UnstructuredMarkdownLoader` (from `langchain-community`). Handles markdown structure intelligently.

All loaders return a list of `Document` objects — the standard LangChain unit with `.page_content` (the text) and `.metadata` (source path, page number, etc.).

**`chunk_documents(docs)`**
Takes the loaded documents and splits them into smaller pieces. Uses `RecursiveCharacterTextSplitter` from `langchain`, which is the smart splitter: it tries to split on double newlines first (paragraph boundaries), then single newlines, then sentence ends, then spaces, and only splits mid-word as a last resort. This preserves natural language structure.

The key settings:
- `chunk_size=700` — Each chunk is roughly 700 characters (~500–700 tokens). Large enough to contain useful context, small enough to retrieve precisely.
- `chunk_overlap=100` — Adjacent chunks share 100 characters. This is critical: if a sentence spans a chunk boundary, the overlap ensures it appears fully in one of the two adjacent chunks. Without overlap, you'd slice important sentences in half and lose meaning.
- `separators=["\n\n", "\n", ". ", " ", ""]` — Tells the splitter to prefer paragraph breaks, then line breaks, then sentence ends.

**`build_vector_store(chunks, persist)`**
Takes the chunks and:
1. Creates an `HuggingFaceEmbeddings` object using `BAAI/bge-base-en-v1.5` (from `langchain-huggingface`). This model converts text into a 1536-dimensional vector — a list of numbers where similar texts produce similar vectors.
2. Calls `Chroma.from_documents()` (from `langchain-chroma`) which embeds every chunk and stores them in a local ChromaDB database at `./chroma_db/`. If a database already exists, it adds to it rather than recreating from scratch.

**`ingest(sources)`**
The public entry point — calls all three functions in sequence and returns the `Chroma` vectorstore object.

---

### `src/retriever.py` — Phase 2: Hybrid Retrieval + Reranking

**Purpose:** Given a user's question, find the most relevant chunks from the corpus. This is where most of the production-quality work lives. Simple vector search alone misses too much; this file implements a two-stage retrieval strategy.

**Why not just use vector search?**

Vector/semantic search is great at understanding meaning — "What did the CEO say about growth?" will find chunks about revenue expansion even if they don't contain the word "growth." But it struggles with precise terms. If you search for "Section 4.2.1(b)" or "RFC 7231", a vector model may not match the exact string because the embedding space blurs exact tokens. BM25 handles exact matches perfectly. The solution is to use both.

**The pipeline:**

```
Query
  ↓
EnsembleRetriever
  ├── BM25Retriever (keyword match, weight 0.4)
  └── ChromaDB retriever (semantic similarity, weight 0.6)
  ↓
Fused top-10 candidates (via Reciprocal Rank Fusion)
  ↓
CrossEncoder reranker
  ↓
Top-4 final chunks with scores and citation indices
```

**Key components:**

**`RetrievedChunk` (dataclass)**
A structured object wrapping each retrieved piece of text. Contains:
- `content` — The actual text of the chunk
- `metadata` — Source file, page number, etc. (passed through from the original `Document`)
- `score` — The reranker's score (higher = more relevant)
- `source_index` — A 1-based integer (1, 2, 3, 4) used for `[Source N]` citation labels
- `citation` (property) — Formats as `[Source 1] path/to/file.pdf, p.3`

**`build_hybrid_retriever(vectorstore, all_chunks)`**
Creates a `BM25Retriever` from `langchain-community` over the full chunk corpus, and a vector retriever from the ChromaDB vectorstore. Combines them into an `EnsembleRetriever` (also from `langchain-community`) with weights 0.4/0.6. The EnsembleRetriever uses Reciprocal Rank Fusion (RRF) — it merges the ranked lists from both retrievers by combining their reciprocal ranks, which handles the fact that the two retrievers use incompatible scoring scales.

**`rerank(query, candidates)`**
Takes the 10 candidates from hybrid retrieval and re-scores them with a `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2` from `sentence-transformers`). 

A cross-encoder is fundamentally different from bi-encoder (embedding) models. A bi-encoder embeds the query and each document independently and compares them. A cross-encoder takes the query and a document **together as a pair** and scores their relevance jointly — it can see both at once, which means it captures subtle interactions like negation, specificity, and intent far better. It's slower (you can't pre-compute embeddings) but dramatically more precise. This is why it's used as a re-ranker on the smaller candidate set rather than for initial retrieval over the full corpus.

Returns the top 4 (configurable via `TOP_K_FINAL`) ranked by cross-encoder score.

**`retrieve(query, vectorstore, all_chunks)`**
The public function. Calls `build_hybrid_retriever`, runs the query, then calls `rerank`, and wraps the results as `RetrievedChunk` objects with 1-based source indices.

---

### `src/chain.py` — Phase 2: Answer Generation

**Purpose:** Takes the retrieved chunks and generates a cited answer using Claude. This is where citation enforcement, prompt versioning, and the LCEL chain live.

**Key components:**

**`load_prompts()`**
Reads `config/prompts.yaml` and returns the parsed dict. Called fresh on each request, so you can update prompts without restarting the server.

**`get_llm()`**
Returns a `ChatGroq` instance (from `langchain-groq`) pointing at `llama-3.3-70b-versatile`. Temperature is set to 0 for deterministic, non-creative output — you want consistent citations, not creative paraphrasing.

**`build_context_string(chunks)`**
Formats the list of `RetrievedChunk` objects into a numbered block like:

```
[Source 1] (docs/paper.pdf, p.12)
<chunk text here>

---

[Source 2] (docs/guide.md)
<chunk text here>
```

This structure is what lets the LLM write `[Source 1]` in its answer — it can see the label right in the context.

**`build_rag_chain()`**
Constructs the LangChain Expression Language (LCEL) chain:

```python
chain = ChatPromptTemplate | ChatGroq | StrOutputParser
```

`|` is LCEL's pipe operator — it wires the output of one component as the input to the next. `ChatPromptTemplate` (from `langchain-core`) fills in the `{context}` and `{question}` variables from `prompts.yaml`. `ChatGroq` calls Claude. `StrOutputParser` (from `langchain-core`) extracts just the text string from Claude's response object.

**`answer(question, chunks)`**
The main public function. Builds the context string, runs the chain, detects if Claude declined to answer (by checking for "cannot answer" in the response), and returns a dict with the answer text, source citations, and a `refused` boolean.

**`check_faithfulness(answer_text, chunks)`**
A secondary chain that sends the generated answer and the original context back to Claude with a different prompt — asking it to act as a judge and return JSON identifying any claims in the answer not supported by the context. This is a lightweight self-verification layer available on demand.

---

### `src/evaluate.py` — Phase 3: Offline Evaluation

**Purpose:** Systematically measures the quality of the entire RAG pipeline against a curated set of known-good question/answer pairs. Designed to be run in CI/CD so regressions are caught before they reach production.

**Why this matters:** A RAG system can silently degrade. Someone changes a prompt, adjusts chunk size, swaps the embedding model — and precision quietly drops. Without automated evaluation, you find out when users complain. This file makes quality degradation a build failure.

**The pipeline:**

```
golden_dataset.json (curated Q&A pairs)
      ↓
run_rag_on_dataset()   ← Runs the full retrieval + answer pipeline on each question
      ↓
run_ragas_eval()       ← Scores each Q&A with RAGAS metrics
      ↓
print_results_table()  ← Shows scores vs threshold
      ↓
sys.exit(0 or 1)       ← 1 = build failure in CI/CD
```

**Key functions:**

**`load_golden_dataset(path)`**
Reads `eval/golden_dataset.json` — a list of `{"question": "...", "ground_truth": "..."}` objects that you curate manually. These should cover typical queries, edge cases, and questions the system should refuse. Aim for 50–200 pairs.

**`run_rag_on_dataset(golden, vectorstore, all_chunks)`**
Loops through every golden question, runs the full `retrieve()` + `answer()` pipeline on each, and collects four parallel lists:
- `questions` — the input questions
- `answers` — what the system generated
- `contexts` — the chunk texts that were retrieved (as lists of strings)
- `ground_truths` — the correct answers from your golden dataset

These four lists are exactly what RAGAS expects.

**`run_ragas_eval(dataset_dict)`**
Converts the data into a HuggingFace `Dataset` object (what RAGAS requires) and runs three metrics:

- **`faithfulness`** — For each claim in the generated answer, does it have support in the retrieved context? Score 0–1. Low score means the system is hallucinating content not in the documents.
- **`answer_relevancy`** — Does the answer actually address the question? Score 0–1. Low score means the system is answering the wrong thing or being evasive.
- **`context_precision`** — Are the top-ranked retrieved chunks actually relevant to the question? Score 0–1. Low score means the retrieval pipeline is surfacing the wrong passages.

RAGAS wraps the `ChatGroq` and `HuggingFaceEmbeddings` models via `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper` so it can use them internally for its own LLM-based scoring.

**`print_results_table(results, threshold)`**
Uses `rich.Table` to display a formatted table of each metric's score vs the threshold, with colored PASS/FAIL status. Returns the average score.

**`main()`**
Orchestrates everything and crucially calls `sys.exit(1)` if the average score falls below the threshold. A non-zero exit code is what CI/CD systems (GitHub Actions, Jenkins, etc.) treat as a build failure — no further explanation needed.

---

### `main.py` — CLI Entry Point

**Purpose:** A single file that wires all three phases into one command-line tool with three subcommands. You never need to run individual source files directly.

**Subcommands:**

**`python main.py ingest [sources...]`**
Calls `src/ingest.py`. If you pass file paths or URLs as arguments, those are ingested. If you pass nothing, it auto-discovers all PDFs and Markdown files in the `docs/` folder.

**`python main.py ask "your question" [--check-faithfulness]`**
1. Loads the existing ChromaDB vectorstore
2. Reconstructs the full chunk list (needed for BM25)
3. Calls `retrieve()` from `src/retriever.py`
4. Calls `answer()` from `src/chain.py`
5. Displays the answer in a `rich.Panel` with formatted Markdown, followed by the source citations
6. If `--check-faithfulness` is passed, also runs `check_faithfulness()` and reports any unsupported claims

**`python main.py evaluate [--dataset path] [--threshold 0.70]`**
Delegates to `src/evaluate.py`. The default threshold is 0.70 — if the average RAGAS score drops below 70%, the command exits with code 1.

**Why lazy imports?** Each subcommand uses `from src.xxx import yyy` inside the function body rather than at the top of the file. This means `python main.py ingest` doesn't load sentence-transformers or RAGAS, and `python main.py ask` doesn't load RAGAS. Startup is faster and errors from missing dependencies only appear when you actually use that subcommand.

---

### `eval/golden_dataset.json`

**What it is:** A JSON array of manually verified question/answer pairs used as the ground truth for evaluation.

**How to build it well:** These should not be generated by the LLM — the whole point is to have human-verified correct answers. Go through your corpus, write questions a real user would ask, and write the correct answer from the document. Include:
- Standard factual questions the system should answer well
- Questions about specific terms (testing BM25's keyword matching)
- Questions requiring understanding context (testing semantic search)
- Questions that are NOT in the documents (the system should refuse these — include them with `"ground_truth": "This information is not available in the provided documents."`)

The recommended size is 50–200 pairs. Fewer and the evaluation is too noisy; more and it covers your domain well enough to catch most regressions.

---

### `.github/workflows/eval.yml` — CI/CD Pipeline

**What it is:** A GitHub Actions workflow that automatically runs the evaluation pipeline on every push to `main` and every pull request.

**What it does, step by step:**

1. **Checkout** — Pulls the code from the repository
2. **Set up Python 3.11** — Installs the Python version
3. **Install dependencies** — Runs `pip install -r requirements.txt`
4. **Ingest test documents** — Runs `python main.py ingest docs/` to populate ChromaDB with the test corpus
5. **Run RAG evaluation** — Runs `python main.py evaluate --dataset eval/golden_dataset.json --threshold 0.70`

If step 5 exits with code 1 (quality below threshold), GitHub Actions marks the check as failed. A failing check blocks the PR from being merged (if branch protection rules are configured). This is the mechanism that catches quality regressions automatically.

**Secrets:** `HF_TOKEN` and `GROQ_API_KEY` are stored as GitHub repository secrets (Settings → Secrets and variables → Actions), not in the code. The workflow reads them as environment variables.

---

## How the Phases Connect

The three phases are designed to be additive — Phase 1 produces the ChromaDB store that Phase 2 reads; Phase 2 produces answers that Phase 3 evaluates; Phase 3 wraps Phase 1 + 2 to measure the full system.

```
Phase 1: ingest.py
  └── writes: chroma_db/ (vector store on disk)

Phase 2: retriever.py + chain.py
  ├── reads: chroma_db/
  └── produces: answers with [Source N] citations

Phase 3: evaluate.py
  ├── runs Phase 2 on every question in golden_dataset.json
  ├── scores with RAGAS
  └── exits 0 (pass) or 1 (fail) → CI/CD gate
```

---

## LangChain Packages Summary

| Package from docs.langchain.com | Used in | For |
|---|---|---|
| `langchain` | `ingest.py`, `retriever.py` | `RecursiveCharacterTextSplitter`, `EnsembleRetriever` |
| `langchain-core` | `chain.py`, `retriever.py` | `Document`, `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough` |
| `langchain-community` | `ingest.py`, `retriever.py` | `PyPDFLoader`, `UnstructuredMarkdownLoader`, `WebBaseLoader`, `BM25Retriever` |
| `langchain-groq` | `chain.py`, `evaluate.py` | `ChatGroq` (Claude as LLM) |
| `langchain-huggingface | `BAAI/bge-base-en-v1.5` — free, local, 768-dim) |
| `langchain-chroma` | `ingest.py`, `retriever.py`, `evaluate.py` | `Chroma` (persistent vector store) |
