# ⚖️ Legal RAG — Production-Quality Legal AI Assistant

A retrieval-augmented generation (RAG) system for Indian law (IPC, CrPC, Constitution, etc.) built to
internship/portfolio standard.  Combines **hybrid search**, **cross-encoder reranking**, and **Google Gemini**
to answer legal questions with source citations.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Running the Ingestion Pipeline](#running-the-ingestion-pipeline)
8. [Running the App](#running-the-app)
9. [Evaluation](#evaluation)
10. [Evaluation Results](#evaluation-results)
11. [Design Decisions](#design-decisions)
12. [Contributing](#contributing)

---

## Project Overview

Legal RAG is an AI assistant that answers questions about Indian law by:

- Searching a corpus of legal text (IPC, CrPC, CPC, IEA, HMA, Motor Vehicles Act, etc.)
- Grounding every answer in retrieved passages (no hallucination beyond context)
- Citing the source document for transparency

This project was built by **refactoring and merging** three earlier prototypes into a single,
clean, modular codebase suitable for production review.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit UI (app/)                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │  query(question)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    pipeline.py  (orchestrator)                   │
│                                                                  │
│  ┌──────────────┐     ┌───────────────────────────────────────┐ │
│  │ embeddings.py│────▶│         Hybrid Retriever              │ │
│  │ (BGE query)  │     │  ┌──────────────┐ ┌───────────────┐  │ │
│  └──────────────┘     │  │ vector_store │ │ bm25_retriever│  │ │
│                        │  │  (FAISS IP)  │ │  (BM25Okapi)  │  │ │
│                        │  └──────┬───────┘ └───────┬───────┘  │ │
│                        │         └────────┬──────────┘          │ │
│                        │           RRF fusion (α=0.7)           │ │
│                        └────────────────┬──────────────────────┘ │
│                                         │                        │
│                                         ▼                        │
│                             ┌────────────────────┐              │
│                             │   reranker.py       │              │
│                             │ (BGE cross-encoder) │              │
│                             └─────────┬──────────┘              │
│                                       │ top-3 chunks             │
│                                       ▼                          │
│                             ┌────────────────────┐              │
│                             │  generator.py       │              │
│                             │  (Google Gemini)    │              │
│                             └─────────┬──────────┘              │
└───────────────────────────────────────┼────────────────────────┘
                                        │
                              answer + sources + chunks
```

---

## How the RAG Pipeline Works

### Stage 1 — Ingestion & Chunking

Raw legal JSON / TXT / PDF files are loaded from `data/raw/`, cleaned
(whitespace normalisation, character filtering), and split into overlapping
chunks of ~800 tokens (3200 characters) with 150-token overlap.

Overlap prevents answers from being cut at chunk boundaries.
Chunks are saved to `data/processed_chunks.json`.

### Stage 2 — Embedding

Each chunk is embedded using **BAAI/bge-large-en** (1024-dim, state-of-the-art
English retrieval model).  Embeddings are L2-normalised for cosine similarity
and saved to `artifacts/embeddings.npy`.

### Stage 3 — Indexing

A **FAISS IndexFlatIP** (exact inner-product search) is built over the chunk
embeddings.  Being exact (not approximate) ensures no candidates are dropped at
the retrieval stage.

A **BM25Okapi** index (rank-bm25) is built over the tokenised chunk texts for
keyword-level matching.

### Stage 4 — Hybrid Retrieval

At query time:

1. The query is embedded with the BGE query instruction prefix.
2. FAISS returns the top-N dense candidates.
3. BM25 returns the top-N sparse candidates.
4. **Reciprocal Rank Fusion (RRF)** merges both result lists:

   ```
   hybrid_score = α * (1/(60 + dense_rank)) + (1-α) * (1/(60 + bm25_rank))
   ```

   Default α = 0.7 (vector-heavy blend).  RRF is scale-invariant and robust
   to score distribution differences between the two retrievers.

### Stage 5 — Cross-Encoder Reranking

The top-10 RRF candidates are passed to **BAAI/bge-reranker-large**, a
cross-encoder that scores each (query, passage) pair jointly.  The top-3
reranked chunks are kept.

Cross-encoders significantly outperform bi-encoder cosine similarity for
reranking because they model query-passage interaction directly.

### Stage 6 — Grounded Generation

The 3 top-ranked chunks are formatted into a grounded prompt and sent to
**Google Gemini** (gemini-2.5-flash → 1.5-flash fallback).

The system instruction explicitly tells Gemini:
- Answer only from the provided context.
- Say "Not found in legal database." if the answer is absent.
- Cite source document names.

---

## Tech Stack

| Component         | Library / Model                     |
|-------------------|-------------------------------------|
| Embeddings        | BAAI/bge-large-en (sentence-transformers) |
| Dense Index       | FAISS IndexFlatIP (faiss-cpu)       |
| Sparse Search     | BM25Okapi (rank-bm25)               |
| Fusion            | Reciprocal Rank Fusion (custom)     |
| Reranker          | BAAI/bge-reranker-large             |
| LLM               | Google Gemini (google-generativeai) |
| UI                | Streamlit                           |
| Config            | python-dotenv                       |
| PDF loading       | PyMuPDF (fitz)                      |

---

## Project Structure

```
legal-rag-final/
│
├── data/
│   ├── raw/                  ← place your .txt / .json / .pdf files here
│   ├── cleaned/              ← auto-generated cleaned text files
│   ├── processed_chunks.json ← auto-generated by chunking step
│   └── test_questions.csv    ← evaluation question set
│
├── artifacts/                ← auto-generated ML artifacts
│   ├── embeddings.npy
│   ├── metadata.json
│   └── faiss_index/
│       └── index.bin
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py          ← load & clean raw documents
│   ├── chunking.py           ← split into overlapping chunks
│   ├── embeddings.py         ← BGE embedding + persistence
│   ├── vector_store.py       ← FAISS index build/load/search
│   ├── bm25_retriever.py     ← BM25 keyword search
│   ├── hybrid_retriever.py   ← RRF fusion of dense + sparse
│   ├── reranker.py           ← BGE cross-encoder reranking
│   ├── generator.py          ← Gemini grounded generation
│   ├── pipeline.py           ← end-to-end orchestrator
│   └── evaluation.py         ← offline evaluation pipeline
│
├── app/
│   └── streamlit_app.py      ← Streamlit UI
│
├── config.py                 ← all configuration constants
├── seed_data.py              ← one-time data copy script
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- A Google Gemini API key ([get one free](https://aistudio.google.com/))

### 1. Clone / enter the project

```bash
cd legal-rag-final
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

---

## Running the Ingestion Pipeline

Run these steps **once** to build all artifacts:

```bash
# Step 0 — Copy legal datasets from source repos (if available)
python seed_data.py

# Step 1 — Load raw files, clean, write to data/cleaned/
python src/ingestion.py

# Step 2 — Split cleaned documents into overlapping chunks
python src/chunking.py

# Step 3 — Embed all chunks (downloads ~1.3 GB BGE model on first run)
python src/embeddings.py

# Step 4 — Build and save FAISS index
python src/vector_store.py
```

> **Tip:** Place any additional `.txt`, `.json`, or `.pdf` legal files in
> `data/raw/` before running.  The ingestion step handles all three formats.

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

For the interactive CLI (no UI):

```bash
python src/pipeline.py
```

---

## Evaluation

```bash
# Run all 10 sample questions, print summary metrics
python src/evaluation.py

# Interactive scoring (manually label each answer)
python src/evaluation.py --interactive

# Limit to first 5 questions
python src/evaluation.py --limit 5
```

Results are saved to `data/eval_results.csv`.

---

## Evaluation Results

The system does not claim fabricated accuracy numbers.  Instead, three transparent
metrics are reported:

| Metric                  | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| **Retrieval success rate** | Fraction of queries where ≥1 relevant chunk was retrieved.     |
| **Grounded answer rate**   | Fraction of answers that did not trigger the "not found" fallback. |
| **Keyword overlap (Jaccard)** | Average token overlap between generated answer and reference.  |
| **Manual accuracy**        | Human-scored accuracy from interactive evaluation mode.         |

Due to cross-encoder reranking and the BGE-large embedding model, retrieval quality
is substantially better than TF-IDF or small embedding models.  Run the evaluation
script on your own machine to get figures tied to your specific dataset.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **RRF over weighted score normalisation** | BM25 and cosine scores have incompatible distributions; RRF is scale-invariant and requires no calibration. |
| **FAISS IndexFlatIP** | For datasets up to ~1M chunks, exact search is fast enough and avoids ANN approximation errors. |
| **BGE-large-en** | Top performer on BEIR benchmark for English retrieval tasks; suffix `-en` ensures English-specific training. |
| **Cross-encoder reranking** | Bi-encoder cosine similarity cannot model query-passage interaction; cross-encoders give substantially better precision. |
| **Gemini fallback chain** | Quota limits cause unpredictable failures; 2.5-flash → 1.5-flash → 1.5-pro ensures high availability. |
| **Grounded system prompt** | Explicitly constraining the LLM to the provided context prevents hallucination in the legal domain. |

---

## Contributing

Pull requests are welcome.  For major changes, please open an issue first.

---

*Built as a portfolio project demonstrating production-quality RAG engineering.*
