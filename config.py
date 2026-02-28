"""
config.py — Central configuration for the Legal RAG system.

All tunable parameters, model names, and path constants live here so that
changing a single value propagates through the entire pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Project root (one level above this file)
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ──────────────────────────────────────────────
# Data paths
# ──────────────────────────────────────────────
DATA_DIR           = ROOT_DIR / "data"
RAW_DIR            = DATA_DIR / "raw"
CLEANED_DIR        = DATA_DIR / "cleaned"
PROCESSED_CHUNKS   = DATA_DIR / "processed_chunks.json"
TEST_QUESTIONS_CSV = DATA_DIR / "test_questions.csv"

# ──────────────────────────────────────────────
# Artifact paths
# ──────────────────────────────────────────────
ARTIFACTS_DIR    = ROOT_DIR / "artifacts"
EMBEDDINGS_PATH  = ARTIFACTS_DIR / "embeddings.npy"
METADATA_PATH    = ARTIFACTS_DIR / "metadata.json"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index"

# ──────────────────────────────────────────────
# Embedding model
# bge-small-en-v1.5 : 133 MB, 384-dim, low RAM
# bge-base-en-v1.5  : 440 MB, 768-dim, balanced
# bge-large-en      : 1.3 GB, 1024-dim, best quality (high RAM)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE    = 800     # tokens (approx characters ÷ 4)
CHUNK_OVERLAP = 150     # token overlap between consecutive chunks

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K_RETRIEVAL = 20    # candidates fetched before reranking (larger pool = better BM25 coverage)
TOP_K_RERANK    = 3     # final chunks sent to the generator
HYBRID_ALPHA    = 0.5   # 50/50 dense + BM25 blend (balanced for legal keyword queries)

# ──────────────────────────────────────────────
# Reranker
# ms-marco-MiniLM-L-6-v2 : ~80 MB, fast, low RAM
# ms-marco-MiniLM-L-12-v2: ~130 MB, slightly better
# BAAI/bge-reranker-large : 1.1 GB, best quality (high RAM)
# ──────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ──────────────────────────────────────────────
# Gemini
# ──────────────────────────────────────────────
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = 0.2

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = "INFO"
