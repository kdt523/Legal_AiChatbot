"""
embeddings.py — Create and persist dense vector embeddings for all text chunks.

Model: BAAI/bge-large-en (1024-dim) via sentence-transformers.
Artifacts saved:
  • artifacts/embeddings.npy   – numpy array of shape (N, 1024)
  • artifacts/metadata.json    – list of chunk dicts aligned to embedding rows
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Model loader (singleton pattern to avoid reloading on every call)
# ──────────────────────────────────────────────────────────────────────────────

_model = None


def get_embedding_model():
    """
    Lazily load the sentence-transformer embedding model.

    Returns:
        sentence_transformers.SentenceTransformer instance.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


# ──────────────────────────────────────────────────────────────────────────────
# Core embedding functions
# ──────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str], batch_size: int = 32,
                show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of strings into dense vectors using BGE-large-en.

    BGE models expect a query prefix "Represent this sentence: " for passages.
    We rely on the model's default encoding behaviour (no manual prefix needed
    for passage encoding).

    Args:
        texts:         List of strings to embed.
        batch_size:    Mini-batch size for GPU/CPU inference.
        show_progress: Show tqdm progress bar.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype float32.
    """
    model = get_embedding_model()
    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size}) ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalise for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string with the BGE query prefix.

    BGE models expect "Represent this question for searching relevant passages: "
    prepended to the query for best retrieval performance.

    Args:
        query: The user's question or search query.

    Returns:
        np.ndarray of shape (EMBEDDING_DIM,), dtype float32.
    """
    model = get_embedding_model()
    # BGE v1.5 query instruction for retrieval
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    vec = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec[0].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_embeddings(embeddings: np.ndarray, metadata: List[Dict]) -> None:
    """
    Persist embeddings and aligned chunk metadata to disk.

    Args:
        embeddings: Array of shape (N, dim).
        metadata:   List of N chunk dicts (same order as embeddings).
    """
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(str(config.EMBEDDINGS_PATH), embeddings)
    logger.info(f"Embeddings saved → {config.EMBEDDINGS_PATH}  shape={embeddings.shape}")

    with open(config.METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)
    logger.info(f"Metadata saved  → {config.METADATA_PATH}  ({len(metadata)} entries)")


def load_embeddings() -> Tuple[np.ndarray, List[Dict]]:
    """
    Load previously saved embeddings and metadata from disk.

    Returns:
        Tuple of (embeddings np.ndarray, metadata list).

    Raises:
        FileNotFoundError: If artifacts are missing (run ingestion + embedding first).
    """
    if not config.EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {config.EMBEDDINGS_PATH}\n"
            "Run: python src/embeddings.py"
        )
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {config.METADATA_PATH}\n"
            "Run: python src/embeddings.py"
        )

    embeddings = np.load(str(config.EMBEDDINGS_PATH)).astype(np.float32)
    with open(config.METADATA_PATH, encoding="utf-8") as fh:
        metadata = json.load(fh)

    logger.info(f"Loaded embeddings: shape={embeddings.shape}, chunks={len(metadata)}")
    return embeddings, metadata


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline helper
# ──────────────────────────────────────────────────────────────────────────────

def build_embeddings(chunks: List[Dict] = None) -> Tuple[np.ndarray, List[Dict]]:
    """
    Build embeddings for all chunks and save artifacts.

    If *chunks* is None, the function loads them from config.PROCESSED_CHUNKS.

    Args:
        chunks: Optional pre-loaded list of chunk dicts.

    Returns:
        Tuple of (embeddings, metadata).
    """
    if chunks is None:
        from chunking import load_chunks  # type: ignore
        chunks = load_chunks()

    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    save_embeddings(embeddings, chunks)
    return embeddings, chunks


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    embs, meta = build_embeddings()
    print(f"\nEmbeddings built: {embs.shape}  |  Metadata entries: {len(meta)}")
