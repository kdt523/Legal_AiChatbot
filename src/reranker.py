"""
reranker.py — Cross-encoder reranking of retrieved candidate chunks.

Model: BAAI/bge-reranker-large
A cross-encoder scores each (query, passage) pair jointly, giving much higher
accuracy than bi-encoder approximate search at the cost of O(k) inference calls.
We rerank the hybrid retrieval candidates and keep only the top-K.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Singleton model loader
# ──────────────────────────────────────────────────────────────────────────────

_cross_encoder = None


def get_reranker():
    """
    Lazily load the BGE cross-encoder reranker.

    Returns:
        sentence_transformers.CrossEncoder instance.
    """
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        logger.info(f"Loading reranker: {config.RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(
            config.RERANKER_MODEL,
            max_length=512,
        )
        logger.info("Reranker loaded.")
    return _cross_encoder


# ──────────────────────────────────────────────────────────────────────────────
# Public reranking function
# ──────────────────────────────────────────────────────────────────────────────

def rerank(query: str,
           candidates: List[Tuple[Dict, float]],
           top_k: int = config.TOP_K_RERANK) -> List[Tuple[Dict, float]]:
    """
    Rerank a list of candidate (chunk, score) pairs using a cross-encoder.

    The cross-encoder replaces the first-stage retrieval scores entirely with
    higher-quality joint query-passage relevance scores.

    Args:
        query:      The original user query.
        candidates: Output of hybrid_retriever.retrieve() —
                    list of (chunk_dict, hybrid_score) tuples.
        top_k:      Number of top-ranked chunks to return.

    Returns:
        Sorted list of (chunk_dict, reranker_score) tuples, best first.
        Length is at most top_k.
    """
    if not candidates:
        return []

    ce = get_reranker()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk, _ in candidates]

    # Score all pairs in a single forward pass (batched internally)
    scores: List[float] = ce.predict(pairs, batch_size=16).tolist()

    # Pair each chunk with its new cross-encoder score
    reranked = [
        (chunk, float(score))
        for (chunk, _), score in zip(candidates, scores)
    ]

    # Sort descending by cross-encoder score
    reranked.sort(key=lambda x: x[1], reverse=True)

    top = reranked[:top_k]
    logger.debug(
        f"Reranked {len(candidates)} candidates → kept top {len(top)}. "
        f"Best score: {top[0][1]:.4f}" if top else ""
    )
    return top
