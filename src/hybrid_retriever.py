"""
hybrid_retriever.py — Combine dense (FAISS) and sparse (BM25) retrieval.

Reciprocal Rank Fusion (RRF) is used instead of raw score normalisation because:
  • BM25 and cosine scores live on different scales.
  • RRF is parameter-free and robust to score distribution differences.
  • A configurable α weight (HYBRID_ALPHA) still allows tuning the blend.

Formula (per candidate chunk):
    rrf_score = α * (1 / (rrf_k + dense_rank)) + (1-α) * (1 / (rrf_k + bm25_rank))

where rrf_k = 60 (standard constant from the original RRF paper).
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

_RRF_K = 60          # standard Reciprocal Rank Fusion constant


class HybridRetriever:
    """
    Hybrid retriever that fuses dense vector search and BM25 keyword search.

    Args:
        vector_store   : Initialised FAISSVectorStore instance.
        bm25_retriever : Initialised BM25Retriever instance.
        alpha          : Weight of dense retrieval (0–1).
                         alpha=1.0 → pure vector,  alpha=0.0 → pure BM25.
        top_k          : Number of final candidates to return before reranking.
    """

    def __init__(self, vector_store, bm25_retriever,
                 alpha: float = config.HYBRID_ALPHA,
                 top_k: int = config.TOP_K_RETRIEVAL) -> None:
        self.vector_store    = vector_store
        self.bm25_retriever  = bm25_retriever
        self.alpha           = alpha
        self.top_k           = top_k

    def retrieve(self, query: str, query_vector=None) -> List[Tuple[Dict, float]]:
        """
        Retrieve the top-k chunks using hybrid RRF fusion.

        Args:
            query:        The user's question (used for BM25).
            query_vector: Pre-computed query embedding (1-D float32).
                          If None, it is computed on the fly via embeddings.embed_query().

        Returns:
            Sorted list of (chunk_dict, hybrid_score) tuples, best first.
        """
        # ── 1. Ensure we have a query vector ─────────────────────────────────
        if query_vector is None:
            from embeddings import embed_query  # type: ignore
            query_vector = embed_query(query)

        # ── 2. Fetch candidates from both systems ─────────────────────────────
        fetch_k = self.top_k * 3     # over-fetch to improve fusion coverage

        dense_results = self.vector_store.search(query_vector, top_k=fetch_k)
        bm25_results  = self.bm25_retriever.search(query,      top_k=fetch_k)

        # ── 3. Build rank maps keyed by chunk_id ──────────────────────────────
        dense_rank: Dict[str, int] = {
            chunk["chunk_id"]: rank
            for rank, (chunk, _) in enumerate(dense_results, start=1)
        }
        bm25_rank: Dict[str, int] = {
            chunk["chunk_id"]: rank
            for rank, (chunk, _) in enumerate(bm25_results, start=1)
        }

        # ── 4. Build unified candidate pool ───────────────────────────────────
        candidates: Dict[str, Dict] = {}
        for chunk, _ in dense_results + bm25_results:
            candidates.setdefault(chunk["chunk_id"], chunk)

        # ── 5. Compute RRF-weighted hybrid score for each candidate ───────────
        scored: List[Tuple[Dict, float]] = []
        for cid, chunk in candidates.items():
            d_rank = dense_rank.get(cid, fetch_k + 1)
            b_rank = bm25_rank.get(cid,  fetch_k + 1)

            hybrid_score = (
                self.alpha       * (1.0 / (_RRF_K + d_rank))
                + (1 - self.alpha) * (1.0 / (_RRF_K + b_rank))
            )
            scored.append((chunk, hybrid_score))

        # ── 6. Sort and return top_k ───────────────────────────────────────────
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.top_k]
