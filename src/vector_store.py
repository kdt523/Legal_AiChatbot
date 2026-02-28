"""
vector_store.py — FAISS-based dense vector index.

Wraps a faiss.IndexFlatIP (inner-product / cosine similarity after L2 normalisation)
for efficient approximate nearest-neighbour search over chunk embeddings.

Artifacts:
  • artifacts/faiss_index/   – saved FAISS index directory
  • Metadata is loaded from artifacts/metadata.json via embeddings.load_embeddings()
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Lightweight FAISS vector store for dense retrieval.

    Attributes:
        index     : faiss index object (IndexFlatIP).
        metadata  : list of chunk dicts aligned to index row order.
    """

    def __init__(self) -> None:
        self.index    = None
        self.metadata: List[Dict] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Building / saving
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        """
        Build a new FAISS IndexFlatIP from pre-computed L2-normalised embeddings.

        We use inner-product (IP) because the embedding model already outputs
        L2-normalised vectors, making cosine similarity ≡ inner product.

        Args:
            embeddings: float32 array of shape (N, dim).
            metadata:   List of N chunk dicts aligned to row order.
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")

        dim = embeddings.shape[1]
        logger.info(f"Building FAISS IndexFlatIP  dim={dim}  vectors={len(embeddings)}")

        self.index    = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)              # add all vectors at once
        self.metadata = metadata

        logger.info(f"FAISS index built. Total vectors: {self.index.ntotal}")

    def save(self, path: Path = config.FAISS_INDEX_PATH) -> None:
        """
        Persist the FAISS index to disk.

        Args:
            path: Directory (will be created) where the index binary is stored.
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError("faiss-cpu is required.")

        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.bin"))
        logger.info(f"FAISS index saved → {path / 'index.bin'}")

    def load(self, path: Path = config.FAISS_INDEX_PATH) -> None:
        """
        Load a previously saved FAISS index and the aligned metadata.

        Args:
            path: Directory containing 'index.bin'.
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            raise ImportError("faiss-cpu is required.")

        index_file = path / "index.bin"
        if not index_file.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_file}\n"
                "Run: python src/vector_store.py"
            )

        self.index = faiss.read_index(str(index_file))
        logger.info(f"FAISS index loaded. Vectors: {self.index.ntotal}")

        # Load aligned metadata
        from embeddings import load_embeddings  # type: ignore
        _, self.metadata = load_embeddings()

    # ──────────────────────────────────────────────────────────────────────────
    # Searching
    # ──────────────────────────────────────────────────────────────────────────

    def search(self, query_vector: np.ndarray, top_k: int = config.TOP_K_RETRIEVAL
               ) -> List[Tuple[Dict, float]]:
        """
        Return the top-k most similar chunks for a query vector.

        Args:
            query_vector: 1-D float32 array of shape (dim,).
            top_k:        Number of results to return.

        Returns:
            List of (chunk_dict, score) tuples sorted by descending score.
        """
        if self.index is None:
            raise RuntimeError("Index is not loaded. Call build() or load() first.")

        qvec = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(qvec, top_k)

        results: List[Tuple[Dict, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for empty slots
                continue
            results.append((self.metadata[idx], float(score)))

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def load_vector_store() -> FAISSVectorStore:
    """
    Load the FAISS index from the default artifact path.

    Returns:
        Initialised FAISSVectorStore ready for search().
    """
    vs = FAISSVectorStore()
    vs.load()
    return vs


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point — build and save the index
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=config.LOG_LEVEL)

    from embeddings import load_embeddings  # type: ignore

    embeddings, metadata = load_embeddings()

    vs = FAISSVectorStore()
    vs.build(embeddings, metadata)
    vs.save()

    print(f"\nFAISS index built and saved. Total vectors: {vs.index.ntotal}")
