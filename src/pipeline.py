"""
pipeline.py — End-to-end Legal RAG orchestrator.

Wires together:
  query → embed → hybrid retrieve (FAISS + BM25) → cross-encoder rerank → Gemini generate

Usage:
  from src.pipeline import LegalRAGPipeline
  pipeline = LegalRAGPipeline()
  result   = pipeline.query("What is the punishment for murder under IPC?")
  print(result["answer"])
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


class LegalRAGPipeline:
    """
    Production-ready Legal RAG pipeline.

    Lazy-loads all heavy components (embedding model, FAISS index, reranker)
    on first use so that import time is minimal.

    Attributes:
        _vector_store  : FAISSVectorStore (loaded on first query).
        _bm25          : BM25Retriever    (loaded on first query).
        _hybrid        : HybridRetriever  (composed on first query).
        _ready         : Flag indicating whether components are initialised.
    """

    def __init__(self) -> None:
        self._vector_store = None
        self._bm25         = None
        self._hybrid       = None
        self._ready        = False

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _load_components(self) -> None:
        """Load all retrieval components from disk artifacts."""
        if self._ready:
            return

        logger.info("Initialising Legal RAG pipeline components…")
        t0 = time.time()

        # 1. FAISS vector store
        from vector_store import FAISSVectorStore  # type: ignore
        self._vector_store = FAISSVectorStore()
        self._vector_store.load()

        # 2. BM25 retriever
        from bm25_retriever import BM25Retriever  # type: ignore
        self._bm25 = BM25Retriever.from_chunks_file()

        # 3. Hybrid retriever
        from hybrid_retriever import HybridRetriever  # type: ignore
        self._hybrid = HybridRetriever(
            vector_store=self._vector_store,
            bm25_retriever=self._bm25,
            alpha=config.HYBRID_ALPHA,
            top_k=config.TOP_K_RETRIEVAL,
        )

        self._ready = True
        logger.info(f"Pipeline ready in {time.time() - t0:.2f}s.")

    # ──────────────────────────────────────────────────────────────────────────
    # Public query method
    # ──────────────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict:
        """
        Run the full RAG pipeline for a legal question.

        Steps:
          1. Embed the query using BGE-large-en.
          2. Retrieve top candidates via hybrid retrieval (FAISS + BM25 / RRF).
          3. Rerank candidates with the BGE cross-encoder.
          4. Generate an answer with Gemini using the top-K reranked chunks.

        Args:
            question: The user's legal question string.

        Returns:
            Dict with keys:
              • 'answer'   : Generated answer string.
              • 'sources'  : List of source document names (deduplicated).
              • 'chunks'   : List of top-K chunk dicts (text + metadata).
              • 'query'    : The original question.
              • 'latency_s': Total pipeline latency in seconds.
        """
        self._load_components()

        t0 = time.time()

        # ── Step 1: Embed query ───────────────────────────────────────────────
        from embeddings import embed_query  # type: ignore
        query_vector = embed_query(question)

        # ── Step 2: Hybrid retrieval ──────────────────────────────────────────
        candidates = self._hybrid.retrieve(question, query_vector=query_vector)
        logger.info(f"Hybrid retrieval returned {len(candidates)} candidates.")

        # ── Step 3: Cross-encoder reranking ───────────────────────────────────
        from reranker import rerank  # type: ignore
        top_chunks_with_scores = rerank(question, candidates, top_k=config.TOP_K_RERANK)

        top_chunks = [chunk for chunk, _ in top_chunks_with_scores]
        logger.info(f"Reranked to {len(top_chunks)} final chunks.")

        # ── Step 4: Generate answer ────────────────────────────────────────────
        from generator import generate_answer  # type: ignore
        answer = generate_answer(question, top_chunks)

        # ── Collate sources ───────────────────────────────────────────────────
        seen = set()
        sources: List[str] = []
        for chunk in top_chunks:
            src = chunk.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        latency = round(time.time() - t0, 2)
        logger.info(f"Pipeline complete in {latency}s.")

        return {
            "query":     question,
            "answer":    answer,
            "sources":   sources,
            "chunks":    top_chunks,
            "latency_s": latency,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point — interactive REPL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import readline  # noqa: F401  (enables arrow keys on Unix)

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n⚖️  Legal RAG Pipeline — Interactive Mode")
    print("   Type your legal question, or 'exit' to quit.\n")

    pipeline = LegalRAGPipeline()

    while True:
        try:
            q = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if q.lower() in {"exit", "quit", "q", ""}:
            print("Goodbye.")
            break

        result = pipeline.query(q)
        print("\n" + "═" * 70)
        print(f"ANSWER:\n{result['answer']}")
        print(f"\nSOURCES: {', '.join(result['sources']) or 'N/A'}")
        print(f"LATENCY: {result['latency_s']}s")
        print("═" * 70 + "\n")
