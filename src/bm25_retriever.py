"""
bm25_retriever.py — BM25 keyword-based retrieval using rank_bm25.

BM25 (Best Match 25) is a classical probabilistic term-frequency model that
complements dense retrieval by capturing exact lexical matches (section numbers,
statute citations, etc.) that embeddings sometimes miss.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer.

    Lowercases input, removes non-alphanumeric characters (keeps digits and
    hyphens common in IPC sections), and splits on whitespace.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    return text.split()


class BM25Retriever:
    """
    BM25-based keyword retriever over a corpus of text chunks.

    Attributes:
        chunks : List of chunk dicts containing at least a 'text' field.
        _bm25  : rank_bm25.BM25Okapi instance.
    """

    def __init__(self, chunks: List[Dict]) -> None:
        """
        Build a BM25 index over the provided chunks.

        Args:
            chunks: List of chunk dicts, each with a 'text' key.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            raise ImportError(
                "rank-bm25 is required. Install with: pip install rank-bm25"
            )

        self.chunks = chunks
        tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built over {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVAL
               ) -> List[Tuple[Dict, float]]:
        """
        Retrieve the top-k chunks most relevant to the query by BM25 score.

        Args:
            query: The query string.
            top_k: Number of results to return.

        Returns:
            List of (chunk_dict, score) tuples sorted by descending BM25 score.
            Scores are raw BM25 values (not normalised).
        """
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)                 # ndarray
        top_indices = scores.argsort()[::-1][:top_k]           # descending

        results: List[Tuple[Dict, float]] = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    @classmethod
    def from_chunks_file(cls, path: Path = config.PROCESSED_CHUNKS) -> "BM25Retriever":
        """
        Convenience factory: load chunks from disk and build BM25 index.

        Args:
            path: Path to the processed_chunks.json file.

        Returns:
            Initialised BM25Retriever.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {path}\n"
                "Run the ingestion pipeline first."
            )
        with open(path, encoding="utf-8") as fh:
            chunks = json.load(fh)
        return cls(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# CLI smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=config.LOG_LEVEL)
    retriever = BM25Retriever.from_chunks_file()
    query = "IPC section 302 punishment for murder"
    results = retriever.search(query, top_k=5)
    print(f"\nTop BM25 results for: '{query}'\n" + "─" * 60)
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"[{rank}] score={score:.4f}  source={chunk['source']}")
        print(f"      {chunk['text'][:200]}\n")
