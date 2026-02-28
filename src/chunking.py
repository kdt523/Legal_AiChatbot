"""
chunking.py — Split cleaned legal documents into overlapping text chunks.

Uses a character-based recursive splitter that respects paragraph and sentence
boundaries.  Chunk size and overlap are configured in config.py.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# Approximate characters per token for English text
_CHARS_PER_TOKEN = 4


def _split_text(text: str, chunk_chars: int, overlap_chars: int) -> List[str]:
    """
    Recursively split *text* into chunks of at most *chunk_chars* characters,
    with *overlap_chars* character overlap, preferring splits on double-newlines,
    then single newlines, then spaces.

    Args:
        text:          Input text to split.
        chunk_chars:   Maximum chunk size in characters.
        overlap_chars: Overlap size in characters.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_chars:
        return [text]

    separators = ["\n\n", "\n", ". ", " ", ""]
    for sep in separators:
        if sep == "":
            # Hard split as last resort
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_chars, len(text))
                chunks.append(text[start:end])
                start = end - overlap_chars if (end - overlap_chars) > start else end
            return chunks

        splits = text.split(sep)
        if len(splits) == 1:
            continue  # separator not found, try next

        chunks: List[str] = []
        current = ""
        for part in splits:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from end of previous
                overlap_start = max(0, len(current) - overlap_chars)
                current = current[overlap_start:] + (sep if current[overlap_start:] else "") + part

        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]

    return [text]


def chunk_document(doc: Dict, chunk_size: int = config.CHUNK_SIZE,
                   chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Dict]:
    """
    Chunk a single document dict into a list of chunk dicts.

    Each chunk dict contains:
      - chunk_id   : unique identifier "{source}::chunk_{n}"
      - source     : original filename
      - chunk_index: 0-based position within the document
      - text       : chunk text

    Args:
        doc:          Dict with keys 'source' and 'text'.
        chunk_size:   Target chunk size in tokens (converted to characters).
        chunk_overlap: Overlap in tokens between consecutive chunks.

    Returns:
        List of chunk dicts.
    """
    chunk_chars   = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * _CHARS_PER_TOKEN

    text   = doc["text"]
    source = doc.get("source", "unknown")

    raw_chunks = _split_text(text, chunk_chars, overlap_chars)

    chunks: List[Dict] = []
    for idx, raw in enumerate(raw_chunks):
        stripped = raw.strip()
        if not stripped:
            continue
        chunks.append({
            "chunk_id":    f"{source}::chunk_{idx}",
            "source":      source,
            "chunk_index": idx,
            "text":        stripped,
        })

    logger.debug(f"  {source}: {len(chunks)} chunks created.")
    return chunks


def chunk_documents(documents: List[Dict],
                    chunk_size: int = config.CHUNK_SIZE,
                    chunk_overlap: int = config.CHUNK_OVERLAP) -> List[Dict]:
    """
    Chunk a list of document dicts and return the combined list of chunk dicts.

    chunk_id is assigned as a globally unique "{source}::g{N}" to prevent
    collisions when multiple short section-level documents from the same source
    each produce a single chunk (which would all get chunk_index=0 otherwise).

    Also persists the result to config.PROCESSED_CHUNKS (JSON).

    Args:
        documents:     Output of ingestion.ingest_all().
        chunk_size:    Token chunk size.
        chunk_overlap: Token overlap between chunks.

    Returns:
        Flat list of all chunk dicts.
    """
    all_chunks: List[Dict] = []
    global_counter = 0

    for doc in documents:
        for chunk in chunk_document(doc, chunk_size, chunk_overlap):
            # Overwrite chunk_id with globally unique ID
            chunk["chunk_id"] = f"{chunk['source']}::g{global_counter}"
            global_counter += 1
            all_chunks.append(chunk)

    logger.info(f"Chunking complete. Total chunks: {len(all_chunks)}")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PROCESSED_CHUNKS, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, ensure_ascii=False, indent=2)
    logger.info(f"Chunks saved → {config.PROCESSED_CHUNKS}")

    return all_chunks



def load_chunks(path: Path = config.PROCESSED_CHUNKS) -> List[Dict]:
    """
    Load pre-computed chunks from disk.

    Args:
        path: Path to the JSON file produced by chunk_documents().

    Returns:
        List of chunk dicts.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run the ingestion pipeline first: python src/ingestion.py"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ingestion import ingest_all
    docs   = ingest_all()
    chunks = chunk_documents(docs)
    print(f"Total chunks produced: {len(chunks)}")
