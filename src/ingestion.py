"""
ingestion.py — Load and clean raw legal text files.

Supports: .txt, .json (flat list of records with a 'text' field), .pdf (via PyMuPDF).
Cleaned files are saved to data/cleaned/ so that downstream steps are reproducible.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise whitespace, remove non-printable characters, and strip boilerplate
    headers/footers that are common in scanned legal PDFs.

    Args:
        text: Raw string extracted from a legal document.

    Returns:
        Cleaned string ready for chunking.
    """
    # Remove non-printable / control characters (keep newlines & tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces / tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Per-format loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    """Load a plain text file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


# ──────────────────────────────────────────────────────────────────────────────
# IPC chapter → starting section number mapping (official IPC 1860 numbering)
# Used to embed real section numbers in chunk text so BM25 can find "section 302"
# ──────────────────────────────────────────────────────────────────────────────
_IPC_CHAPTER_MAP_PATH = Path(__file__).parent.parent / "data" / "ipc_chapter_sections.json"

def _load_ipc_chapter_map() -> Dict:
    """Load chapter → starting section number mapping for IPC."""
    if _IPC_CHAPTER_MAP_PATH.exists():
        with open(_IPC_CHAPTER_MAP_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _load_json_sections(path: Path) -> List[Dict]:
    """
    Load a JSON file and return one document dict per section/record.

    For IPC (ipc.json), injects real section numbers using the chapter map
    so that queries like 'section 302' find the correct provision.

    Returns:
        List of {'source': filename, 'text': section_text} dicts.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        data = json.load(fh)

    source = path.name
    docs: List[Dict] = []

    # Load IPC chapter map only for ipc.json
    ipc_chapter_map = _load_ipc_chapter_map() if source.lower() == "ipc.json" else {}
    # Per-chapter counter: tracks how many sections we've seen in each chapter
    chapter_counters: Dict[str, int] = {}

    if isinstance(data, str):
        return [{"source": source, "text": data}]

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # IPC/CrPC-style: section_title + section_desc present
                if "section_title" in item and "section_desc" in item:
                    title   = str(item.get("section_title",   "")).strip()
                    desc    = str(item.get("section_desc",    "")).strip()
                    chapter = str(item.get("chapter_title",   "")).strip()
                    chapter_num = str(item.get("chapter", "")).strip()

                    # Compute real section number using chapter map (IPC only)
                    section_num = str(item.get("section", "")).strip()
                    if not section_num and ipc_chapter_map and chapter_num in ipc_chapter_map:
                        chapter_start = ipc_chapter_map[chapter_num]
                        chapter_counters[chapter_num] = chapter_counters.get(chapter_num, 0) + 1
                        section_num = str(chapter_start + chapter_counters[chapter_num] - 1)

                    parts_list = []
                    if chapter:
                        parts_list.append(f"Chapter {chapter_num}: {chapter}")
                    if section_num:
                        parts_list.append(f"Section {section_num}: {title}")
                    else:
                        parts_list.append(f"Provision: {title}")
                    parts_list.append(desc)
                    text = "\n".join(parts_list)
                else:
                    # Generic: try known field names
                    text = ""
                    for field in ("text", "description", "content", "body"):
                        if field in item and item[field]:
                            text = str(item[field]).strip()
                            break
                if text:
                    docs.append({"source": source, "text": text})
            elif isinstance(item, str) and item.strip():
                docs.append({"source": source, "text": item.strip()})
        return docs



    if isinstance(data, dict):
        return [{"source": source, "text": json.dumps(data, indent=2)}]

    return [{"source": source, "text": str(data)}]


def _load_pdf(path: Path) -> str:
    """Load text from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore
    except ImportError:
        raise ImportError(
            "PyMuPDF is required to load PDF files. "
            "Install with: pip install pymupdf"
        )

    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def load_document(path: Path) -> str:
    """
    Dispatch to the correct loader based on file extension.
    Returns raw text (for txt/pdf). For JSON use _load_json_sections directly.
    """
    ext = path.suffix.lower()
    loaders = {
        ".txt":  _load_txt,
        ".pdf":  _load_pdf,
    }
    if ext not in loaders:
        raise ValueError(f"Unsupported file type '{ext}' via load_document. Use _load_json_sections for JSON.")

    logger.info(f"Loading [{ext}] → {path.name}")
    return loaders[ext](path)


# ──────────────────────────────────────────────────────────────────────────────
# High-level ingestion API
# ──────────────────────────────────────────────────────────────────────────────

def ingest_all(raw_dir: Path = config.RAW_DIR,
               cleaned_dir: Path = config.CLEANED_DIR) -> List[Dict]:
    """
    Walk raw_dir, load every supported file, clean it, and return
    a flat list of document dicts ready for chunking.

    • JSON files  → one dict per section  (precise, section-level granularity)
    • TXT / PDF   → one dict per file

    Args:
        raw_dir:     Directory containing raw legal documents.
        cleaned_dir: Directory where cleaned text files will be saved.

    Returns:
        List of dicts: [{ 'source': filename, 'text': cleaned_text }, ...]
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    supported = {".txt", ".json", ".pdf"}
    files = sorted(p for p in raw_dir.rglob("*") if p.suffix.lower() in supported)

    if not files:
        logger.warning(
            f"No supported files found in {raw_dir}. "
            "Add .txt / .json / .pdf documents and re-run."
        )
        return []

    documents: List[Dict] = []
    for fp in files:
        try:
            ext = fp.suffix.lower()
            if ext == ".json":
                # Per-section ingestion — preserves section boundaries
                sections = _load_json_sections(fp)
                cleaned_sections = []
                for sec in sections:
                    sec["text"] = clean_text(sec["text"])
                    if sec["text"]:
                        cleaned_sections.append(sec)
                        documents.append(sec)

                # Save a single merged cleaned file for reference
                merged = "\n\n---\n\n".join(s["text"] for s in cleaned_sections)
                out_path = cleaned_dir / (fp.stem + ".txt")
                out_path.write_text(merged, encoding="utf-8")
                logger.info(f"  ✔ {fp.name} → {len(cleaned_sections)} sections → {out_path.name}")

            else:
                raw_text = load_document(fp)
                clean    = clean_text(raw_text)
                out_path = cleaned_dir / (fp.stem + ".txt")
                out_path.write_text(clean, encoding="utf-8")
                logger.info(f"  ✔ Saved cleaned file → {out_path.name}")
                documents.append({"source": fp.name, "text": clean})

        except Exception as exc:
            logger.error(f"  ✘ Failed to process {fp.name}: {exc}")
            continue

    logger.info(f"Ingestion complete. {len(documents)} document sections loaded from {len(files)} file(s).")
    return documents


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = ingest_all()
    print(f"\nTotal documents ingested: {len(docs)}")
