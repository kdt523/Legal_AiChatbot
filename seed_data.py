"""
seed_data.py — Copy and prepare legal JSON datasets from AskLegal source repository.

This is a one-time setup script that copies the IPC, CrPC, CPC, IEA, HMA, MVA, NIA,
and IDA JSON files from the AskLegal source repo into data/raw/ so the ingestion
pipeline can process them.

Run from the legal-rag-final/ directory:
    python seed_data.py
"""

import shutil
import sys
from pathlib import Path

# ── Resolve project root ───────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
RAW_DIR    = ROOT / "data" / "raw"
SOURCE_DIR = ROOT.parent / "AskLegal.ai-AI-Legal-Assistant" / "laws_json"

DATASETS = [
    "ipc.json",
    "crpc.json",
    "cpc.json",
    "iea.json",
    "hma.json",
    "MVA.json",
    "nia.json",
    "ida.json",
]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not SOURCE_DIR.exists():
        print(f"[WARN] Source directory not found: {SOURCE_DIR}")
        print("       Place your .json / .txt / .pdf legal files in data/raw/ manually.")
        return

    copied = 0
    for fname in DATASETS:
        src = SOURCE_DIR / fname
        dst = RAW_DIR / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  ✔ Copied {fname}")
            copied += 1
        else:
            print(f"  ✘ Not found: {src}")

    print(f"\nDone. {copied}/{len(DATASETS)} dataset(s) copied to {RAW_DIR}")
    print("Next step: run  python src/ingestion.py")


if __name__ == "__main__":
    main()
