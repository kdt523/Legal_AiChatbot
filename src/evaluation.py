"""
evaluation.py — Offline evaluation pipeline for the Legal RAG system.

Loads questions from data/test_questions.csv, runs each through the RAG
pipeline, allows manual scoring, and reports aggregate metrics.

Expected CSV columns:
    question   — the legal question (required)
    expected   — reference answer or key terms (optional, for BLEU/ROUGE)
    label      — pre-scored label 1/0 if batch scoring mode is used (optional)

Output:
    • data/eval_results.csv   — per-question results
    • Console summary table   — retrieval rate, grounded rate, scores
"""

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

EVAL_RESULTS_PATH = config.DATA_DIR / "eval_results.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_questions(csv_path: Path) -> List[Dict]:
    """Read test questions from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Test questions file not found: {csv_path}\n"
            "Create data/test_questions.csv with a 'question' column."
        )
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if r.get("question", "").strip()]
    logger.info(f"Loaded {len(rows)} test questions from {csv_path.name}.")
    return rows


def _is_grounded(answer: str) -> bool:
    """
    Heuristic: an answer is considered 'grounded' if it does NOT contain
    common hallucination signals and is not an error/not-found message.
    """
    lower = answer.lower()
    not_found_signals = [
        "not found in legal database",
        "unable to generate",
        "no relevant",
        "no content returned",
    ]
    return not any(sig in lower for sig in not_found_signals)


def _compute_keyword_overlap(answer: str, expected: str) -> float:
    """
    Simple keyword overlap score (Jaccard) between answer and expected text.
    Returns a value in [0, 1].
    """
    if not expected.strip():
        return 0.0
    answer_tokens   = set(answer.lower().split())
    expected_tokens = set(expected.lower().split())
    if not expected_tokens:
        return 0.0
    intersection = answer_tokens & expected_tokens
    union        = answer_tokens | expected_tokens
    return len(intersection) / len(union)


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(csv_path: Path = config.TEST_QUESTIONS_CSV,
                   interactive: bool = False,
                   limit: Optional[int] = None) -> List[Dict]:
    """
    Run the full evaluation pipeline.

    Args:
        csv_path:    Path to test_questions.csv.
        interactive: If True, prompt user to manually score each answer (1/0).
        limit:       Optionally cap the number of questions evaluated.

    Returns:
        List of per-question result dicts.
    """
    from pipeline import LegalRAGPipeline  # type: ignore

    questions = _load_questions(csv_path)
    if limit:
        questions = questions[:limit]

    pipeline  = LegalRAGPipeline()
    results: List[Dict] = []

    print(f"\n{'='*70}")
    print(f"  Legal RAG Evaluation — {len(questions)} question(s)")
    print(f"{'='*70}\n")

    for i, row in enumerate(questions, 1):
        question = row["question"].strip()
        expected = row.get("expected", "").strip()

        print(f"[{i}/{len(questions)}] {question}")

        t0 = time.time()
        try:
            result = pipeline.query(question)
            answer    = result["answer"]
            sources   = result["sources"]
            latency   = result["latency_s"]
            retrieval_ok = len(result["chunks"]) > 0
        except Exception as exc:
            logger.error(f"Pipeline error for Q{i}: {exc}")
            answer, sources, latency, retrieval_ok = str(exc), [], 0.0, False

        grounded    = _is_grounded(answer)
        kw_overlap  = _compute_keyword_overlap(answer, expected)

        # Manual scoring
        manual_score: Optional[int] = None
        if interactive:
            print(f"\n  ANSWER: {answer[:400]}{'...' if len(answer) > 400 else ''}")
            print(f"  SOURCES: {', '.join(sources)}")
            raw = input("  Score (1=correct, 0=wrong, s=skip): ").strip().lower()
            if raw == "1":
                manual_score = 1
            elif raw == "0":
                manual_score = 0
            print()

        record = {
            "q_index":       i,
            "question":      question,
            "expected":      expected,
            "answer":        answer,
            "sources":       "|".join(sources),
            "retrieval_ok":  int(retrieval_ok),
            "grounded":      int(grounded),
            "kw_overlap":    round(kw_overlap, 4),
            "manual_score":  manual_score if manual_score is not None else "",
            "latency_s":     latency,
        }
        results.append(record)

    # ── Save results ──────────────────────────────────────────────────────────
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Results saved → {EVAL_RESULTS_PATH}")

    _print_summary(results)
    return results


def _print_summary(results: List[Dict]) -> None:
    """Print a human-readable summary of evaluation metrics."""
    n = len(results)
    if n == 0:
        return

    retrieval_rate = sum(r["retrieval_ok"] for r in results) / n
    grounded_rate  = sum(r["grounded"]     for r in results) / n
    avg_overlap    = sum(r["kw_overlap"]   for r in results) / n
    avg_latency    = sum(r["latency_s"]    for r in results) / n

    scored = [r for r in results if r["manual_score"] != ""]
    manual_accuracy = (
        sum(int(r["manual_score"]) for r in scored) / len(scored)
        if scored else None
    )

    print("\n" + "═" * 70)
    print("  EVALUATION SUMMARY")
    print("═" * 70)
    print(f"  Total questions        : {n}")
    print(f"  Retrieval success rate : {retrieval_rate:.1%}  (chunks found)")
    print(f"  Grounded answer rate   : {grounded_rate:.1%}  (no 'not found' signal)")
    print(f"  Keyword overlap (avg)  : {avg_overlap:.4f}  (Jaccard, vs expected)")
    print(f"  Average latency        : {avg_latency:.2f}s")
    if manual_accuracy is not None:
        print(f"  Manual accuracy        : {manual_accuracy:.1%}  ({len(scored)} scored)")
    print("═" * 70)
    print(f"\n  Full results saved to: {EVAL_RESULTS_PATH}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=config.LOG_LEVEL)

    parser = argparse.ArgumentParser(
        description="Evaluate the Legal RAG pipeline."
    )
    parser.add_argument(
        "--csv",  default=str(config.TEST_QUESTIONS_CSV),
        help="Path to test_questions.csv"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Prompt for manual scoring after each answer."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of questions to evaluate."
    )
    args = parser.parse_args()

    run_evaluation(
        csv_path=Path(args.csv),
        interactive=args.interactive,
        limit=args.limit,
    )
