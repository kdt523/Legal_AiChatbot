"""
generator.py — Gemini LLM answer generation with grounded prompting.

Uses the official google-genai SDK (not the deprecated google-generativeai).
Sends the top-k reranked chunks as context to Google Gemini and returns a
grounded, citation-aware answer.  Falls back across model versions on quota
errors to maximise availability.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Fallback model chain
# ──────────────────────────────────────────────────────────────────────────────
_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def _get_client():
    """Create and return a configured google.genai Client."""
    try:
        from google import genai                    # type: ignore
        from google.genai import types              # type: ignore  # noqa: F401
    except ImportError:
        raise ImportError(
            "google-genai is required. "
            "Install with: pip install google-genai"
        )
    if not config.GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file:  GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTION = """You are a knowledgeable legal AI assistant specializing in Indian law.

CORE RULES:
1. Answer the user's question using ONLY the CONTEXT passages provided below.
2. The context contains legal provisions — answer by explaining the relevant law found in context.
3. If the user mentions a section number (e.g. "Section 302") but the context contains a provision titled "Punishment for murder", understand these refer to the same law and answer from that provision.
4. Be helpful: paraphrase and explain the law clearly in plain English.
5. Cite the provision title (e.g. "Punishment for murder") in your answer.
6. ONLY say "Not found in legal database." if the topic is completely absent from ALL provided context passages.
7. Do NOT hallucinate penalties, sections, or legal text not present in the context."""



def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Construct the grounded RAG prompt sent to Gemini.

    Args:
        query:  The user's legal question.
        chunks: Top-k reranked chunk dicts (each has 'text', 'source', 'chunk_id').

    Returns:
        Formatted prompt string.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        text   = chunk.get("text", "").strip()
        context_blocks.append(f"[SOURCE {i}: {source}]\n{text}")

    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"=== CONTEXT ===\n{context_str}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== ANSWER ===\n"
    )
    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# Robust generation with model fallback
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: List[Dict], retries: int = 2) -> str:
    """
    Generate an answer using Gemini with grounded context.

    Implements:
      • Model fallback (2.5-flash → 2.0-flash → 1.5-flash → 1.5-pro) on quota errors.
      • Per-model retry on transient failures.

    Args:
        query:   The user question.
        chunks:  Top-k reranked chunk dicts.
        retries: Number of per-model retry attempts on transient errors.

    Returns:
        Generated answer string.
    """
    if not chunks:
        return "No relevant legal passages were found for your query."

    from google.genai import types  # type: ignore

    client = _get_client()
    prompt = build_prompt(query, chunks)

    models_to_try = [config.GEMINI_MODEL] + [
        m for m in _FALLBACK_MODELS if m != config.GEMINI_MODEL
    ]

    for model_name in models_to_try:
        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=config.GEMINI_TEMPERATURE,
                        max_output_tokens=1024,
                    ),
                )

                text = _extract_text(response)
                if text.strip() and "No content returned" not in text:
                    logger.info(
                        f"Answer generated via {model_name} (attempt {attempt + 1})."
                    )
                    return text.strip()

            except Exception as exc:
                err = str(exc)
                if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                    logger.warning(
                        f"Quota exceeded on {model_name}. Switching to next model."
                    )
                    break
                if attempt < retries:
                    wait = 2 ** attempt
                    logger.warning(
                        f"Transient error on {model_name} (attempt {attempt + 1}): {err}. "
                        f"Retrying in {wait}s…"
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"Failed on {model_name}: {exc}")

    return "I was unable to generate an answer at this time. Please try again."


def _extract_text(response) -> str:
    """
    Safely extract text from a google.genai response object.

    Args:
        response: google.genai GenerateContentResponse.

    Returns:
        Text string, or a fallback message.
    """
    try:
        if hasattr(response, "text") and response.text:
            return response.text
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                return parts[0].text
    except Exception:
        pass
    return "(No content returned — possibly cold start or quota limit)"
