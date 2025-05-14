"""
Light wrapper so the UI code stays tidy.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from rag import retrieve, generate

def ask_rag(question: str, top_k: int = 4) -> str:
    """
    Retrieve chunks & generate an answer.
    Returns assistant reply (str).
    """
    hits = retrieve.fetch(question, top_k=top_k)
    return generate.answer(question, hits)
