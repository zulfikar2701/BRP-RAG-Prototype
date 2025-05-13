#!/usr/bin/env python

from pathlib import Path
import glob
from rag import ingest, retrieve, generate

STAMP = Path(".ingest.done")

def ingest_if_needed():
    """Run ingest only if a PDF is newer than the last ingest."""
    pdfs = glob.glob("data/pdfs/*.pdf")
    if not pdfs:
        raise FileNotFoundError("No PDFs found in data/pdfs/")
    if (not STAMP.exists()
            or max(Path(p).stat().st_mtime for p in pdfs) > STAMP.stat().st_mtime):
        print("⏳ Ingesting PDFs …")
        ingest.run()
        STAMP.touch()

def chat():
    ingest_if_needed()                      # runs once
    print("🔄 Warming up model (first run only)…")
    _ = generate.answer("hello", [])        # quick dummy call to load weights
    print("\n✅ Ready!  Ask about your PDFs (blank line = quit)\n")

    while True:
        try:
            question = input("› ").strip()
            if not question:
                print("Bye!")
                break

            hits = retrieve.fetch(question, top_k=4)
            answer = generate.answer(question, hits)
            print("\n" + answer + "\n")
        except KeyboardInterrupt:
            print("\nInterrupted. Bye!")
            break

if __name__ == "__main__":
    chat()
