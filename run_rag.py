#!/usr/bin/env python
from rag import ingest, retrieve, generate

def main():
    # 1) Ingest (run once, or whenever PDFs change)
    ingest.run()

    # 2) Ask something
    question = "Summarise the key points of page 3."
    hits = retrieve.fetch(question, top_k=4)
    print("Top chunks:", [h["entity"]["chunk"][:80] for h in hits])

    # 3) Generate answer
    print("\nAnswer:\n", generate.answer(question, hits))

if __name__ == "__main__":
    main()
