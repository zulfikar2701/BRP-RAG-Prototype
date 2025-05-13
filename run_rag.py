#!/usr/bin/env python
from rag import ingest, retrieve, generate
from pathlib import Path
import glob
from time import sleep

STAMP = Path(".ingest.done")

def pdfs_need_ingest(folder="data/pdfs/*.pdf") -> bool:
    if not STAMP.exists():
        return True
    newest_pdf = max(Path(p).stat().st_mtime for p in glob.glob(folder))
    return newest_pdf > STAMP.stat().st_mtime

def main():
    # 1) Ingest (run once, or whenever PDFs change)
    if pdfs_need_ingest():
        ingest.run()
        STAMP.touch()
        print("✅ Ingested new PDFs.")

    # 2) Ask something
    question = "When the Bank outsources supporting work (pekerjaan penunjang), what three criteria must that work satisfy?"
    hits = retrieve.fetch(question, top_k=4)
    print("Top chunks:", [h["entity"]["chunk"][:80] for h in hits])

    # 3) Generate answer
    print("\nAnswer:\n", generate.answer(question, hits), flush=True)
    sleep(1)                # give the stream time to flush


if __name__ == "__main__":
    main()
