"""
Read every PDF in data/pdfs, split, embed, and push into Milvus
"""
from pathlib import Path
import numpy as np, torch, tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from .config import *

def run(source_dir=Path(__file__).parent.parent / "data" / "pdfs"):
    # 1) load
    docs = PyPDFDirectoryLoader(str(source_dir)).load()

    # 2) split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # 3) embed (GPU)
    encoder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    vecs = torch.tensor(encoder.encode([c.page_content for c in chunks]))
    vecs = np.array(vecs / torch.linalg.norm(vecs, dim=1, keepdim=True))  # L2-norm

    # 4) build dicts Milvus wants
    to_insert = [
        {"chunk": c.page_content, "source": c.metadata.get("page_number", ""), "vector": v.astype("float32")}
        for c, v in zip(chunks, vecs)
    ]

    # 5) push into Milvus Lite
    mc = MilvusClient(DB_PATH)
    mc.create_collection(COLLECTION, encoder.get_sentence_embedding_dimension(),
                         auto_id=True, overwrite=True, consistency_level="Eventually")
    mc.insert(COLLECTION, to_insert, progress_bar=True)
    print(f"✅ Ingested {len(to_insert)} chunks.")
