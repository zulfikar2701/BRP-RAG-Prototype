from pymilvus import MilvusClient
import torch, numpy as np
from sentence_transformers import SentenceTransformer
from .config import *

encoder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

def fetch(query, top_k=4):
    mc = MilvusClient(DB_PATH)
    qvec = encoder.encode([query])
    qvec = torch.tensor(qvec)
    qvec = torch.nn.functional.normalize(qvec, p=2, dim=1)
    qvec = list(map(np.float32, qvec))

    hits = mc.search(COLLECTION, data=qvec,
                     limit=top_k,
                     output_fields=["chunk", "source"])
    # flatten result list
    return [h for batch in hits for h in batch]
