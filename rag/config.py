# Reusable settings
EMBED_MODEL = "LazarusNLP/all-indo-e5-small-v4"  # good English encoder 1 k-dim  :contentReference[oaicite:3]{index=3}
COLLECTION  = "my_pdfs"
DB_PATH     = "milvus_demo.db"          # Lite stores a single file here
CHUNK_SIZE  = 512
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.1)   # 10 %   :contentReference[oaicite:4]{index=4}
LLM_MODEL   = "google/gemma-2-2b-it"