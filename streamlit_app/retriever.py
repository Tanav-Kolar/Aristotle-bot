import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Config
INDEX_FILE    = "data/embeddings-index/nicomachean_ethics_structured.index"
METADATA_FILE = "data/embeddings-index/nicomachean_ethics_structured.metadata.json"
QA_MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"

# Load FAISS index & metadata at import time
_index    = faiss.read_index(INDEX_FILE)
_metadata = json.load(open(METADATA_FILE, "r", encoding="utf-8"))
_encoder  = SentenceTransformer(QA_MODEL_NAME)

def retrieve_top_k(question: str, k: int = 3):
    """
    Returns list of {book, section, text} for top-k relevant passages.
    """
    q_emb = _encoder.encode([f"question: {question}"], convert_to_numpy=True).astype("float32")
    D, I = _index.search(q_emb, k)
    return [ _metadata[i] for i in I[0] ]


if __name__ == "__main__":
    # Quick test
    q = "What is the highest good according to Aristotle?"
    top3 = retrieve_top_k(q, k=3)
    for r in top3:
        print(f"Book {r['book']} §{r['section']}: {r['text']}…\n")