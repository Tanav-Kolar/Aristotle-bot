import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Config
SECTIONS_JSON      = "data/processed/nicomachean_ethics_sections.json"
EMB_NPY            = "data/embeddings-index/nicomachean_ethics_embeddings.npy"
INDEX_FILE         = "data/embeddings-index/nicomachean_ethics_structured.index"
METADATA_JSON      = "data/embeddings-index/nicomachean_ethics_structured.metadata.json"
QA_MODEL_NAME      = "multi-qa-MiniLM-L6-cos-v1"

def main():
    #Load sections
    sections = json.load(open(SECTIONS_JSON, "r", encoding="utf-8"))

    #Compute QA-tuned embeddings
    encoder = SentenceTransformer(QA_MODEL_NAME)
    texts   = [sec["text"] for sec in sections]
    embs    = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    np.save(EMB_NPY, embs)
    print(f"Saved embeddings ({embs.shape}) → {EMB_NPY}")

    #Build FAISS index (L2)
    dim   = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, INDEX_FILE)
    print(f"Saved FAISS index → {INDEX_FILE}")

    #Dump metadata
    meta = [
        {"id": i, "book": sec["book"], "section": sec["section"], "text": sec["text"][:200]}
        for i, sec in enumerate(sections)
    ]
    json.dump(meta, open(METADATA_JSON, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved metadata ({len(meta)} entries) → {METADATA_JSON}")

if __name__ == "__main__":
    main()