import json
import numpy as np
import faiss
from pathlib import Path

def build_faiss_index(
    json_path: str,
    index_path: str,
    embeddings_npy_path: str
):
    """
    1. Loads sections with embeddings from a JSON file.
    2. Extracts embeddings into a NumPy array and saves it.
    3. Builds a FAISS L2 index on the embeddings and saves the index.
    4. Exports simple metadata (book & section) for lookup.
    """
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        sections = json.load(f)

    # Extract embeddings
    embeddings = np.array([sec['embedding'] for sec in sections], dtype='float32')

    # Save embeddings as .npy
    np.save(embeddings_npy_path, embeddings)

    # Build FAISS index (L2 distance)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, index_path)

    # Save metadata mapping (id → book & section)
    metadata = [
        {"id": i, "book": sec["book"], "section": sec["section"]}
        for i, sec in enumerate(sections)
    ]
    meta_path = Path(index_path).with_suffix('.metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Embeddings .npy saved to: {embeddings_npy_path}")
    print(f"FAISS index saved to: {index_path}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    # Example usage
    build_faiss_index(
        json_path="data/embeddings-index/nicomachean_ethics_embeds.json",
        index_path="data/embeddings-index/nicomachean_ethics_structured.index",
        embeddings_npy_path="data/embeddings-index/nicomachean_ethics_embeddings.npy"
    )