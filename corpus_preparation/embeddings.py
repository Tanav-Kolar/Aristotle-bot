import json
from sentence_transformers import SentenceTransformer

def embed_sections_from_json(
    input_path: str,
    output_path: str,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
):
    """
    Loads a JSON file of sections (with a "text" field),
    generates embeddings with Sentence-Transformers,
    and writes out a new JSON with an added "embedding" field.
    """
    #Load sections
    with open(input_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    #Load embedding model
    model = SentenceTransformer(model_name)

    #Compute embeddings in batches
    texts = [sec["text"] for sec in sections]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    #Attach embeddings
    for sec, emb in zip(sections, embeddings):
        sec["embedding"] = emb.tolist()

    #Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    embed_sections_from_json(
        input_path="data/processed/nicomachean_ethics_sections.json",
        output_path="data/embeddings-index/nicomachean_ethics_embeds.json")