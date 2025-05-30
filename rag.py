from transformers import AutoTokenizer, AutoModelForCausalLM
from model.retriever import retrieve_top_k

# Config
GEN_PATH = "checkpoints/accel_ethics/epoch_3"  # adjust to your final checkpoint

#Load generator
tokenizer = AutoTokenizer.from_pretrained(GEN_PATH)
generator = AutoModelForCausalLM.from_pretrained(GEN_PATH)
generator.eval()

def rag_answer(question: str, k: int = 5, max_len: int = 500):
    #Retrieve top-k chunks
    chunks = retrieve_top_k(question, k)
    context = "\n\n".join(chunk["text"] for chunk in chunks)

    #Build RAG prompt
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    #Generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs    = generator.generate(
        **inputs,
        max_length= inputs.input_ids.shape[-1] + max_len,
        no_repeat_ngram_size=3,
        num_beams=5,
        early_stopping=True
    )

    #Strip off the prompt + context, keep only the new tokens.
    generated = outputs[0]
    answer_tokens = generated[ inputs.input_ids.shape[-1]: ]  # slice off the prefix
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    return answer

if __name__ == "__main__":
    q = "What is the highest good according to Aristotle?"
    print("Q:", q)
    print("A:", rag_answer(q))