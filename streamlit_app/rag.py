from transformers import AutoTokenizer, AutoModelForCausalLM
from retriever import retrieve_top_k

# Config
CKPT = "checkpoints/accel_ethics/epoch_3"  # adjust to your final checkpoint

#Load generator
tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT)
model.eval()

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
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        truncation=True, 
        max_length=1024
    )
    prefix_len = inputs.input_ids.shape[-1]

    #Generate
    outputs   = model.generate(
        **inputs,
        max_new_tokens=200,
        no_repeat_ngram_size=3,
        num_beams=5,
        early_stopping=True
    )
    gen_tokens = outputs[0][prefix_len:]
    answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    words = answer.split()
    word_limit = 200
    if len(words) > word_limit:
        answer = " ".join(words[:word_limit]) + "..."

    return answer