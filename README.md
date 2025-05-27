# Aristotle-bot

An interactive Retrieval-Augmented Generation (RAG) system over Aristotle’s *Nicomachean Ethics*, fine-tuned with LoRA adapters, served via a Streamlit UI (hostable on Hugging Face Spaces), with a full MLOps pipeline including CI/CD and feedback-driven retraining.

---

## Features

- **Corpus Preparation**  
  • PDF → raw text extraction & cleaning  
  • Split into Books & Sections  
  • Export as JSON with metadata
  
- **Retrieval**  
  • Embed sections with a QA-tuned bi-encoder (`multi-qa-MiniLM-L6-cos-v1`)  
  • Build & load FAISS index for k-NN retrieval.
  
- **Generation**  
  • LoRA-fine-tuned generator on *Nicomachean Ethics*  
  • Instruction-style prompts: “Context:… Question:… Answer:”  
  • Post-processing to return only the distilled answer (200-word cap)
  
- **Streamlit Frontend**  
  • Simple chat UI  
  • 👍/👎 feedback buttons  

  


