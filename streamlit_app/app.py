import streamlit as st
from rag import rag_answer
import json, time

# Load or init feedback store
FB_PATH = "../feedback/feedback.json"
try:
    feedback = json.load(open(FB_PATH))
except FileNotFoundError:
    feedback = []

st.title("Aristotle 📜 RAG Chat")

q = st.text_input("Ask Aristotle…")
if st.button("Submit"):
    with st.spinner("Thinking…"):
        ans = rag_answer(q)
    st.markdown(f"**Answer:** {ans}")

    col1, col2 = st.columns(2)
    if col1.button("👍"):
        feedback.append({"q":q,"a":ans,"rating":1,"ts":time.time()})
    if col2.button("👎"):
        feedback.append({"q":q,"a":ans,"rating":0,"ts":time.time()})

    # persist feedback
    with open(FB_PATH,"w") as f:
        json.dump(feedback, f, indent=2)
