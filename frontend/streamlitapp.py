import streamlit as st
import requests

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺")

# ---- Config ----
# Set this to your deployed FastAPI backend URL (Render, Railway, etc.)
# For local testing it's http://127.0.0.1:8000
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    st.caption("Point this at your deployed FastAPI backend's base URL.")
    if st.button("Clear chat"):
        st.session_state.messages = []

st.title("🩺 Medical RAG Assistant")
st.caption("Answers are based only on the document you indexed. It will say so if it doesn't know.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{backend_url.rstrip('/')}/ask",
                    json={"question": question},
                    timeout=60,
                )
                resp.raise_for_status()
                answer = resp.json().get("answer", "No answer returned.")
            except requests.exceptions.RequestException as e:
                answer = f"⚠️ Could not reach backend: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})