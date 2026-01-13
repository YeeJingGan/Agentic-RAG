import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
BASE_URL = os.getenv("REQUEST_URL", "http://127.0.0.1:8000")

# -----------------------------
# Initialize session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False

# -----------------------------
# Streaming helper
# -----------------------------
def stream_response(url, payload=None):
    """
    Generator that yields streamed text from FastAPI.
    """
    try:
        if payload:
            resp = requests.post(url, json=payload, stream=True, timeout=300)
        else:
            resp = requests.get(url, stream=True, timeout=300)

        resp.raise_for_status()

        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                yield chunk.decode("utf-8", errors="ignore")
                time.sleep(0.01)

    except requests.RequestException as e:
        yield f"\nError: {str(e)}\n"

# -----------------------------
# App UI
# -----------------------------
st.title("Wikipedia Chatbot")
st.caption("Wikipedia-powered RAG with multi-agent reasoning")

# -----------------------------
# Greeting (once)
# -----------------------------
if not st.session_state.greeting_shown:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! 👋 I'm your Wikipedia chatbot. "
            "Ask me anything about Wikipedia articles."
        )
    st.session_state.greeting_shown = True

# -----------------------------
# Render chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input
# -----------------------------
if prompt := st.chat_input("Ask something about Wikipedia..."):
    # --- user message ---
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- assistant response ---
    full_response = ""

    with st.chat_message("assistant"):
        st.markdown("Thinking...")
        # Agent 1
        st.markdown("###### 🧠 Agent 1 — Query Rewrite")
        for chunk in stream_response(f"{BASE_URL}/agent1", {"query": prompt}):
            st.write(chunk)
            response = requests.get(f"{BASE_URL}/state").json()
            st.write(f"New Query: {response.get('updated_query', '')}")
        
        # Agent 2
        st.markdown("###### 🔍 Agent 2 — Knowledge Update")
        for chunk in stream_response(f"{BASE_URL}/agent2"):
            st.write(chunk)
            response = requests.get(f"{BASE_URL}/state").json()
            st.write(f"Knowledge Base Updated: {response.get('is_knowledge_updated', '')}")

        # Agent 3
        st.markdown("###### 📚 Agent 3 — Reasoning")
        for chunk in stream_response(f"{BASE_URL}/agent3"):
            st.write(chunk)
            response = requests.get(f"{BASE_URL}/state").json()
            st.write(f"Final k: {response.get('k', '')}")

        # Final answer
        st.markdown("###### ✅ Final Answer")
        for chunk in stream_response(f"{BASE_URL}/query"):
            full_response += chunk
            st.write(chunk)

    # --- save assistant message ---
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )