# app.py — VicRoads Pulse Chatbot (with custom logo header)
import os
import uuid
import streamlit as st
from typing import List
from ollama import Client
from search import get_context_passages

# ---- Ollama client ----
ollama_client = Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))

# ---- Defaults ----
DEFAULT_MODEL = "llama3"
DEFAULT_MODE = "dense"
DEFAULT_TOPK = 3
SHOW_CONTEXT = False
LOGO_PATH = "Logo.png"  # your Pulse logo file

# ---- RAG generation ----
def generate_answer(question: str, context: List[str], model: str = DEFAULT_MODEL) -> str:
    """Use only retrieved passages to answer."""
    combined_context = " ".join(context)
    prompt = (
        "You are interactive VicRoads Chatbot. "
        "Answer the question using ONLY the information below. "
        "Provide a concise answer in natural language without prefacing with phrases like 'According to...'. "
        "If the answer is not present, reply exactly: 'Sorry, I do not have information about this currently.'\n\n"
        f"Question: {question}\n\n"
        f"Context: {combined_context}\n\n"
        "Answer:"
    )
    try:
        res = ollama_client.generate(model=model, prompt=prompt)
        return getattr(res, "response", "").strip() or ""
    except Exception as e:
        return f"(Error calling model: {e})"

# ---- Thread helpers ----
def _new_thread_name(n: int) -> str:
    return f"Thread {n}"

def create_new_thread() -> str:
    """Create a new empty thread and return its id."""
    thread_id = uuid.uuid4().hex[:8]
    next_idx = len(st.session_state.threads) + 1
    st.session_state.threads[thread_id] = {
        "name": _new_thread_name(next_idx),
        "messages": [],
    }
    st.session_state.current_thread_id = thread_id
    return thread_id

def switch_thread(thread_id: str):
    st.session_state.current_thread_id = thread_id

def current_thread():
    return st.session_state.threads[st.session_state.current_thread_id]

# ---- Page config ----
st.set_page_config(page_title="Pulse - VicRoads Chatbot", page_icon=LOGO_PATH, layout="centered")

# ---- Header with logo ----
st.markdown(
    """
    <style>
        .centered-logo {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: -20px;
            margin-bottom: -15px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="centered-logo">', unsafe_allow_html=True)
st.image(LOGO_PATH, width=250)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Session bootstrap ----
if "threads" not in st.session_state:
    st.session_state.threads = {}
if "current_thread_id" not in st.session_state or st.session_state.current_thread_id not in st.session_state.threads:
    create_new_thread()

# =========================
# Sidebar: thread controls
# =========================
with st.sidebar:
    st.subheader("Conversations")
    if st.button("➕ New thread", use_container_width=True):
        create_new_thread()
        st.rerun()

    # Buttons to switch threads
    if st.session_state.threads:
        st.caption("Switch thread")
        for tid, data in st.session_state.threads.items():
            label = ("👉 " if tid == st.session_state.current_thread_id else "• ") + data["name"]
            if st.button(label, key=f"switch_{tid}", use_container_width=True):
                switch_thread(tid)
                st.rerun()

    st.markdown("---")
    if st.button("🧹 Clear current thread", use_container_width=True):
        st.session_state.threads[st.session_state.current_thread_id]["messages"] = []
        st.rerun()

    st.markdown("---")
    # VicRoads link button
    try:
        st.link_button("🌐 VicRoads", "https://www.vicroads.vic.gov.au/", use_container_width=True)
    except Exception:
        st.markdown(
            '<a href="https://www.vicroads.vic.gov.au/" target="_blank" rel="noopener">'
            '<button style="width:100%;padding:0.6rem;border-radius:0.5rem;">🌐 VicRoads</button>'
            "</a>",
            unsafe_allow_html=True,
        )

# ---- Show chat history ----
thread = current_thread()
for msg in thread["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- Input box ----
prompt = st.chat_input("Ask about VicRoads (type 'exit' to clear this thread')")
if prompt:
    if prompt.strip().lower() == "exit":
        thread["messages"] = []
        st.rerun()

    # Add user message
    thread["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Retrieve context with defaults
    try:
        context_passages = get_context_passages(prompt, mode=DEFAULT_MODE, top_k=DEFAULT_TOPK)
    except TypeError:
        context_passages = get_context_passages(prompt)

    # Optional: display retrieved context
    if SHOW_CONTEXT:
        with st.expander("Retrieved context"):
            for i, p in enumerate(context_passages, 1):
                st.markdown(f"**{i}.** {p}")

    # Generate answer
    answer = (
        generate_answer(prompt, context_passages, model=DEFAULT_MODEL)
        if context_passages
        else "Sorry, I do not have information about this currently."
    )

    # Add assistant message
    thread["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)