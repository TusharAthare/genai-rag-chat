import streamlit as st

from recruiter.config import get_api_key
from recruiter.ui_scoring import render_scoring_tab
from recruiter.ui_chat import render_chat_tab


# ---------- App Config ----------
st.set_page_config(page_title="Recruiter Assistant + RAG (Gemini)", layout="wide")


# ---------- Environment ----------
api_key = get_api_key()


# ---------- UI: Sidebar ----------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Save threshold (0–10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    model = st.selectbox(
        "Chat model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    st.caption("Embeddings: models/embedding-001 (Google)")


st.title("Recruiter Assistant")
tab_score, tab_chat = st.tabs(["Candidate Scoring", "Chat (PDF RAG)"])

with tab_score:
    render_scoring_tab(api_key=api_key, threshold=threshold)

with tab_chat:
    render_chat_tab(api_key=api_key, model_name=model)
