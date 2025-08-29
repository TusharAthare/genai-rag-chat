from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


@st.cache_resource(show_spinner=False)
def get_embeddings(key: str):
    """Create (and cache) Google Generative AI embeddings client."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)


def build_or_load_vectorstore(saved_paths: tuple[str, ...], files_hash: str, key: str):
    """Build or load a FAISS vector store for the uploaded PDFs."""
    persist_root = Path(".faiss")
    persist_dir = persist_root / files_hash
    persist_root.mkdir(exist_ok=True)
    embeddings = get_embeddings(key)

    if persist_dir.exists():
        db = FAISS.load_local(str(persist_dir), embeddings, allow_dangerous_deserialization=True)
        return db

    documents = []
    for p in saved_paths:
        loader = PyPDFLoader(str(p))
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(persist_dir))
    return db


def make_retriever(db: FAISS):
    """Create an MMR-based retriever from a FAISS store."""
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})


def make_llm(model_name: str, key: str):
    """Construct a Chat Google Generative AI LLM client."""
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=key)


def format_source(doc) -> str:
    """Format a retrieved document's source metadata for display."""
    src = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    page_str = f", p.{page + 1}" if isinstance(page, int) else ""
    return f"{Path(src).name}{page_str}"

