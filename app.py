import os
import io
import re
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# ---------- App Config ----------
st.set_page_config(page_title="Recruiter Assistant + RAG (Gemini)", layout="wide")


# ---------- Environment ----------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Please set GEMINI_API_KEY as an environment variable before running.")
    st.stop()


# ---------- Helpers ----------
def _hash_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _hash_files(file_bytes_list: List[bytes]) -> str:
    h = hashlib.sha256()
    for b in file_bytes_list:
        h.update(b)
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def get_embeddings(key: str):
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)


def save_uploaded_files(files: List[io.BytesIO], base_dir: Path) -> Tuple[List[Path], str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    bytes_list: List[bytes] = []
    for uf in files:
        data = uf.getvalue() if hasattr(uf, "getvalue") else uf.read()
        name = getattr(uf, "name", None) or f"upload_{_hash_bytes(data)[:8]}.pdf"
        p = base_dir / Path(name).name
        with open(p, "wb") as f:
            f.write(data)
        paths.append(p)
        bytes_list.append(data)
    return paths, _hash_files(bytes_list)


def read_pdf_text(path: Path) -> str:
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)


def mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    n = float(len(vectors))
    return [s / n for s in sums]


def embed_text_average(text: str, embeddings) -> List[float]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = [c.page_content if hasattr(c, "page_content") else c for c in splitter.create_documents([text])]
    vecs = embeddings.embed_documents(chunks)
    return mean_vector(vecs)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def sim_to_score(sim: float) -> float:
    # Map cosine [-1,1] -> [0,10]
    score = (sim + 1.0) / 2.0 * 10.0
    return round(max(0.0, min(10.0, score)), 2)


def extract_contact_info(text: str, fallback_name: str) -> Dict[str, str]:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s()-]{7,}\d)", text)
    # naive name guess: first non-empty line that has letters and spaces and not email/phone
    name = fallback_name
    for line in text.splitlines():
        l = line.strip()
        if len(l) >= 2 and any(c.isalpha() for c in l) and (not re.search(r"@|\d", l)):
            name = l[:80]
            break
    return {
        "name": name,
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(1) if phone_match else "",
    }


# ---------- DB ----------
DB_PATH = Path("recruiting.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            score REAL,
            file_path TEXT,
            jd_hash TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_jd ON candidates(jd_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_emailjd ON candidates(email, jd_hash)")
    conn.commit()
    return conn


def candidate_exists(conn: sqlite3.Connection, email: str, jd_hash: str) -> bool:
    if not email:
        return False
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM candidates WHERE email=? AND jd_hash=? LIMIT 1", (email, jd_hash))
    return cur.fetchone() is not None


def save_candidate(conn: sqlite3.Connection, info: Dict[str, str], score: float, file_path: Path, jd_hash: str):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO candidates(name, email, phone, score, file_path, jd_hash, created_at) VALUES (?,?,?,?,?,?,?)",
        (
            info.get("name", ""),
            info.get("email", ""),
            info.get("phone", ""),
            float(score),
            str(file_path),
            jd_hash,
            datetime.utcnow().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()


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
    st.subheader("Step 1: Upload Job Description (JD)")
    jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd_uploader")

    st.subheader("Step 2: Upload Candidate Resumes")
    resumes = st.file_uploader("Upload resume PDFs", type=["pdf"], accept_multiple_files=True, key="resumes_uploader")

    if jd_file and resumes:
        if st.button("Score Candidates"):
            conn = init_db()
            embeddings = get_embeddings(api_key)

            # Save JD and resumes
            jd_dir = Path("uploads") / "jd"
            resume_dir = Path("uploads") / "resumes"
            (jd_path_list, _), (resume_paths, _) = (
                save_uploaded_files([jd_file], jd_dir),
                save_uploaded_files(resumes, resume_dir),
            )
            jd_path = jd_path_list[0]

            with st.spinner("Processing JD…"):
                jd_text = read_pdf_text(jd_path)
                jd_hash = _hash_text(jd_text)
                jd_vec = embed_text_average(jd_text, embeddings)

            st.write("\n")
            st.subheader("Step 3: Candidate Scores")
            progress = st.progress(0)
            results: List[Dict[str, str]] = []
            saved_count = 0

            for idx, rp in enumerate(resume_paths, start=1):
                with st.spinner(f"Scoring {Path(rp).name}…"):
                    r_text = read_pdf_text(rp)
                    r_vec = embed_text_average(r_text, embeddings)
                    sim = cosine_similarity(jd_vec, r_vec)
                    score = sim_to_score(sim)
                    info = extract_contact_info(r_text, Path(rp).stem)
                    saved = False
                    if score >= threshold and not candidate_exists(conn, info.get("email", ""), jd_hash):
                        save_candidate(conn, info, score, Path(rp), jd_hash)
                        saved = True
                        saved_count += 1

                    results.append(
                        {
                            "candidate": info.get("name", Path(rp).stem),
                            "email": info.get("email", ""),
                            "phone": info.get("phone", ""),
                            "score": score,
                            "saved": "Yes" if saved else "No",
                            "file": Path(rp).name,
                        }
                    )
                progress.progress(idx / max(1, len(resume_paths)))

            st.success(f"Scoring complete. Saved {saved_count} candidate(s) ≥ {threshold} to database.")
            st.table(results)

            st.subheader("Step 4: Saved Candidates (this JD)")
            cur = conn.cursor()
            cur.execute("SELECT name, email, phone, score, file_path, created_at FROM candidates WHERE jd_hash=? ORDER BY score DESC, created_at DESC", (jd_hash,))
            rows = cur.fetchall()
            if rows:
                st.table(
                    [
                        {
                            "candidate": r[0],
                            "email": r[1],
                            "phone": r[2],
                            "score": r[3],
                            "file": Path(r[4]).name,
                            "saved_at": r[5],
                        }
                        for r in rows
                    ]
                )
            else:
                st.info("No candidates saved for this JD yet.")
    else:
        st.info("Upload one JD and at least one resume to begin.")


def build_or_load_vectorstore(saved_paths: Tuple[str, ...], files_hash: str, key: str):
    # Keep previous chat functionality with FAISS persistence
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
    return db.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})


def make_llm(model_name: str, key: str):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=key)


def format_source(doc) -> str:
    src = doc.metadata.get("source", "")
    page = doc.metadata.get("page")
    page_str = f", p.{page + 1}" if isinstance(page, int) else ""
    return f"{Path(src).name}{page_str}"


with tab_chat:
    st.subheader("Upload PDF(s) for Chat")
    uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="chat_uploader")

    if uploaded_files:
        # Save files
        upload_dir = Path("uploads") / "chat"
        saved_paths, files_hash = save_uploaded_files(uploaded_files, upload_dir)

        with st.spinner("Indexing or loading vector store…"):
            db = build_or_load_vectorstore(tuple(map(str, saved_paths)), files_hash, api_key)
            retriever = make_retriever(db)
            llm = make_llm(model, api_key)
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.write("Ask a question about your PDFs")

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("Type your question…", key="chat_input")

        if user_msg:
            st.session_state.messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            chat_pairs: List[Tuple[str, str]] = []
            last_user: str | None = None
            for m in st.session_state.messages:
                if m["role"] == "user":
                    last_user = m["content"]
                elif m["role"] == "assistant" and last_user is not None:
                    chat_pairs.append((last_user, m["content"]))
                    last_user = None

            with st.spinner("Thinking…"):
                result = qa.invoke({"question": user_msg, "chat_history": chat_pairs})

            answer = result.get("answer", "")
            sources = result.get("source_documents", [])

            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    st.markdown("\n**Sources:**")
                    for i, doc in enumerate(sources, start=1):
                        st.markdown(f"- {i}. {format_source(doc)}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("Upload one or more PDF files to start chatting.")
