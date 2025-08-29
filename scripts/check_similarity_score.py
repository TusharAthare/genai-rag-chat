import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def read_pdf_text(path: Path) -> str:
    docs = PyPDFLoader(str(path)).load()
    return "\n\n".join(d.page_content for d in docs)


def mean_vector(vectors):
    if not vectors:
        return []
    dim = len(vectors[0])
    sums = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += x
    n = float(len(vectors))
    return [s / n for s in sums]


def embed_text_average(text: str, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = [c.page_content if hasattr(c, "page_content") else c for c in splitter.create_documents([text])]
    vecs = embeddings.embed_documents(chunks)
    return mean_vector(vecs)


def cosine_similarity(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def sim_to_score(sim: float) -> float:
    # Tightened mapping to align with app.py
    min_sim, max_sim = 0.50, 0.90
    if sim <= min_sim:
        return 0.0
    score = (sim - min_sim) / (max_sim - min_sim) * 10.0
    return round(max(0.0, min(10.0, score)), 2)


def main():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not set. Cannot compute embeddings.")
        return 2

    jd_path = Path("uploads/jd/Calfus JD - Python Backend.pdf")
    cv_path = Path("uploads/resumes/Pooja_Resume.pdf")
    if not jd_path.exists() or not cv_path.exists():
        print("Missing files:")
        print(jd_path, jd_path.exists())
        print(cv_path, cv_path.exists())
        return 3

    print("Loading PDFs and computing embeddings (this calls Google API)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=key)

    jd_text = read_pdf_text(jd_path)
    cv_text = read_pdf_text(cv_path)

    jd_vec = embed_text_average(jd_text, embeddings)
    cv_vec = embed_text_average(cv_text, embeddings)

    sim = cosine_similarity(jd_vec, cv_vec)
    score = sim_to_score(sim)

    print(f"Cosine similarity: {sim:.4f}")
    print(f"Score (tightened): {score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

