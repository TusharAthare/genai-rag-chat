import io
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def _hash_bytes(data: bytes) -> str:
    """Compute a stable SHA-256 hex digest for raw bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _hash_text(text: str) -> str:
    """Hash a text string with SHA-256 using UTF-8 encoding."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _hash_files(file_bytes_list: List[bytes]) -> str:
    """Hash multiple files' bytes deterministically as a single digest."""
    h = hashlib.sha256()
    for b in file_bytes_list:
        h.update(b)
    return h.hexdigest()


def save_uploaded_files(files: List[io.BytesIO], base_dir: Path) -> Tuple[List[Path], str]:
    """Persist uploaded file-like objects to disk and return paths + hash."""
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
    """Extract plain text from a PDF file using `PyPDFLoader`."""
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    return "\n\n".join(d.page_content for d in docs)


def mean_vector(vectors: List[List[float]]) -> List[float]:
    """Compute the element-wise arithmetic mean of equal-length vectors."""
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
    """Embed text by chunking and averaging all chunk embeddings."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = [c.page_content if hasattr(c, "page_content") else c for c in splitter.create_documents([text])]
    vecs = embeddings.embed_documents(chunks)
    return mean_vector(vecs)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def sim_to_score(sim: float) -> float:
    """Map cosine similarity to a stricter 0–10 score.

    Tightened linear mapping: similarities below 0.50 score 0; 0.90 maps to 10.
    This raises the bar so 8/10 ≈ 0.82 cosine instead of ≈0.60.
    """
    min_sim, max_sim = 0.50, 0.90
    if sim <= min_sim:
        return 0.0
    score = (sim - min_sim) / (max_sim - min_sim) * 10.0
    return round(max(0.0, min(10.0, score)), 2)


def extract_contact_info(text: str, fallback_name: str) -> Dict[str, str]:
    """Extract basic contact info (name/email/phone) from resume text."""
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s()-]{7,}\d)", text)
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

