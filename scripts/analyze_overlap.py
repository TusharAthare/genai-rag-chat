import re
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_text(p: Path) -> str:
    docs = PyPDFLoader(str(p)).load()
    return "\n\n".join(d.page_content for d in docs)


def top_terms(txt: str, n: int = 40):
    word_re = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]{1,}")
    stop = set(
        "the a an and or of to for in on with by as from that this is are was were be being been at into about over under it its our your you we they them he she his her not no etc etc.".split()
    )
    words = [w.lower() for w in word_re.findall(txt) if w.lower() not in stop and len(w) > 2]
    return Counter(words).most_common(n)


def main():
    jd_path = Path("uploads/jd/Calfus JD - Python Backend.pdf")
    res_path = Path("uploads/resumes/Pooja_Resume.pdf")

    if not jd_path.exists() or not res_path.exists():
        print("Missing one or both files.")
        print(jd_path, jd_path.exists())
        print(res_path, res_path.exists())
        return

    jd = load_text(jd_path)
    res = load_text(res_path)

    jd_top = top_terms(jd, 50)
    res_top = top_terms(res, 50)

    jd_vocab = {w for w, _ in jd_top}
    res_vocab = {w for w, _ in res_top}
    common = sorted(jd_vocab & res_vocab)

    print("JD top terms:")
    print(jd_top[:25])
    print("\nResume top terms:")
    print(res_top[:25])
    print("\nCommon top terms (subset):")
    print(common[:40])

    print("\nJD length:", len(jd), "Resume length:", len(res))


if __name__ == "__main__":
    main()
