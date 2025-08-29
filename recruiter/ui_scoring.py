from pathlib import Path
from typing import List, Dict

import streamlit as st

from recruiter.db import init_db, candidate_exists, save_candidate
from recruiter.vector import get_embeddings
from recruiter.utils import (
    save_uploaded_files,
    read_pdf_text,
    _hash_text,
    embed_text_average,
    cosine_similarity,
    sim_to_score,
    extract_contact_info,
)


def render_scoring_tab(api_key: str, threshold: float):
    st.subheader("Step 1: Upload Job Description (JD)")
    jd_file = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd_uploader")

    st.subheader("Step 2: Upload Candidate Resumes")
    resumes = st.file_uploader(
        "Upload resume PDFs", type=["pdf"], accept_multiple_files=True, key="resumes_uploader"
    )

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
            cur.execute(
                "SELECT name, email, phone, score, file_path, created_at FROM candidates WHERE jd_hash=? ORDER BY score DESC, created_at DESC",
                (jd_hash,),
            )
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

