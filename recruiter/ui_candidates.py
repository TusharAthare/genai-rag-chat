from pathlib import Path
from typing import Optional

import streamlit as st

from recruiter.db import init_db, fetch_candidates, delete_candidate


def render_candidates_tab():
    st.subheader("Saved Candidates")

    # Filters
    with st.expander("Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            jd_hash_filter: Optional[str] = st.text_input("JD Hash (optional)", key="cand_filter_jd")
        with col2:
            email_filter: Optional[str] = st.text_input("Email contains (optional)", key="cand_filter_email")

    conn = init_db()

    rows = fetch_candidates(conn, jd_hash=jd_hash_filter or None, email=email_filter or None)

    if not rows:
        st.info("No candidates found.")
        return

    st.caption(f"Found {len(rows)} candidate(s)")

    # Render each candidate with a small card and a delete button
    for r in rows:
        cid, name, email, phone, score, file_path, jd_hash, created_at = r
        with st.container():
            top_cols = st.columns([4, 2])
            with top_cols[0]:
                st.markdown(f"**{name or 'Unknown'}**")
                st.write(f"Email: {email or '-'}  |  Phone: {phone or '-'}")
                st.write(f"Score: {score:.2f}  |  JD: {jd_hash[:8] + '…' if jd_hash else '-'}  |  Saved: {created_at}")
                if file_path:
                    p = Path(file_path)
                    if p.is_file():
                        with open(p, "rb") as fh:
                            data = fh.read()
                        st.download_button(
                            label="Download resume (PDF)",
                            data=data,
                            file_name=p.name,
                            mime="application/pdf",
                            key=f"dl_{cid}",
                        )
                    else:
                        st.caption("Resume file not found on disk.")
            with top_cols[1]:
                if st.button("Delete", key=f"del_{cid}"):
                    deleted = delete_candidate(conn, cid)
                    if deleted:
                        st.success(f"Deleted candidate id={cid}")
                        st.rerun()
                    else:
                        st.error("Delete failed. Try again.")
