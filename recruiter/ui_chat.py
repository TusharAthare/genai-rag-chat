from pathlib import Path
from typing import List, Tuple

import streamlit as st
from langchain.chains import ConversationalRetrievalChain

from recruiter.utils import save_uploaded_files
from recruiter.vector import build_or_load_vectorstore, make_llm, make_retriever, format_source


def render_chat_tab(api_key: str, model_name: str):
    st.subheader("Upload PDF(s) for Chat")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True, key="chat_uploader"
    )

    if uploaded_files:
        # Save files
        upload_dir = Path("uploads") / "chat"
        saved_paths, files_hash = save_uploaded_files(uploaded_files, upload_dir)

        with st.spinner("Indexing or loading vector store…"):
            db = build_or_load_vectorstore(tuple(map(str, saved_paths)), files_hash, api_key)
            retriever = make_retriever(db)
            llm = make_llm(model_name, api_key)
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

