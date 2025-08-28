import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load API Key from environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ Please set GEMINI_API_KEY as an environment variable before running.")
    st.stop()

st.set_page_config(page_title="Chat with your PDF (Gemini)", layout="wide")
st.title("📄 Chat with your PDF (Gemini RAG)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save file temporarily
    temp_path = "uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Create embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # Build FAISS vector store
    db = FAISS.from_documents(docs, embeddings)

    # Retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

    # Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)

    # RAG pipeline
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Chat input
    st.subheader("Ask a question about your PDF")
    query = st.text_input("Your question:")

    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
        st.markdown(f"**Q:** {query}")
        st.markdown(f"**A:** {answer}")
