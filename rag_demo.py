import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set GEMINI_API_KEY as an environment variable")

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "Summarize the main points of this document."
print(qa.run(query))