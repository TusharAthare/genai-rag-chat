# Chat with Your PDF — Gemini RAG (FAISS + LangChain + Streamlit)

A minimal **Retrieval-Augmented Generation (RAG)** app:
- Upload a PDF
- It creates **embeddings** of the document with **Gemini embeddings**
- Stores them in a local **FAISS** vector index
- Lets you **ask questions** in a chat-like UI powered by **Gemini**

---

## ✨ Features
- **PDF ingestion** (`pypdf`) with chunking for better recall
- **Embeddings** via `GoogleGenerativeAIEmbeddings`
- **Vector store** using `FAISS` (local, fast, no external dependency)
- **Retriever + LLM** pipeline using `LangChain` and **Gemini** (2.0-flash by default)
- **Streamlit UI** for simple, friendly interaction

---

## 🧱 Tech Stack
- Python, Streamlit
- LangChain, langchain-community, langchain-google-genai
- google-generativeai (Gemini API)
- FAISS (faiss-cpu)
- pypdf

---

## 📁 Project Structure
```
genai-rag/
├─ app.py                 # Streamlit web app
├─ rag_demo.py            # Minimal CLI demo script
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```

---

## 🚀 Quickstart (Windows)
**1) Create & activate venv**
```bash
python -m venv venv
venv\Scripts\activate
```

**2) Install dependencies**
```bash
pip install streamlit langchain langchain-community langchain-google-genai faiss-cpu pypdf google-generativeai
```

**3) Set your Gemini API key**

PowerShell:
```powershell
$env:GEMINI_API_KEY="your_actual_key_here"
echo $env:GEMINI_API_KEY
```

CMD (Command Prompt) — permanent:
```cmd
setx GEMINI_API_KEY "your_actual_key_here"
:: close & reopen CMD
echo %GEMINI_API_KEY%
```

**4) Run the app**
```bash
streamlit run app.py
```
The app opens at http://localhost:8501. Upload a PDF and start asking questions.

Note: You can also enter your Gemini API key directly in the app sidebar. It is stored only for your current browser session and is cleared when you close the tab.

---

## ⚙️ Configuration & Tuning
- `chunk_size=500`, `chunk_overlap=50`: good defaults; increase overlap if answers miss context.
- Retriever `k=3`: number of chunks fetched; try 3–5.
- Model: `gemini-2.0-flash` is fast & cost-effective; use `gemini-2.5-pro` for harder tasks.

---

## ☁️ Deployment

### Streamlit Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → pick your repo and `app.py`.
3. Set **Secrets**:
   ```toml
   GEMINI_API_KEY="your_api_key_here"
   ```
4. Deploy → you get a public URL.

---

## 🔒 Security & Privacy
- Do **not** commit your API key to GitHub; use **Secrets**.
- Uploaded PDFs are processed locally in the app instance.
- For production, add **rate limits**, **auth**, and a **retention policy**.

---

## 🛣️ Roadmap Ideas
- Support **multiple PDFs** and persist FAISS to disk
- Add **chat history** + better prompts
- Use **citations** (return the source chunk/page for each answer)
- Guardrails (banned topics, max tokens, safety filters)
- Dockerfile + CI/CD

---
