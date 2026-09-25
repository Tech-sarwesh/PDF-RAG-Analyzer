# 🤖 Gemini PDF RAG Analyzer

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its content.

Built using **Google Gemini, LangChain, FAISS, PyPDF, and Streamlit**.

---

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Text chunking
- 🧠 Gemini embeddings
- 🔎 FAISS similarity search
- 🤖 Gemini-powered answers
- 📚 Retrieved source pages
- ⚙️ Adjustable chunk size, overlap, and Top-K
- 🌐 Streamlit web interface

---

## 🏗️ Architecture

```text
PDF
 ↓
PyPDFLoader
 ↓
Text Chunking
 ↓
Gemini Embeddings
 ↓
FAISS Vector Store
 ↓
User Question
 ↓
Question Embedding
 ↓
FAISS Similarity Search
 ↓
Relevant Chunks
 ↓
Gemini LLM
 ↓
Final Answer + Sources
