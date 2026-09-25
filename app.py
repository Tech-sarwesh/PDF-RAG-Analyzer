import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY is not configured. "
        "Please create a .env file and add your Gemini API key."
    )
    st.stop()


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Gemini PDF RAG Analyzer",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 3. TITLE
# ============================================================

st.title("🤖 Gemini PDF RAG Analyzer")

st.write(
    """
    Upload a PDF and ask questions about it using
    Retrieval-Augmented Generation (RAG) powered by Google Gemini.
    """
)


# ============================================================
# 4. SIDEBAR
# ============================================================

st.sidebar.header("⚙️ RAG Settings")


chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=300,
    max_value=2000,
    value=1000,
    step=100
)


chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=150,
    step=50
)


top_k = st.sidebar.slider(
    "Number of Retrieved Chunks",
    min_value=1,
    max_value=10,
    value=4
)


# ============================================================
# 5. GEMINI EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY
)


# ============================================================
# 6. GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)


# ============================================================
# 7. SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None


# ============================================================
# 8. PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ============================================================
# 9. BUILD KNOWLEDGE BASE
# ============================================================

if uploaded_file is not None:

    if st.button("🔨 Build Knowledge Base"):

        try:

            with st.spinner(
                "Reading PDF and creating embeddings..."
            ):

                # ------------------------------------------------
                # Save PDF temporarily
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    pdf_path = temp_file.name


                # ------------------------------------------------
                # Load PDF
                # ------------------------------------------------

                loader = PyPDFLoader(
                    pdf_path
                )

                documents = loader.load()


                # ------------------------------------------------
                # Split documents
                # ------------------------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                chunks = text_splitter.split_documents(
                    documents
                )


                # ------------------------------------------------
                # Create FAISS vector database
                # ------------------------------------------------

                vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings
                )


                # ------------------------------------------------
                # Save in Streamlit session
                # ------------------------------------------------

                st.session_state.vectorstore = vectorstore

                st.session_state.pdf_name = (
                    uploaded_file.name
                )


                # ------------------------------------------------
                # Delete temporary PDF
                # ------------------------------------------------

                os.remove(pdf_path)


            st.success(
                f"Knowledge base created successfully! "
                f"{len(chunks)} chunks were created."
            )


        except Exception as e:

            st.error(
                f"Error while processing PDF:\n\n{str(e)}"
            )


# ============================================================
# 10. CURRENT DOCUMENT
# ============================================================

if st.session_state.vectorstore is not None:

    st.info(
        f"📄 Current document: "
        f"**{st.session_state.pdf_name}**"
    )


# ============================================================
# 11. QUESTION
# ============================================================

question = st.text_input(
    "💬 Ask a question about your PDF:"
)


# ============================================================
# 12. ASK QUESTION
# ============================================================

if st.button("🔎 Ask Gemini"):

    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload a PDF and build the knowledge base first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching the document and asking Gemini..."
            ):

                # ------------------------------------------------
                # Create retriever
                # ------------------------------------------------

                retriever = (
                    st.session_state.vectorstore
                    .as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": top_k
                        }
                    )
                )


                # ------------------------------------------------
                # Retrieve relevant chunks
                # ------------------------------------------------

                retrieved_docs = retriever.invoke(
                    question
                )


                # ------------------------------------------------
                # Create context
                # ------------------------------------------------

                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )


                # ------------------------------------------------
                # RAG Prompt
                # ------------------------------------------------

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """
You are a document question-answering assistant.

Answer the user's question using ONLY the
information contained in the provided context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the context,
   say that the information was not found in the document.
4. Give a clear and useful answer.
5. Keep the answer focused on the user's question.

DOCUMENT CONTEXT:

{context}
"""
                        ),

                        (
                            "human",
                            "{question}"
                        )
                    ]
                )


                # ------------------------------------------------
                # Create final prompt
                # ------------------------------------------------

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )


                # ------------------------------------------------
                # Send request to Gemini
                # ------------------------------------------------

                response = llm.invoke(
                    final_prompt
                )


            # ====================================================
            # ANSWER
            # ====================================================

            st.subheader("🤖 Gemini Answer")

            st.write(
                response.content
            )


            # ====================================================
            # SOURCES
            # ====================================================

            st.subheader("📚 Retrieved Sources")


            for i, doc in enumerate(
                retrieved_docs,
                start=1
            ):

                page_number = (
                    doc.metadata.get(
                        "page",
                        "Unknown"
                    )
                )


                with st.expander(
                    f"Source {i} — Page {page_number}"
                ):

                    st.write(
                        doc.page_content
                    )


        except Exception as e:

            st.error(
                f"Error while generating answer:\n\n{str(e)}"
            )