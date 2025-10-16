import os
import logging
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# ---- LangChain (modern split packages) ----
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Chat/Embeddings providers
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# PDF loaders
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

# Optional: configure Google SDK (langchain_google_genai uses env var under the hood)
import google.generativeai as genai

# ----------------- Setup -----------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

st.set_page_config(page_title="RAG Chatbot (Budget PDF Ready)", page_icon="📊", layout="wide")
st.title("📊 RAG Chatbot — PDF Tables & Numbers")

# ----------------- Session State -----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "provider" not in st.session_state:
    st.session_state.provider = "OpenAI"
if "debug" not in st.session_state:
    st.session_state.debug = False

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("Model Provider", ["OpenAI", "Gemini"], index=0)
    st.session_state.provider = provider

    st.checkbox("Show Debug Info (sources, scores, chunks)", value=False, key="debug")

    st.divider()
    st.subheader("📄 Document")
    # Upload area at top
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # Stacked controls (one per row)
    if st.button("Process Document"):
        if not uploaded_file:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Parsing PDF and building vector store..."):
                try:
                    # Save temp file
                    tmp_path = "temp_upload.pdf"
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Prefer PyMuPDF (cleaner tables), fallback to PyPDF
                    try:
                        loader = PyMuPDFLoader(tmp_path)
                        docs = loader.load()
                        logger.info("Loaded with PyMuPDFLoader")
                    except Exception as e_mupdf:
                        logger.warning(f"PyMuPDFLoader failed ({e_mupdf}). Falling back to PyPDFLoader.")
                        loader = PyPDFLoader(tmp_path)
                        docs = loader.load()

                    if not docs:
                        st.error("No text extracted from PDF. Check the file content.")
                        st.stop()

                    # Table-friendly chunking
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=700,
                        chunk_overlap=120,
                        separators=["\n\n", "\n", " ", ""],
                    )
                    splits = splitter.split_documents(docs)

                    # Choose embeddings
                    if provider == "OpenAI":
                        openai_key = os.getenv("OPENAI_API_KEY")
                        if not openai_key:
                            st.error("OPENAI_API_KEY is missing in .env")
                            st.stop()
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                    else:
                        google_key = os.getenv("GOOGLE_API_KEY")
                        if not google_key:
                            st.error("GOOGLE_API_KEY is missing in .env")
                            st.stop()
                        genai.configure(api_key=google_key)
                        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

                    # Fresh Chroma (memory; set persist_directory if you want persistence)
                    st.session_state.vector_store = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        collection_name="pdf_chunks"
                    )
                    st.session_state.docs_loaded = True
                    st.success(f"Processed ✅  Pages: {len(docs)}  |  Chunks: {len(splits)}")
                except Exception as e:
                    st.session_state.vector_store = None
                    st.session_state.docs_loaded = False
                    st.error(f"Error while processing PDF: {e}")
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

    if st.button("Clear Vector Store"):
        st.session_state.vector_store = None
        st.session_state.docs_loaded = False
        st.success("Cleared vector store.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.success("Cleared chat history.")

    st.divider()
    st.caption("Tip: Ask precise, number-seeking questions. Example:\n"
               "• 'What is Kerala’s fiscal deficit (BE) for 2025–26?'\n"
               "• 'Allocations to Education, Health, and Agriculture in 2025–26 (BE)?'")

# ----------------- LLM selection -----------------
def get_llm():
    if st.session_state.provider == "OpenAI":
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is missing.")
            return None
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is missing.")
            return None
        # convert_system_message_to_human ensures consistent formatting
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, convert_system_message_to_human=True)

# ----------------- Numeric-aware Retriever Wrapper -----------------
NUMERIC_HINTS = ["₹", "crore", "%", "BE", "RE", "revenue", "expenditure", "deficit", "allocation", "budget", "grant", "tax"]

def numeric_score(text: str) -> int:
    """Heuristic: boost chunks dense with numbers and budget keywords."""
    digits = sum(ch.isdigit() for ch in text)
    hints = sum(h.lower() in text.lower() for h in NUMERIC_HINTS)
    return digits + (8 * hints)

def rerank_numeric_first(docs, top_k=8):
    scored = [(numeric_score(d.page_content), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]

def build_retriever():
    # base retriever with MMR for diversity
    base = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 40, "lambda_mult": 0.7}
    )

    def _get_docs(query: str):
        initial = base.get_relevant_documents(query)
        # numeric questions get a better shot with rerank
        return rerank_numeric_first(initial, top_k=8)

    return _get_docs

# ----------------- Prompt -----------------
SYSTEM_PROMPT = """You are a precise budget analyst. Answer ONLY using the provided PDF context.
Rules:
- Quote figures exactly as shown (include units like ₹ crore, %, BE/RE, and the year).
- If the table/section label appears (e.g., "Table A-4"), mention it.
- Include the PDF page number when possible (given with each chunk).
- If the exact figure is not present in the context, answer: "Not found in the uploaded document context."
- Be concise and factual. No speculation.
Question: {question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)

def format_docs_for_prompt(docs):
    """Attach page numbers so the model trusts and cites correctly."""
    out = []
    for d in docs:
        page = d.metadata.get("page", "NA")
        out.append(f"[page {page}] {d.page_content}")
    return "\n\n".join(out)

# ----------------- Main Chat -----------------
st.header("💬 Chat")

if not st.session_state.vector_store:
    st.info("Upload and process a PDF in the sidebar to begin.")
    st.stop()

llm = get_llm()
if llm is None:
    st.stop()

retriever_fn = build_retriever()

# Build RAG chain explicitly
rag_chain = (
    {
        "context": (lambda q: format_docs_for_prompt(retriever_fn(q))),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Chat input
user_q = st.text_input("Ask a question about your document:")
ask_col1, ask_col2 = st.columns([1, 1])
with ask_col1:
    run_btn = st.button("Ask")
with ask_col2:
    show_sources_toggle = st.checkbox("Show retrieved sources", value=True)

if run_btn and user_q:
    try:
        answer = rag_chain.invoke(user_q)
        st.session_state.chat_history.append((user_q, answer))

        # Display chat
        for q, a in st.session_state.chat_history[-10:]:  # last 10 entries
            st.write(f"👤 **You:** {q}")
            st.write(f"🤖 **Bot ({st.session_state.provider}):** {a}")
            st.write("---")

        # Sources/debug
        if show_sources_toggle or st.session_state.debug:
            with st.expander("🔎 Retrieved sources (top matches)"):
                docs = retriever_fn(user_q)
                for i, d in enumerate(docs, 1):
                    page = d.metadata.get("page", "NA")
                    st.markdown(f"**Source {i} — page {page}**")
                    if st.session_state.debug:
                        st.caption(f"(score heuristic: {numeric_score(d.page_content)})")
                    st.text(d.page_content[:1600])

    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("RAG run failed")
