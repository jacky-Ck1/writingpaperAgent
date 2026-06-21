"""
RAG (Retrieval-Augmented Generation) module.

Components:
    loader           - PDF document loading
    splitter         - Section-aware smart chunking (delegates to smart_splitter)
    smart_splitter   - Academic paper section detection
    embedding        - Embedding model creation
    vectorstore      - FAISS / Chroma vector store management
    hybrid_retriever - BM25 (LangChain built-in) + Dense + Reranker pipeline
"""

from .loader import load_pdf
from .splitter import split_documents
from .smart_splitter import smart_split_documents
from .embedding import create_embeddings
from .vectorstore import build_faiss_index, load_index, save_index
from .hybrid_retriever import HybridRetriever, Reranker

__all__ = [
    "load_pdf",
    "split_documents",
    "smart_split_documents",
    "create_embeddings",
    "build_faiss_index",
    "load_index",
    "save_index",
    "HybridRetriever",
    "Reranker",
]
