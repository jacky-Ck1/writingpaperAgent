"""
Configuration management with pluggable provider support.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Project configuration - all values overridable via environment variables."""

    # ---- Provider selection ----
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bge")
    VECTORSTORE_PROVIDER = os.getenv("VECTORSTORE_PROVIDER", "faiss")

    # ---- LLM: DeepSeek ----
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # ---- LLM: OpenAI ----
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

    # ---- LLM: Local ----
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b")

    # ---- Embedding: BGE (local) ----
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ---- Embedding: OpenAI ----
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # ---- Embedding: Local ----
    LOCAL_EMBEDDING_BASE_URL = os.getenv("LOCAL_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")

    # ---- Vector Store ----
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

    # ---- Document processing ----
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    SMART_CHUNKING = os.getenv("SMART_CHUNKING", "true").lower() == "true"

    # ---- Retrieval ----
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
    RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "mmr")
    ENABLE_HYBRID_RETRIEVAL = os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower() == "true"

    # ---- Paths ----
    DATA_DIR = os.getenv("DATA_DIR", "data")
    PDF_DIR = os.path.join(DATA_DIR, "pdf")
    INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")

    # ---- Workflow ----
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

    # ---- Evaluation ----
    EVALUATION_OUTPUT_DIR = os.getenv("EVALUATION_OUTPUT_DIR", "data/evaluation_reports")
    ENABLE_AUTO_EVALUATION = os.getenv("ENABLE_AUTO_EVALUATION", "false").lower() == "true"
