"""
Paper Writing Agent v2.1 - Pluggable LLM Agent Framework for Academic Writing.

Architecture:
    providers/    - Pluggable LLM / Embedding / VectorStore backends
    rag/          - Hybrid retrieval (BM25 + Dense + Reranker) + smart chunking
    agents/       - RAGAgent for paper Q&A
    workflows/    - LangGraph multi-agent writing pipelines
    evaluation/   - Multi-dimensional quality evaluation
    api/          - FastAPI production server
    config.py     - Unified configuration (env-driven)

Quick start:
    python main.py build --pdf data/pdf/paper.pdf
    python main.py qa --query "What is CLIP?"
    python main.py write --topic "Deep Learning in NLP"
    uvicorn src.api.server:app --reload
"""

__version__ = "2.1.0"
