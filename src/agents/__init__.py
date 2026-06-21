"""
Agent module: intelligent agents for RAG and multi-agent collaboration.

Exports:
    RAGAgent - Retrieval-augmented agent for paper Q&A
"""

from .base_agent import BaseAgent
from .rag_agent import RAGAgent

__all__ = ["BaseAgent", "RAGAgent"]
