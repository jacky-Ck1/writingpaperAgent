"""
Document splitter: delegates to smart section-aware splitter when enabled.
"""
from typing import List, Optional
from langchain_core.documents import Document
from .smart_splitter import split_documents as _smart_split
from ..config import Config


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split documents into chunks.

    When SMART_CHUNKING is enabled, uses section-aware splitting
    that detects academic paper structure (Abstract, Introduction, etc.).

    Args:
        documents: Document list from loader.
        chunk_size: Override default chunk size.
        chunk_overlap: Override default overlap.

    Returns:
        List of chunked Documents with enriched metadata.
    """
    chunk_size = chunk_size or Config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    return _smart_split(documents, chunk_size, chunk_overlap, smart=Config.SMART_CHUNKING)
