"""
Hybrid Retrieval: BM25 (sparse) + Dense (vector) + Cross-encoder Reranker.

Fusion strategy: Reciprocal Rank Fusion (RRF) - parameter-free, proven effective.
Reranker: cross-encoder model for final relevance scoring.

Reference:
    - RRF: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet..."
    - BGE-Reranker: BAAI/bge-reranker-base
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.retrievers import BM25Retriever


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================

def _reciprocal_rank_fusion(
    sparse_docs: List[Document],
    dense_results: List[Tuple[Document, float]],
    k: int = 60,
    top_n: int = 10,
) -> List[Tuple[Document, float]]:
    """Merge BM25 (rank-only) + dense (scored) results with RRF.

    RRF score = sum(1 / (k + rank_in_list)) across both lists.

    BM25 doesn't expose relevance scores, so it contributes only rank
    position. Dense results carry their original similarity scores on top
    of the RRF rank contribution.
    """
    doc_scores: dict = {}

    # BM25 side: rank only, no score
    for rank, doc in enumerate(sparse_docs, start=1):
        key = doc.page_content[:200]
        doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank)

    # Dense side: rank position
    for rank, (doc, _score) in enumerate(dense_results, start=1):
        key = doc.page_content[:200]
        doc_scores[key] = doc_scores.get(key, 0) + 1.0 / (k + rank)

    # Map keys back to Document objects
    all_docs: dict = {}
    for doc in sparse_docs:
        all_docs[doc.page_content[:200]] = doc
    for doc, _ in dense_results:
        all_docs[doc.page_content[:200]] = doc

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(all_docs[key], score) for key, score in ranked[:top_n]]


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

class Reranker:
    """Cross-encoder reranker using HuggingFace models.

    Default: BAAI/bge-reranker-base (strong, efficient).
    Alternative: cross-encoder/ms-marco-MiniLM-L-6-v2 (lighter).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None

    def _lazy_load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = 5,
    ) -> List[Tuple[Document, float]]:
        """Score documents against query using cross-encoder."""
        if not documents:
            return []

        model = self._lazy_load()
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


# ============================================================================
# Hybrid Retriever (orchestrator)
# ============================================================================

class HybridRetriever:
    """Orchestrates sparse + dense + reranker pipeline.

    Usage:
        retriever = HybridRetriever(vectorstore, embedding, documents)
        results = retriever.retrieve("What is CLIP?", top_k=5, use_reranker=True)
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        embedding: Embeddings,
        documents: List[Document],
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_device: str = "cpu",
        enable_reranker: bool = True,
    ):
        self._vectorstore = vectorstore
        self._embedding = embedding
        self._documents = documents
        # LangChain built-in BM25 via rank_bm25 library
        self._bm25 = BM25Retriever.from_documents(documents)
        self._reranker = Reranker(reranker_model, reranker_device) if enable_reranker else None
        self._enable_reranker = enable_reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
        use_reranker: bool = True,
    ) -> List[Tuple[Document, float]]:
        """Execute full hybrid retrieval pipeline.

        1. BM25 (sparse, LangChain built-in) + dense (vector) search
           Both retrieve top_k * 3 candidates for RRF merging.
        2. Fuse results with RRF
        3. Rerank with cross-encoder (optional)
        """
        fetch_k = top_k * 3

        # Sparse retrieval: sync k so BM25 and dense fetch the same number of candidates
        self._bm25.k = fetch_k
        sparse_docs: List[Document] = self._bm25.invoke(query)

        # Dense retrieval: FAISS returns (doc, L2 distance); invert to score
        dense_raw = self._vectorstore.similarity_search_with_score(query, k=fetch_k)
        dense_results = [(doc, 1.0 / (1.0 + dist)) for doc, dist in dense_raw]

        # RRF fusion (BM25 contributes rank only, dense contributes rank + score)
        fused = _reciprocal_rank_fusion(sparse_docs, dense_results, k=rrf_k, top_n=top_k * 2)

        # Rerank
        if self._reranker and use_reranker:
            docs_only = [doc for doc, _ in fused]
            return self._reranker.rerank(query, docs_only, top_n=top_k)

        return fused[:top_k]

    def as_langchain_retriever(self, top_k: int = 5):
        """Wrap as LangChain-compatible retriever."""
        from langchain_core.retrievers import BaseRetriever

        class _HybridLC(BaseRetriever):
            retriever: HybridRetriever
            k: int

            def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
                results = self.retriever.retrieve(query, top_k=self.k, **kwargs)
                return [doc for doc, _ in results]

        return _HybridLC(retriever=self, k=top_k)
