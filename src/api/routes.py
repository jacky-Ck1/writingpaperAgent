"""API route definitions for the Paper Writing Agent."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os

router = APIRouter(prefix="/api/v1", tags=["agent"])


# ---- Request / Response models ----

class QARequest(BaseModel):
    query: str = Field(..., description="Question about the paper")
    top_k: int = Field(3, ge=1, le=20)
    use_reranker: bool = Field(True)


class QAResponse(BaseModel):
    query: str
    answer: str
    sources: List[dict]
    retrieval_method: str
    model_used: str


class WritingRequest(BaseModel):
    topic: str = Field(..., description="Writing topic")
    max_iterations: int = Field(3, ge=1, le=10)
    style: str = Field("academic")


class WritingResponse(BaseModel):
    topic: str
    draft: str
    iterations: int
    approved: bool


class RAGWritingRequest(BaseModel):
    query: str = Field(..., description="Research question")
    max_iterations: int = Field(3, ge=1, le=10)


class RAGWritingResponse(BaseModel):
    query: str
    summary: str
    refined_summary: Optional[str]
    sources_count: int
    iterations: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    embedding_provider: str
    vectorstore_provider: str


# ---- Helpers ----

def _get_llm():
    from ..providers.llm import LLMProviderFactory
    return LLMProviderFactory.from_config().create()


def _get_hybrid_retriever():
    from ..providers.embedding import EmbeddingProviderFactory
    from ..providers.vectorstore import VectorStoreFactory
    from ..rag.vectorstore import load_index
    from ..rag.hybrid_retriever import HybridRetriever

    emb = EmbeddingProviderFactory.from_config().create()
    vs = VectorStoreFactory.from_config().load(emb)

    documents = []
    try:
        from ..rag.loader import load_pdf
        import glob
        from ..config import Config
        pdfs = glob.glob(os.path.join(Config.PDF_DIR, "*.pdf"))
        if pdfs:
            documents = load_pdf(pdfs[0])
    except Exception:
        pass

    return HybridRetriever(vs, emb, documents)


# ---- Routes ----

@router.get("/health", response_model=HealthResponse)
def health_check():
    from .. import __version__
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=os.getenv("LLM_PROVIDER", "deepseek"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "bge"),
        vectorstore_provider=os.getenv("VECTORSTORE_PROVIDER", "faiss"),
    )


@router.post("/qa", response_model=QAResponse)
def rag_qa(req: QARequest):
    """RAG Q&A with hybrid retrieval (BM25 + Dense + Reranker)."""
    try:
        llm = _get_llm()
        retriever = _get_hybrid_retriever()

        results = retriever.retrieve(
            req.query,
            top_k=req.top_k,
            use_reranker=req.use_reranker,
        )
        docs = [doc for doc, _ in results]

        context = "\n\n".join(
            f"[{d.metadata.get('section', '')}] {d.page_content}" for d in docs
        )

        prompt = f"""Answer based on the following paper excerpts:

{context}

Question: {req.query}"""

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return QAResponse(
            query=req.query,
            answer=answer,
            sources=[{"section": d.metadata.get("section", ""), "preview": d.page_content[:200]} for d in docs],
            retrieval_method="hybrid (BM25 + dense + reranker)" if req.use_reranker else "hybrid (BM25 + dense)",
            model_used=f"{os.getenv('LLM_PROVIDER','deepseek')}:{os.getenv('DEEPSEEK_MODEL', os.getenv('OPENAI_MODEL','unknown'))}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/write", response_model=WritingResponse)
def write_paper(req: WritingRequest):
    """Multi-agent writing: Planner -> Writer(s) -> Reviewer."""
    try:
        from ..workflows.writing_graph import create_writing_graph
        llm = _get_llm()
        graph = create_writing_graph(llm=llm, max_iterations=req.max_iterations)

        result = graph.invoke({
            "topic": req.topic,
            "outline": "",
            "draft": [],
            "feedback": "",
            "approved": False,
            "revision_type": None,
            "iteration": 0,
            "max_iterations": req.max_iterations,
        })

        draft_text = "\n\n".join(result.get("draft", []))
        return WritingResponse(
            topic=req.topic,
            draft=draft_text or result.get("outline", ""),
            iterations=result.get("iteration", 0),
            approved=result.get("approved", False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-write", response_model=RAGWritingResponse)
def rag_writing(req: RAGWritingRequest):
    """RAG writing: retrieve papers + generate academic summary."""
    try:
        from ..workflows.rag_writing_graph import create_rag_writing_graph
        llm = _get_llm()
        hybrid_retriever = _get_hybrid_retriever()
        graph = create_rag_writing_graph(
            llm=llm, max_iterations=req.max_iterations,
            hybrid_retriever=hybrid_retriever,
        )

        result = graph.invoke({
            "query": req.query,
            "retrieved_docs": [],
            "summary": "",
            "review_feedback": "",
            "refined_summary": "",
            "approved": False,
            "iteration": 0,
            "max_iterations": req.max_iterations,
        })

        return RAGWritingResponse(
            query=req.query,
            summary=result.get("summary", ""),
            refined_summary=result.get("refined_summary"),
            sources_count=len(result.get("retrieved_docs", [])),
            iterations=result.get("iteration", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
