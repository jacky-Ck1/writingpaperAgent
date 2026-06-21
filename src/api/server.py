"""
FastAPI application entry point.

Start with:
    uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

Open http://localhost:8000/docs for interactive Swagger UI.
"""

import sys
import os

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router

app = FastAPI(
    title="Paper Writing Agent API",
    description="""
## AI-Powered Academic Paper Writing Agent

### Features
- **RAG Q&A**: Ask questions about indexed papers
- **Multi-Agent Writing**: Planner → Writer → Reviewer workflow
- **RAG Writing**: Auto-retrieve papers + generate academic summaries

### Architecture
Pluggable providers for LLM, Embedding, and Vector Store.
Hybrid retrieval with BM25 + Dense + Cross-encoder reranking.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Paper Writing Agent",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ---- Run standalone ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
