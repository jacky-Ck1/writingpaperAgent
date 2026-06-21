# Project Architecture v2.0

## Overall Architecture

The project follows a **pluggable layered architecture** with dependency inversion at every boundary.

```
┌─────────────────────────────────────────────────┐
│                  API Layer (FastAPI)              │
├─────────────────────────────────────────────────┤
│            Core Layer (UnifiedAgent, Router)      │
├──────────┬──────────┬────────────┬───────────────┤
│ Agents   │Workflows │Evaluation  │ Tools         │
├──────────┴──────────┴────────────┴───────────────┤
│              RAG Layer (retrieval pipeline)        │
├─────────────────────────────────────────────────┤
│         Provider Abstraction Layer ★              │
│  ┌──────────┐ ┌───────────┐ ┌───────────────┐   │
│  │   LLM    │ │ Embedding │ │ Vector Store  │   │
│  │ Provider │ │ Provider  │ │   Provider    │   │
│  └──────────┘ └───────────┘ └───────────────┘   │
└─────────────────────────────────────────────────┘
```

## 1. Provider Abstraction Layer (`src/providers/`)

**Design principle**: Dependency inversion. All upper layers depend on abstract
interfaces (`BaseLLMProvider`, `BaseEmbeddingProvider`, `BaseVectorStoreProvider`),
never on concrete implementations.

### LLM Provider

| Provider | Backend | Config |
|---|---|---|
| `DeepSeekProvider` | DeepSeek API (OpenAI-compatible) | `LLM_PROVIDER=deepseek` |
| `OpenAIProvider` | OpenAI GPT-4o / GPT-3.5 | `LLM_PROVIDER=openai` |
| `LocalProvider` | Ollama / vLLM / any OpenAI-compatible server | `LLM_PROVIDER=local` |

Usage:
```python
from src.providers.llm import LLMProviderFactory

provider = LLMProviderFactory.from_config()  # reads env
llm = provider.create(temperature=0.7)
```

### Embedding Provider

| Provider | Backend | Config |
|---|---|---|
| `BGEProvider` | HuggingFace BGE (local) | `EMBEDDING_PROVIDER=bge` |
| `OpenAIEmbeddingProvider` | text-embedding-3 (API) | `EMBEDDING_PROVIDER=openai` |
| `LocalEmbeddingProvider` | Ollama / TEI (local server) | `EMBEDDING_PROVIDER=local` |

### Vector Store Provider

| Provider | Backend | Best For |
|---|---|---|
| `FAISSProvider` | FAISS IVF index | < 1M vectors, high recall |
| `ChromaProvider` | Chroma (persistent) | Metadata filtering, dev-friendly |

## 2. Smart Document Chunking (`src/rag/smart_splitter.py`)

Instead of blind fixed-size chunking, this module:

1. **Detects paper sections** via regex patterns (numbered, named, markdown-style headings)
2. **Splits within section boundaries** — no chunk spans across sections
3. **Adaptive chunk sizes**: 400 tokens for abstract/intro/conclusion, 800 for methods/experiments
4. **Preserves metadata**: section name, page numbers, chunk strategy

```python
from src.rag.smart_splitter import smart_split_documents
chunks = smart_split_documents(documents)
# Each chunk now has metadata: {"section": "3.1 Experiment Setup", "pages": [4,5]}
```

## 3. Hybrid Retrieval (`src/rag/hybrid_retriever.py`)

Pipeline: **BM25 (sparse) + Dense (vector) → RRF fusion → Cross-encoder rerank**

```
                ┌── BM25Retriever ──┐
Query ─────────┤                    ├── RRF(k=60) ──→ Reranker ──→ Top-K
                └── VectorStore ────┘
```

- **BM25**: Keyword-based sparse retrieval, catches exact term matches
- **Dense**: Semantic vector search, catches paraphrases and concepts
- **RRF**: Reciprocal Rank Fusion — parameter-free merge of both rankings
- **Reranker**: Cross-encoder (BGE-Reranker or ms-marco-MiniLM) for final relevance scoring

## 4. API Service Layer (`src/api/`)

Production-ready FastAPI server with Swagger UI auto-generation.

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET  /api/v1/health` — health + provider status
- `POST /api/v1/qa` — RAG QA with hybrid retrieval options
- `POST /api/v1/write` — multi-agent writing workflow
- `POST /api/v1/rag-write` — RAG + agent comprehensive writing

## 5. Data Flow

### RAG QA (with hybrid retrieval)
```
User Query
  → HybridRetriever.retrieve()
     → BM25Retriever.search()        [sparse]
     → VectorStore.similarity_search [dense]
     → reciprocal_rank_fusion()      [merge]
     → Reranker.rerank()             [refine]
  → LLM.generate(context + query)
  → Answer
```

### Writing Workflow
```
Initial State
  → Planner   — generates outline
  → Dispatch  — distributes to writers
  → [WriterA, WriterB] — parallel writing
  → Reviewer  — evaluates quality
  → Decision  — approve / revise outline / revise draft
  → Loop or End
```

## Extension Guide

### Adding a new LLM provider
1. Create class inheriting `BaseLLMProvider` in `src/providers/llm.py`
2. Implement `create()` and `provider_name`
3. Register in `LLMProviderFactory._registry`
4. Add config section to `from_config()`

### Adding a new vector store
1. Create class inheriting `BaseVectorStoreProvider` in `src/providers/vectorstore.py`
2. Implement `create()`, `load()`, and static `from_documents()`
3. Register in `VectorStoreFactory._registry`
