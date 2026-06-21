"""
Vector Store Provider abstraction layer.

Supports:
    - FAISS (local, fast, good for <1M vectors)
    - Chroma (persistent, metadata filtering)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings


class BaseVectorStoreProvider(ABC):
    """Abstract vector store backend."""

    @abstractmethod
    def create(self, embedding: Embeddings, **kwargs) -> VectorStore:
        """Return a fresh vector store instance."""
        ...

    @abstractmethod
    def load(self, embedding: Embeddings, path: str, **kwargs) -> VectorStore:
        """Load a persisted vector store from disk."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


class FAISSProvider(BaseVectorStoreProvider):
    """FAISS backend - fast in-memory / disk-persisted IVF index."""

    def __init__(self, index_path: str = "data/faiss_index"):
        self._index_path = index_path

    @property
    def provider_name(self) -> str:
        return "faiss"

    def create(self, embedding: Embeddings, **kwargs) -> VectorStore:
        from langchain_community.vectorstores import FAISS
        return FAISS  # not created here; returned for from_documents pattern

    def load(self, embedding: Embeddings, path: str = None, **kwargs) -> VectorStore:
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(
            path or self._index_path,
            embedding,
            allow_dangerous_deserialization=True,
        )

    @staticmethod
    def from_documents(
        docs: List[Document],
        embedding: Embeddings,
        index_path: str = "data/faiss_index",
    ) -> VectorStore:
        from langchain_community.vectorstores import FAISS
        store = FAISS.from_documents(docs, embedding)
        import os
        os.makedirs(index_path, exist_ok=True)
        store.save_local(index_path)
        return store


class ChromaProvider(BaseVectorStoreProvider):
    """Chroma backend - persistent, metadata-rich, good for filtering."""

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self._persist_dir = persist_dir

    @property
    def provider_name(self) -> str:
        return "chroma"

    def create(self, embedding: Embeddings, **kwargs) -> VectorStore:
        from langchain_chroma import Chroma
        return Chroma

    def load(self, embedding: Embeddings, path: str = None, **kwargs) -> VectorStore:
        from langchain_chroma import Chroma
        return Chroma(
            persist_directory=path or self._persist_dir,
            embedding_function=embedding,
        )

    @staticmethod
    def from_documents(
        docs: List[Document],
        embedding: Embeddings,
        persist_dir: str = "data/chroma_db",
    ) -> VectorStore:
        from langchain_chroma import Chroma
        return Chroma.from_documents(
            docs, embedding,
            persist_directory=persist_dir,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class VectorStoreFactory:
    _registry = {
        "faiss": FAISSProvider,
        "chroma": ChromaProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type) -> None:
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseVectorStoreProvider:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown vector store '{name}'. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def from_config(cls):
        import os
        from ..config import Config

        provider_name = os.getenv("VECTORSTORE_PROVIDER", "faiss").lower()

        if provider_name == "faiss":
            return FAISSProvider(index_path=Config.INDEX_DIR)
        elif provider_name == "chroma":
            return ChromaProvider(
                persist_dir=os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
            )
        else:
            raise ValueError(f"Unsupported vector store: {provider_name}")
