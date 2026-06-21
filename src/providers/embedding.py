"""
Embedding Provider abstraction layer - pluggable embedding backends.

Supports:
    - HuggingFace BGE / multilingual models (local)
    - OpenAI text-embedding-3 (API)
    - Local embedding server (Ollama / text-embeddings-inference)
"""

from abc import ABC, abstractmethod
from typing import List
from langchain_core.embeddings import Embeddings


class BaseEmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def create(self, **kwargs) -> Embeddings:
        """Return a LangChain-compatible embedding instance."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


class BGEProvider(BaseEmbeddingProvider):
    """HuggingFace BGE models, runs locally.

    Default: BAAI/bge-base-en-v1.5 (good balance of speed/quality).
    Chinese: BAAI/bge-large-zh-v1.5
    Multilingual: BAAI/bge-m3
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._device = device
        self._normalize = normalize

    @property
    def provider_name(self) -> str:
        return "bge"

    def create(self, **kwargs) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings  # 0.1+ path

        model_name = kwargs.pop("model_name", self._model_name)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": kwargs.pop("device", self._device)},
            encode_kwargs={"normalize_embeddings": kwargs.pop("normalize", self._normalize)},
            **kwargs,
        )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings (text-embedding-3-small / text-embedding-3-large)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    @property
    def provider_name(self) -> str:
        return "openai"

    def create(self, **kwargs) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        params = {
            "model": kwargs.pop("model", self._model),
            "api_key": kwargs.pop("api_key", self._api_key),
        }
        if self._base_url:
            params["base_url"] = self._base_url
        params.update(kwargs)
        return OpenAIEmbeddings(**params)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding server (Ollama / TEI) via OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
        api_key: str = "not-needed",
    ):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "local"

    def create(self, **kwargs) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=kwargs.pop("model", self._model),
            api_key=kwargs.pop("api_key", self._api_key),
            base_url=kwargs.pop("base_url", self._base_url),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class EmbeddingProviderFactory:
    _registry = {
        "bge": BGEProvider,
        "openai": OpenAIEmbeddingProvider,
        "local": LocalEmbeddingProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type) -> None:
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseEmbeddingProvider:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown embedding provider '{name}'. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def from_config(cls):
        import os
        from ..config import Config

        provider_name = os.getenv("EMBEDDING_PROVIDER", "bge").lower()

        if provider_name == "bge":
            return BGEProvider(
                model_name=Config.EMBEDDING_MODEL,
                device=Config.EMBEDDING_DEVICE,
            )
        elif provider_name == "openai":
            return OpenAIEmbeddingProvider(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            )
        elif provider_name == "local":
            return LocalEmbeddingProvider(
                base_url=os.getenv("LOCAL_EMBEDDING_BASE_URL", "http://localhost:11434/v1"),
                model=os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-embed-text"),
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name}")
