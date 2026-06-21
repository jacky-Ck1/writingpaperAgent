"""
LLM Provider abstraction layer - pluggable backend design.

Supports:
    - DeepSeek (via OpenAI-compatible API)
    - OpenAI (GPT-4, GPT-3.5, etc.)
    - Local models (via vLLM / Ollama OpenAI-compatible endpoint)

Design: dependency inversion; all consumers depend on BaseLLMProvider interface,
not concrete implementations. Switch provider via config/env only.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


class BaseLLMProvider(ABC):
    """Abstract LLM provider - all backends implement this."""

    @abstractmethod
    def create(self, **kwargs) -> BaseChatModel:
        """Return a LangChain-compatible chat model instance."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
        ...


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model

    @property
    def provider_name(self) -> str:
        return "deepseek"

    def create(self, **kwargs) -> BaseChatModel:
        return ChatOpenAI(
            model=kwargs.pop("model", self._model),
            api_key=kwargs.pop("api_key", self._api_key),
            base_url=kwargs.pop("base_url", self._base_url),
            **kwargs,
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI native provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    @property
    def provider_name(self) -> str:
        return "openai"

    def create(self, **kwargs) -> BaseChatModel:
        params = {
            "model": kwargs.pop("model", self._model),
            "api_key": kwargs.pop("api_key", self._api_key),
        }
        if self._base_url:
            params["base_url"] = self._base_url
        params.update(kwargs)
        return ChatOpenAI(**params)


class LocalProvider(BaseLLMProvider):
    """Local models via vLLM / Ollama / any OpenAI-compatible server.

    Typical usage:
        LOCAL_LLM_BASE_URL=http://localhost:11434/v1
        LOCAL_LLM_MODEL=qwen2.5:7b
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "qwen2.5:7b",
        api_key: str = "not-needed",
    ):
        self._base_url = base_url
        self._model = model
        self._api_key = api_key

    @property
    def provider_name(self) -> str:
        return "local"

    def create(self, **kwargs) -> BaseChatModel:
        return ChatOpenAI(
            model=kwargs.pop("model", self._model),
            api_key=kwargs.pop("api_key", self._api_key),
            base_url=kwargs.pop("base_url", self._base_url),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMProviderFactory:
    """Factory that wires provider selection from config / env.

    Usage:
        llm = LLMProviderFactory.from_config()
        model = llm.create(temperature=0.7)
    """

    _registry: Dict[str, type] = {
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
        "local": LocalProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type) -> None:
        """Extend with custom providers."""
        cls._registry[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLLMProvider:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider '{name}'. Available: {list(cls._registry)}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def from_config(cls):
        """Read provider config from environment and return a configured instance."""
        import os
        from ..config import Config

        provider_name = os.getenv("LLM_PROVIDER", "deepseek").lower()

        if provider_name == "deepseek":
            return DeepSeekProvider(
                api_key=Config.DEEPSEEK_API_KEY,
                base_url=Config.DEEPSEEK_BASE_URL,
                model=Config.DEEPSEEK_MODEL,
            )
        elif provider_name == "openai":
            return OpenAIProvider(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
        elif provider_name == "local":
            return LocalProvider(
                base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
                model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
