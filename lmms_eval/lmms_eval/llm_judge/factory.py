import os
from typing import Optional

from .base import ServerInterface
from .protocol import ServerConfig
from .providers import AzureOpenAIProvider, OpenAIProvider


class ProviderFactory:
    """Factory for creating the configured VLM judge provider."""

    _provider_classes = {
        "azure": AzureOpenAIProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def create_provider(cls, api_type: Optional[str] = None, config: Optional[ServerConfig] = None) -> ServerInterface:
        if api_type is None:
            api_type = os.getenv("API_TYPE", "azure").lower()

        if api_type not in cls._provider_classes:
            raise ValueError(f"Unknown API type: {api_type}. Supported types: {list(cls._provider_classes.keys())}")

        return cls._provider_classes[api_type](config=config)

    @classmethod
    def register_provider(cls, api_type: str, judge_class: type):
        if not issubclass(judge_class, ServerInterface):
            raise ValueError(f"{judge_class} must be a subclass of ServerInterface")
        cls._provider_classes[api_type] = judge_class
