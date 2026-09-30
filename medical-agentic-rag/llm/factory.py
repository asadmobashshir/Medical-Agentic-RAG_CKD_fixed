"""LLM provider factory for Google Gemini."""
from __future__ import annotations
import logging
from typing import Callable
from config.settings import Settings, get_settings
from llm.base import BaseLLMClient, UnavailableLLMClient, classify_llm_error
from llm.gemini_client import GeminiClient
logger = logging.getLogger(__name__)
_PROVIDERS: dict[str, Callable[[Settings], BaseLLMClient]] = {"gemini": lambda settings: GeminiClient(settings)}
def build_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    settings = settings or get_settings()
    factory = _PROVIDERS.get(settings.llm_provider)
    if factory is None:
        return UnavailableLLMClient(f"Unknown llm_provider: {settings.llm_provider}")
    try:
        client = factory(settings)
    except Exception as exc:
        return UnavailableLLMClient(classify_llm_error(exc))
    if not client.available:
        return UnavailableLLMClient(str(client.health().get("reason", "LLM backend unavailable")))
    return client
