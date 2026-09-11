"""LLM provider factory.

Groq is the default and only fully implemented provider. Registering another
provider means adding it to ``_PROVIDERS``; the rest of the codebase depends
only on :class:`llm.base.BaseLLMClient`.
"""

from __future__ import annotations

import logging
from typing import Callable

from config.settings import Settings, get_settings
from llm.base import BaseLLMClient, UnavailableLLMClient, classify_llm_error
from llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

_PROVIDERS: dict[str, Callable[[Settings], BaseLLMClient]] = {
    "groq": lambda settings: GroqClient(settings),
}


def build_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Return the configured LLM client, or an explicit unavailable stub.

    The stub is deliberate: the orchestrator must be able to detect that no
    generative backend exists and switch to its evidence-only path rather than
    silently degrading into invention.
    """
    settings = settings or get_settings()
    factory = _PROVIDERS.get(settings.llm_provider)
    if factory is None:
        return UnavailableLLMClient(f"Unknown llm_provider: {settings.llm_provider}")

    try:
        client = factory(settings)
    except Exception as exc:  # noqa: BLE001
        reason = classify_llm_error(exc)
        logger.warning("Failed to build LLM client: %s", reason)
        return UnavailableLLMClient(reason)

    if not client.available:
        reason = client.health().get("reason", "LLM backend unavailable")
        logger.warning("LLM backend unavailable: %s", reason)
        return UnavailableLLMClient(str(reason))
    return client
