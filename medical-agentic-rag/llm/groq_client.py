"""Groq chat client - the primary and fully implemented LLM provider.

Handles native tool calling when the installed SDK exposes it, retries with
exponential backoff, timeouts, and malformed-argument recovery. The client never
falls back to a different (paid) provider; if Groq is unreachable it raises and
the orchestrator degrades to its deterministic, non-generative path.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any, Sequence

from config.settings import Settings, get_settings
from llm.base import (
    LLM_STATUS_MISSING_KEY,
    LLM_STATUS_OK,
    LLM_STATUS_SDK_MISSING,
    BaseLLMClient,
    LLMError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
    LLMToolCall,
    LLMUnavailableError,
    classify_llm_error,
)

logger = logging.getLogger(__name__)

_RETRYABLE_MARKERS = (
    "rate limit", "rate_limit", "timeout", "timed out", "temporarily",
    "503", "502", "504", "overloaded", "connection", "internal server error",
)


class GroqClient(BaseLLMClient):
    """Thin, defensive wrapper around the official ``groq`` SDK."""

    provider = "groq"
    supports_tool_calling = True

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any | None = None
        self._init_error: str | None = None
        self.model = self._settings.groq_model
        self._initialise()

    def _initialise(self) -> None:
        if not self._settings.has_llm_credentials:
            self._init_error = LLM_STATUS_MISSING_KEY
            logger.warning("Groq client disabled: %s", self._init_error)
            return
        try:
            from groq import Groq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on env
            self._init_error = f"{LLM_STATUS_SDK_MISSING}: {exc}"
            logger.warning("Groq client disabled: %s", self._init_error)
            return
        try:
            self._client = Groq(
                api_key=self._settings.groq_api_key,
                timeout=self._settings.request_timeout,
                max_retries=0,  # retries are handled here so backoff is observable
            )
        except Exception as exc:  # noqa: BLE001
            self._init_error = classify_llm_error(exc)
            logger.warning("Groq client disabled: %s", self._init_error)

    @property
    def available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------ chat
    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        if self._client is None:
            raise LLMUnavailableError(self._init_error or "Groq client unavailable")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": (
                self._settings.groq_temperature if temperature is None else temperature
            ),
            "max_tokens": max_tokens or self._settings.groq_max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = tool_choice or "auto"
        elif json_mode:
            # response_format is unsupported alongside tools on some models.
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                completion = self._client.chat.completions.create(**payload)
                return self._parse_completion(completion)
            except Exception as exc:  # noqa: BLE001 - normalise every SDK error
                last_error = exc
                message = str(exc).lower()
                if json_mode and "response_format" in message:
                    payload.pop("response_format", None)
                    logger.info("Model rejected response_format; retrying without it.")
                    continue
                if not self._is_retryable(message) or attempt >= self._settings.llm_max_retries:
                    break
                delay = min(8.0, (2**attempt) * 0.5) + random.uniform(0, 0.3)
                logger.warning(
                    "Groq call failed (attempt %d/%d): %s - retrying in %.1fs",
                    attempt + 1,
                    self._settings.llm_max_retries + 1,
                    exc,
                    delay,
                )
                time.sleep(delay)

        text = str(last_error or "unknown error")
        if "timeout" in text.lower() or "timed out" in text.lower():
            raise LLMTimeoutError(f"Groq request timed out: {text}") from last_error
        raise LLMError(f"Groq request failed: {text}") from last_error

    @staticmethod
    def _is_retryable(message: str) -> bool:
        return any(marker in message for marker in _RETRYABLE_MARKERS)

    @staticmethod
    def _parse_completion(completion: Any) -> LLMResponse:
        try:
            choice = completion.choices[0]
            message = choice.message
        except (AttributeError, IndexError) as exc:
            raise LLMResponseError("Groq returned no choices") from exc

        tool_calls: list[LLMToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            if function is None:
                continue
            raw_args = getattr(function, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Discarding tool call %s with unparseable arguments: %.200s",
                    getattr(function, "name", "?"),
                    raw_args,
                )
                continue
            if not isinstance(arguments, dict):
                continue
            tool_calls.append(
                LLMToolCall(
                    id=getattr(call, "id", "") or "",
                    name=getattr(function, "name", "") or "",
                    arguments=arguments,
                )
            )

        usage = {}
        raw_usage = getattr(completion, "usage", None)
        if raw_usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(raw_usage, key, None)
                if value is not None:
                    usage[key] = value

        return LLMResponse(
            text=(getattr(message, "content", None) or "").strip(),
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(completion, "model", None),
            usage=usage,
        )

    def health(self) -> dict[str, Any]:
        info = super().health()
        info["model"] = self.model
        if self._init_error:
            info["reason"] = self._init_error
        return info

    def ping(self) -> tuple[bool, str]:
        """Cheap connectivity probe used by the smoke test and the UI status panel.

        The returned message is always one of the fixed status strings (or a short
        model reply), never a raw exception, so it is safe to render.
        """
        if self._client is None:
            return False, self._init_error or "unavailable"
        try:
            response = self.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=8,
                temperature=0.0,
            )
            return True, response.text[:60] or "(empty reply)"
        except LLMError as exc:
            return False, classify_llm_error(exc)

    @property
    def status(self) -> str:
        """Human-readable status with no secret material."""
        return LLM_STATUS_OK if self.available else (self._init_error or "unavailable")
