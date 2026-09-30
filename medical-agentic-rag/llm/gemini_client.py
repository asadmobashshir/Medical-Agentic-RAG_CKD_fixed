"""Google Gemini API client for the CKD Agentic RAG system."""
from __future__ import annotations
import logging, random, time
from typing import Any, Sequence
from config.settings import Settings, get_settings
from llm.base import (
    LLM_STATUS_MISSING_KEY, LLM_STATUS_OK, LLM_STATUS_SDK_MISSING,
    BaseLLMClient, LLMError, LLMResponse, LLMResponseError,
    LLMTimeoutError, LLMUnavailableError, classify_llm_error,
)
logger = logging.getLogger(__name__)
_RETRYABLE_MARKERS = ("rate limit","resource exhausted","429","timeout","timed out",
                      "temporarily","503","502","504","overloaded","connection",
                      "internal server error")

class GeminiClient(BaseLLMClient):
    provider = "gemini"
    # The planner uses its provider-neutral JSON path. The application's ToolRegistry
    # remains responsible for executing tools.
    supports_tool_calling = False

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: Any | None = None
        self._init_error: str | None = None
        self.model = self._settings.gemini_model
        self._initialise()

    def _initialise(self) -> None:
        if not self._settings.has_llm_credentials:
            self._init_error = LLM_STATUS_MISSING_KEY
            return
        try:
            from google import genai
        except ImportError as exc:
            self._init_error = f"{LLM_STATUS_SDK_MISSING}: {exc}"
            return
        try:
            self._client = genai.Client(api_key=self._settings.gemini_api_key)
        except Exception as exc:
            self._init_error = classify_llm_error(exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def _convert_messages(messages: Sequence[dict[str, Any]]):
        from google.genai import types
        system_parts, contents = [], []
        for message in messages:
            role = str(message.get("role", "user")).lower()
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(x.get("text", "")) if isinstance(x, dict) else str(x)
                    for x in content
                )
            content = str(content or "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            gemini_role = "model" if role in {"assistant", "model"} else "user"
            if content:
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part(text=content)]
                ))
        if not contents:
            contents.append(types.Content(role="user", parts=[types.Part(text="Please respond.")]))
        return "\n\n".join(system_parts) or None, contents

    def chat(self, messages: Sequence[dict[str, Any]], *, tools=None,
             tool_choice=None, temperature=None, max_tokens=None, json_mode=False) -> LLMResponse:
        if self._client is None:
            raise LLMUnavailableError(self._init_error or "Gemini client unavailable")
        if tools:
            raise LLMError("Gemini adapter uses JSON planning; native tool calls are disabled.")
        try:
            from google.genai import types
        except ImportError as exc:
            raise LLMError(f"{LLM_STATUS_SDK_MISSING}: {exc}") from exc

        system_instruction, contents = self._convert_messages(messages)
        kwargs = {
            "temperature": self._settings.gemini_temperature if temperature is None else temperature,
            "max_output_tokens": max_tokens or self._settings.gemini_max_tokens,
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**kwargs)

        last_error = None
        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config
                )
                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    raise LLMResponseError("Gemini returned an empty response.")
                return LLMResponse(text=text, model=self.model)
            except Exception as exc:
                last_error = exc
                msg = str(exc).lower()
                if not any(x in msg for x in _RETRYABLE_MARKERS) or attempt >= self._settings.llm_max_retries:
                    break
                delay = min(8.0, (2 ** attempt) * 0.5) + random.uniform(0, 0.3)
                time.sleep(delay)

        text = str(last_error or "unknown error")
        if "timeout" in text.lower() or "timed out" in text.lower():
            raise LLMTimeoutError(f"Gemini request timed out: {text}") from last_error
        raise LLMError(f"Gemini request failed: {text}") from last_error

    def health(self):
        info = super().health()
        info["model"] = self.model
        if self._init_error:
            info["reason"] = self._init_error
        return info

    def ping(self):
        if self._client is None:
            return False, self._init_error or "unavailable"
        try:
            response = self.chat(
                [{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=8, temperature=0.0
            )
            return True, response.text[:60] or "(empty reply)"
        except LLMError as exc:
            return False, classify_llm_error(exc)

    @property
    def status(self):
        return LLM_STATUS_OK if self.available else (self._init_error or "unavailable")
