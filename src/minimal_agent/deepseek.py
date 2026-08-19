from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from minimal_agent.protocol import (
    ModelProfile,
    ModelRequest,
    ProviderError,
    ProviderStreamEvent,
    StreamError,
)
from minimal_agent.providers import OpenAIMessageCodec, classify_provider_error

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_PROFILE = ModelProfile(
    "deepseek",
    DEEPSEEK_MODEL,
    128_000,
    8_192,
    parallel_tool_calls=True,
    prompt_cache=True,
)


class DeepSeekAdapter:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEEPSEEK_MODEL,
        profile: ModelProfile | None = None,
        client: Any | None = None,
    ) -> None:
        if profile is None:
            if model != DEEPSEEK_MODEL:
                raise ValueError(f"Unknown model {model!r}; provide an explicit ModelProfile.")
            profile = DEEPSEEK_PROFILE
        if profile.model != model:
            raise ValueError("Explicit ModelProfile must match the configured model name.")
        self.profile = profile
        self.codec = OpenAIMessageCodec()
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, max_retries=0)
        self._client = client

    def stream(self, request: ModelRequest) -> Iterable[ProviderStreamEvent]:
        try:
            payload = self.codec.request(request)
            if not self.profile.parallel_tool_calls:
                payload.pop("parallel_tool_calls", None)
            payload.update(
                {
                    "model": self.profile.model,
                    "extra_body": {"thinking": {"type": "disabled"}},
                }
            )
            response = self._client.chat.completions.create(**payload)
            yield from self.codec.events(response)
        except ProviderError as error:
            yield StreamError(error)
        except Exception as error:  # noqa: BLE001 - SDK failures are classified here
            yield StreamError(classify_provider_error(error, self.profile.provider))
