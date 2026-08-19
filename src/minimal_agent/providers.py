from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from minimal_agent.protocol import (
    AssistantMessage,
    ChatMessage,
    ContextSummaryMessage,
    ModelProfile,
    ModelRequest,
    ModelUsage,
    ProviderError,
    ProviderErrorKind,
    ProviderStreamEvent,
    StreamEnd,
    StreamError,
    SystemMessage,
    TextDelta,
    ToolCallDelta,
    ToolChoice,
    ToolDefinition,
    ToolResultMessage,
    UsageUpdate,
    UserMessage,
)

OPENAI_PROFILES = {
    "gpt-4o-mini": ModelProfile("openai", "gpt-4o-mini", 128_000, 16_384, parallel_tool_calls=True)
}
ANTHROPIC_PROFILES = {
    "claude-3-5-sonnet-latest": ModelProfile(
        "anthropic", "claude-3-5-sonnet-latest", 200_000, 8_192, parallel_tool_calls=True
    )
}


class MessageCodec(Protocol):
    def request(self, request: ModelRequest) -> dict[str, object]: ...

    def events(self, response: object) -> Iterable[ProviderStreamEvent]: ...


class OpenAIMessageCodec:
    def request(self, request: ModelRequest) -> dict[str, object]:
        options = request.options
        payload: dict[str, object] = {
            "messages": [_to_openai_message(message) for message in request.messages],
            "stream": options.stream,
        }
        if options.stream:
            payload["stream_options"] = {"include_usage": True}
        if options.tools and options.tool_choice is not ToolChoice.NONE:
            payload["tools"] = [_to_openai_tool(tool) for tool in options.tools]
            payload["tool_choice"] = options.tool_choice.value
            payload["parallel_tool_calls"] = True
        if options.max_output_tokens is not None:
            payload["max_tokens"] = options.max_output_tokens
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        timeout = options.metadata.get("request_timeout_seconds")
        if timeout is not None:
            payload["timeout"] = timeout
        return payload

    def events(self, response: object) -> Iterable[ProviderStreamEvent]:
        if hasattr(response, "choices"):
            yield from _openai_response_events(response)
            return
        call_ids: dict[int, str] = {}
        for chunk in response:  # type: ignore[union-attr]
            choices = getattr(chunk, "choices", ()) or ()
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    content = getattr(delta, "content", None)
                    if content:
                        yield TextDelta(str(content))
                    for call in getattr(delta, "tool_calls", ()) or ():
                        index = int(getattr(call, "index", 0))
                        call_id = getattr(call, "id", None)
                        if call_id:
                            call_ids[index] = str(call_id)
                        resolved_id = call_ids.get(index, "")
                        function = getattr(call, "function", None)
                        name = getattr(function, "name", None)
                        yield ToolCallDelta(
                            resolved_id,
                            str(name) if name else None,
                            str(getattr(function, "arguments", "") or ""),
                        )
            usage = _openai_usage(getattr(chunk, "usage", None))
            if usage is not None:
                yield UsageUpdate(usage, _openai_cache_hit(getattr(chunk, "usage", None)))
        yield StreamEnd()


class AnthropicMessageCodec:
    def request(self, request: ModelRequest) -> dict[str, object]:
        system: list[str] = []
        messages: list[dict[str, object]] = []
        for message in request.messages:
            if isinstance(message, SystemMessage):
                system.append(message.content)
            elif isinstance(message, UserMessage):
                messages.append({"role": "user", "content": message.content})
            elif isinstance(message, ContextSummaryMessage):
                messages.append(
                    {
                        "role": "user",
                        "content": f"[Context summary of earlier conversation]\n{message.content}",
                    }
                )
            elif isinstance(message, AssistantMessage):
                blocks: list[dict[str, object]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                blocks.extend(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": _decoded_arguments(call.arguments),
                    }
                    for call in message.tool_calls
                )
                messages.append({"role": "assistant", "content": blocks})
            elif isinstance(message, ToolResultMessage):
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.result.tool_call_id,
                    "content": message.result.to_json(),
                    "is_error": not message.result.ok,
                }
                previous_content = messages[-1]["content"] if messages else None
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(previous_content, list)
                    and all(item.get("type") == "tool_result" for item in previous_content)
                ):
                    previous_content.append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
        options = request.options
        payload: dict[str, object] = {
            "messages": messages,
            "max_tokens": options.max_output_tokens or 4096,
        }
        if system:
            payload["system"] = "\n\n".join(system)
        if options.tools and options.tool_choice is not ToolChoice.NONE:
            payload["tools"] = [_to_anthropic_tool(tool) for tool in options.tools]
            payload["tool_choice"] = _anthropic_tool_choice(options.tool_choice)
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        timeout = options.metadata.get("request_timeout_seconds")
        if timeout is not None:
            payload["timeout"] = timeout
        return payload

    def events(self, response: object) -> Iterable[ProviderStreamEvent]:
        if hasattr(response, "content"):
            yield from _anthropic_response_events(response)
            return
        call_ids: dict[int, str] = {}
        for event in response:  # type: ignore[union-attr]
            event_type = getattr(event, "type", "")
            if event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                if getattr(block, "type", "") == "tool_use":
                    index = int(getattr(event, "index", 0))
                    call_ids[index] = str(getattr(block, "id", ""))
                    yield ToolCallDelta(call_ids[index], str(getattr(block, "name", "")))
            elif event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", "")
                if delta_type == "text_delta":
                    yield TextDelta(str(getattr(delta, "text", "")))
                elif delta_type == "input_json_delta":
                    index = int(getattr(event, "index", 0))
                    yield ToolCallDelta(
                        call_ids.get(index, ""),
                        arguments_delta=str(getattr(delta, "partial_json", "")),
                    )
            elif event_type in ("message_start", "message_delta"):
                source = getattr(event, "message", None) or event
                usage = _anthropic_usage(getattr(source, "usage", None))
                if usage is not None:
                    yield UsageUpdate(usage)
        yield StreamEnd()


class OpenAIAdapter:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        profile: ModelProfile | None = None,
        client: Any | None = None,
        codec: MessageCodec | None = None,
    ) -> None:
        self.profile = _profile(model, profile, OPENAI_PROFILES)
        self.codec = codec or OpenAIMessageCodec()
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, max_retries=0)
        self._client = client

    def stream(self, request: ModelRequest) -> Iterable[ProviderStreamEvent]:
        try:
            payload = self.codec.request(request)
            if not self.profile.parallel_tool_calls:
                payload.pop("parallel_tool_calls", None)
            payload["model"] = self.profile.model
            response = self._client.chat.completions.create(**payload)
            yield from self.codec.events(response)
        except ProviderError as error:
            yield StreamError(error)
        except Exception as error:  # noqa: BLE001 - SDK failures are classified here
            yield StreamError(classify_provider_error(error, self.profile.provider))


class AnthropicAdapter:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "claude-3-5-sonnet-latest",
        profile: ModelProfile | None = None,
        client: Any | None = None,
        codec: MessageCodec | None = None,
    ) -> None:
        self.profile = _profile(model, profile, ANTHROPIC_PROFILES)
        self.codec = codec or AnthropicMessageCodec()
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise ProviderError(
                    ProviderErrorKind.UNSUPPORTED_CAPABILITY,
                    "Anthropic support requires the 'anthropic' optional dependency.",
                ) from error
            client = Anthropic(api_key=api_key, max_retries=0)
        self._client = client

    def stream(self, request: ModelRequest) -> Iterable[ProviderStreamEvent]:
        try:
            payload = self.codec.request(request)
            payload["model"] = self.profile.model
            stream_method = getattr(self._client.messages, "stream", None)
            if callable(stream_method) and request.options.stream:
                context = stream_method(**payload)
                with context as response:
                    yield from self.codec.events(response)
            else:
                response = self._client.messages.create(**payload)
                yield from self.codec.events(response)
        except ProviderError as error:
            yield StreamError(error)
        except Exception as error:  # noqa: BLE001 - SDK failures are classified here
            yield StreamError(classify_provider_error(error, self.profile.provider))


def classify_provider_error(error: Exception, provider: str) -> ProviderError:
    status = getattr(error, "status_code", None)
    request_id = getattr(error, "request_id", None)
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    retry_after = _float_or_none(
        headers.get("retry-after") if isinstance(headers, Mapping) else None
    )
    name = type(error).__name__.lower()
    message = str(error).lower()
    if status == 401 or "authentication" in name:
        kind, retryable = ProviderErrorKind.AUTHENTICATION, False
    elif status == 403 or "permission" in name:
        kind, retryable = ProviderErrorKind.AUTHORIZATION, False
    elif status == 429 or "ratelimit" in name:
        kind, retryable = ProviderErrorKind.RATE_LIMIT, True
    elif "timeout" in name:
        kind, retryable = ProviderErrorKind.TIMEOUT, True
    elif "connection" in name:
        kind, retryable = ProviderErrorKind.NETWORK, True
    elif "context" in message and ("length" in message or "token" in message):
        kind, retryable = ProviderErrorKind.CONTEXT_LENGTH, False
    elif isinstance(status, int) and status >= 500:
        kind, retryable = ProviderErrorKind.SERVER, True
    elif isinstance(status, int) and status >= 400:
        kind, retryable = ProviderErrorKind.INVALID_REQUEST, False
    else:
        kind, retryable = ProviderErrorKind.UNKNOWN, False
    normalized = ProviderError(
        kind,
        f"{provider} request failed ({kind.value}).",
        retryable=retryable,
        status_code=status if isinstance(status, int) else None,
        request_id=str(request_id) if request_id else None,
        retry_after=retry_after,
    )
    normalized.__cause__ = error
    return normalized


def _to_openai_message(message: ChatMessage) -> dict[str, object]:
    if isinstance(message, (SystemMessage, UserMessage)):
        return {
            "role": "system" if isinstance(message, SystemMessage) else "user",
            "content": message.content,
        }
    if isinstance(message, ContextSummaryMessage):
        return {
            "role": "user",
            "content": f"[Context summary of earlier conversation]\n{message.content}",
        }
    if isinstance(message, AssistantMessage):
        payload: dict[str, object] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in message.tool_calls
            ]
        return payload
    return {
        "role": "tool",
        "tool_call_id": message.result.tool_call_id,
        "content": message.result.to_json(),
    }


def _to_openai_tool(definition: ToolDefinition) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": dict(definition.input_schema),
            "strict": definition.strict,
        },
    }


def _to_anthropic_tool(definition: ToolDefinition) -> dict[str, object]:
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": dict(definition.input_schema),
    }


def _openai_response_events(response: object) -> Iterable[ProviderStreamEvent]:
    choices = getattr(response, "choices", ()) or ()
    if not choices:
        raise ProviderError(ProviderErrorKind.INVALID_RESPONSE, "OpenAI returned no choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if content:
        yield TextDelta(str(content))
    for call in getattr(message, "tool_calls", ()) or ():
        function = getattr(call, "function", None)
        yield ToolCallDelta(
            str(getattr(call, "id", "")),
            str(getattr(function, "name", "")),
            str(getattr(function, "arguments", "")),
        )
    usage_value = getattr(response, "usage", None)
    usage = _openai_usage(usage_value)
    if usage is not None:
        yield UsageUpdate(usage, _openai_cache_hit(usage_value))
    yield StreamEnd()


def _anthropic_response_events(response: object) -> Iterable[ProviderStreamEvent]:
    for block in getattr(response, "content", ()) or ():
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            yield TextDelta(str(block.text))
        elif getattr(block, "type", None) == "tool_use":
            yield ToolCallDelta(
                str(block.id),
                str(block.name),
                json.dumps(getattr(block, "input", {}), ensure_ascii=False, separators=(",", ":")),
            )
    usage = _anthropic_usage(getattr(response, "usage", None))
    if usage is not None:
        yield UsageUpdate(usage)
    yield StreamEnd()


def _openai_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    return ModelUsage(
        _int_or_none(getattr(value, "prompt_tokens", None)),
        _int_or_none(getattr(value, "completion_tokens", None)),
        _int_or_none(getattr(value, "prompt_cache_hit_tokens", None)),
    )


def _openai_cache_hit(value: object) -> bool | None:
    cached = _int_or_none(getattr(value, "prompt_cache_hit_tokens", None))
    return None if cached is None else cached > 0


def _anthropic_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    return ModelUsage(
        _int_or_none(getattr(value, "input_tokens", None)),
        _int_or_none(getattr(value, "output_tokens", None)),
        _int_or_none(getattr(value, "cache_read_input_tokens", None)),
    )


def _profile(
    model: str, profile: ModelProfile | None, known: Mapping[str, ModelProfile]
) -> ModelProfile:
    if profile is not None:
        if profile.model != model:
            raise ValueError("Explicit ModelProfile must match the configured model name.")
        return profile
    if model not in known:
        raise ValueError(f"Unknown model {model!r}; provide an explicit ModelProfile.")
    return known[model]


def _anthropic_tool_choice(value: ToolChoice) -> dict[str, str]:
    return {
        "type": {ToolChoice.AUTO: "auto", ToolChoice.NONE: "none", ToolChoice.REQUIRED: "any"}[
            value
        ]
    }


def _decoded_arguments(value: str) -> object:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ProviderError(
            ProviderErrorKind.INVALID_REQUEST,
            "Anthropic cannot encode malformed Tool Call arguments.",
        ) from error
    if not isinstance(decoded, Mapping):
        raise ProviderError(
            ProviderErrorKind.INVALID_REQUEST,
            "Anthropic Tool Call arguments must be a JSON object.",
        )
    return decoded


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
