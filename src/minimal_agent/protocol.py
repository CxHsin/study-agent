from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

MESSAGE_SCHEMA_VERSION = 2


class ToolError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError("Tool Call ID and name cannot be empty.")


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    ok: bool
    data: object = None
    error: ToolError | None = None
    retryable: bool = False

    def to_json(self) -> str:
        payload: dict[str, object] = {"ok": self.ok}
        if self.ok:
            payload["data"] = self.data
        elif self.error is not None:
            payload["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "retryable": self.retryable,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, object]
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Tool Definition name cannot be empty.")
        object.__setattr__(self, "input_schema", MappingProxyType(dict(self.input_schema)))


class _MessageMapping:
    @property
    def role(self) -> str:
        return str(_legacy_message_dict(self)["role"])

    def __getitem__(self, key: str) -> object:
        return _legacy_message_dict(self)[key]

    def get(self, key: str, default: object = None) -> object:
        return _legacy_message_dict(self).get(key, default)

    def values(self):
        return _legacy_message_dict(self).values()

    def __contains__(self, key: object) -> bool:
        return key in _legacy_message_dict(self)


@dataclass(frozen=True)
class SystemMessage(_MessageMapping):
    content: str


@dataclass(frozen=True)
class UserMessage(_MessageMapping):
    content: str


@dataclass(frozen=True)
class AssistantMessage(_MessageMapping):
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.content is None and not self.tool_calls:
            raise ValueError("AssistantMessage requires text or at least one Tool Call.")


@dataclass(frozen=True)
class ToolResultMessage(_MessageMapping):
    result: ToolResult


@dataclass(frozen=True)
class ContextSummaryMessage(_MessageMapping):
    content: str
    summary_id: str


type ChatMessage = (
    SystemMessage | UserMessage | AssistantMessage | ToolResultMessage | ContextSummaryMessage
)
type LegacyMessage = Mapping[str, object]


def message_to_dict(message: ChatMessage, *, include_version: bool = True) -> dict[str, object]:
    data: dict[str, object]
    if isinstance(message, SystemMessage):
        data = {"role": "system", "content": message.content}
    elif isinstance(message, UserMessage):
        data = {"role": "user", "content": message.content}
    elif isinstance(message, AssistantMessage):
        data = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            data["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ]
    elif isinstance(message, ToolResultMessage):
        data = {
            "role": "tool",
            "tool_call_id": message.result.tool_call_id,
            "tool_name": message.result.tool_name,
            "result": _tool_result_to_dict(message.result),
        }
    else:
        data = {
            "role": "context_summary",
            "content": message.content,
            "summary_id": message.summary_id,
        }
    if include_version:
        data["schema_version"] = MESSAGE_SCHEMA_VERSION
    return data


def _legacy_message_dict(message: ChatMessage) -> dict[str, object]:
    data = message_to_dict(message, include_version=False)
    if isinstance(message, AssistantMessage) and message.tool_calls:
        data["tool_calls"] = message.tool_calls
    elif isinstance(message, ToolResultMessage):
        data["content"] = message.result.to_json()
    return data


def message_from_dict(
    raw: Mapping[str, object], *, tool_names: Mapping[str, str] | None = None
) -> ChatMessage:
    version = raw.get("schema_version", 1)
    if version not in (1, MESSAGE_SCHEMA_VERSION):
        raise ValueError(f"Unsupported message schema version: {version}")
    role = raw.get("role")
    if role == "system":
        return SystemMessage(_required_text(raw, "content"))
    if role == "user":
        return UserMessage(_required_text(raw, "content"))
    if role == "context_summary":
        return ContextSummaryMessage(
            _required_text(raw, "content"), _required_text(raw, "summary_id")
        )
    if role == "assistant":
        calls = tuple(_tool_call_from_value(item) for item in _sequence(raw.get("tool_calls")))
        content = raw.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("Assistant message content must be text or null.")
        return AssistantMessage(content, calls)
    if role == "tool":
        call_id = _required_text(raw, "tool_call_id")
        tool_name = str(raw.get("tool_name") or (tool_names or {}).get(call_id, ""))
        result_value = raw.get("result", raw.get("content"))
        return ToolResultMessage(_tool_result_from_value(call_id, tool_name, result_value))
    raise ValueError(f"Unsupported message role: {role!r}")


def normalize_messages(messages: Iterable[ChatMessage | LegacyMessage]) -> tuple[ChatMessage, ...]:
    normalized: list[ChatMessage] = []
    calls: dict[str, str] = {}
    completed: set[str] = set()
    pending: set[str] = set()
    for raw in messages:
        message = raw if _is_message(raw) else message_from_dict(raw, tool_names=calls)
        if isinstance(message, AssistantMessage):
            if pending:
                raise ValueError("Assistant message arrived before pending Tool Results.")
            for call in message.tool_calls:
                if call.id in calls:
                    raise ValueError(f"Duplicate Tool Call ID: {call.id}")
                calls[call.id] = call.name
                pending.add(call.id)
        elif isinstance(message, ToolResultMessage):
            call_id = message.result.tool_call_id
            if call_id not in calls:
                raise ValueError(f"Tool Result has no preceding Tool Call: {call_id}")
            if call_id in completed:
                raise ValueError(f"Duplicate Tool Result: {call_id}")
            if not message.result.tool_name:
                result = message.result
                message = ToolResultMessage(
                    ToolResult(
                        call_id,
                        calls[call_id],
                        result.ok,
                        result.data,
                        result.error,
                        result.retryable,
                    )
                )
            completed.add(call_id)
            pending.remove(call_id)
        elif pending:
            raise ValueError("Non-tool message arrived before pending Tool Results.")
        normalized.append(message)
    return tuple(normalized)


class ProviderErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    INVALID_REQUEST = "invalid_request"
    CONTEXT_LENGTH = "context_length"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    SERVER = "server"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN = "unknown"


class ProviderError(RuntimeError):
    def __init__(
        self,
        kind: ProviderErrorKind,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        self.request_id = request_id
        self.retry_after = retry_after


class ModelError(ProviderError):
    """Compatibility exception; new code should raise ProviderError with a stable kind."""

    def __init__(self, message: str) -> None:
        super().__init__(ProviderErrorKind.UNKNOWN, message)


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming: bool = False
    tool_calls: bool = True
    parallel_tool_calls: bool = False
    cancellation: bool = False
    usage: bool = False
    prompt_cache: bool = False


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model: str
    context_window_tokens: int
    max_output_tokens: int
    streaming: bool = True
    tool_calls: bool = True
    parallel_tool_calls: bool = False
    usage: bool = True
    prompt_cache: bool = False
    cancellation: bool = False

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("Model token limits must be positive.")
        if self.max_output_tokens > self.context_window_tokens:
            raise ValueError("Model output limit cannot exceed its context window.")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            self.streaming,
            self.tool_calls,
            self.parallel_tool_calls,
            self.cancellation,
            self.usage,
            self.prompt_cache,
        )


class ToolChoice(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass(frozen=True)
class RequestOptions:
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: ToolChoice = ToolChoice.AUTO
    max_output_tokens: int | None = None
    temperature: float | None = None
    stream: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_choice", ToolChoice(self.tool_choice))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive.")


@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[ChatMessage, ...]
    options: RequestOptions = field(default_factory=RequestOptions)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", normalize_messages(self.messages))


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    estimated: bool = False
    latency_ms: float | None = None
    cost: float | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage | None = None
    provider_cache_hit: bool | None = None


@dataclass(frozen=True)
class ModelStreamChunk:
    """Compatibility shape for simple in-process test models."""

    content_delta: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""
    done: bool = False
    usage: ModelUsage | None = None
    provider_cache_hit: bool | None = None


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    call_id: str
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class UsageUpdate:
    usage: ModelUsage
    provider_cache_hit: bool | None = None


@dataclass(frozen=True)
class StreamEnd:
    pass


@dataclass(frozen=True)
class StreamError:
    error: ProviderError


type ProviderStreamEvent = TextDelta | ToolCallDelta | UsageUpdate | StreamEnd | StreamError


class ModelAdapter(Protocol):
    profile: ModelProfile

    def stream(self, request: ModelRequest) -> Iterable[ProviderStreamEvent]: ...


def _tool_call_from_value(value: object) -> ToolCall:
    if isinstance(value, ToolCall):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Tool Call must be an object.")
    return ToolCall(
        _required_text(value, "id"),
        _required_text(value, "name"),
        _required_text(value, "arguments"),
    )


def _tool_result_to_dict(result: ToolResult) -> dict[str, object]:
    payload: dict[str, object] = {"ok": result.ok, "retryable": result.retryable}
    if result.ok:
        payload["data"] = result.data
    elif result.error is not None:
        payload["error"] = {"code": result.error.code, "message": result.error.message}
    return payload


def _tool_result_from_value(call_id: str, tool_name: str, value: object) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ToolResult(call_id, tool_name, True, data=value)
    if not isinstance(value, Mapping):
        raise TypeError("Tool Result must be an object or JSON string.")
    ok = bool(value.get("ok", True))
    retryable = bool(value.get("retryable", False))
    error_value = value.get("error")
    error = None
    if not ok and isinstance(error_value, Mapping):
        retryable = bool(error_value.get("retryable", retryable))
        error = ToolError(
            str(error_value.get("code", "TOOL_ERROR")),
            str(error_value.get("message", "Tool execution failed.")),
            retryable,
        )
    return ToolResult(call_id, tool_name, ok, value.get("data"), error, retryable)


def _required_text(value: Mapping[str, object], field_name: str) -> str:
    item = value.get(field_name)
    if not isinstance(item, str):
        raise TypeError(f"Message field {field_name!r} must be text.")
    return item


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _is_message(value: object) -> bool:
    return isinstance(
        value,
        (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage, ContextSummaryMessage),
    )
