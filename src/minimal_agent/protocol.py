from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

type ChatMessage = dict[str, object]


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming: bool = False
    tool_calls: bool = True
    cancellation: bool = False
    usage: bool = False
    prompt_cache: bool = False


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    estimated: bool = False
    latency_ms: float | None = None
    cost: float | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage | None = None
    provider_cache_hit: bool | None = None


@dataclass(frozen=True)
class ModelStreamChunk:
    """A provider-neutral incremental model response fragment."""

    content_delta: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""
    done: bool = False
    usage: ModelUsage | None = None
    provider_cache_hit: bool | None = None


class ModelAdapter(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse: ...

    def stream(self, messages: Sequence[ChatMessage]) -> Iterable[ModelStreamChunk]: ...

    def capabilities(self) -> ProviderCapabilities: ...
