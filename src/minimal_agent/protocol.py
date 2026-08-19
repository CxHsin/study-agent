from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

type ChatMessage = dict[str, object]


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ModelStreamChunk:
    """A provider-neutral incremental model response fragment."""

    content_delta: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments_delta: str = ""
    done: bool = False


class ModelAdapter(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse: ...

    def stream(self, messages: Sequence[ChatMessage]) -> Iterable[ModelStreamChunk]: ...
