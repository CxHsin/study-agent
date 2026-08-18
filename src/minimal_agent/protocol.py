from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

type ChatMessage = dict[str, object]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class ModelAdapter(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse: ...
