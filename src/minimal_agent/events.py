from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AgentEvent:
    kind: str
    data: dict[str, object]
    sequence: int


class AgentEventListener(Protocol):
    def __call__(self, event: AgentEvent) -> None: ...
