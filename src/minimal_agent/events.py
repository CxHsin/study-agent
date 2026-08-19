from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_RESPONSE = "model_response"
    MODEL_CONTENT_DELTA = "model_content_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    STEERING_MESSAGE_ACCEPTED = "steering_message_accepted"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_RESULT_PRODUCED = "tool_result_produced"
    FINAL_RESPONSE = "final_response"
    RUN_STOPPED = "run_stopped"
    RUN_ERROR = "run_error"
    LISTENER_ERROR = "listener_error"


@dataclass(frozen=True)
class AgentEvent:
    run_id: str
    kind: EventKind
    data: Mapping[str, object]
    sequence: int
    occurred_at: datetime
    elapsed_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


class AgentEventListener(Protocol):
    def __call__(self, event: AgentEvent) -> None: ...
