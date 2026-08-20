from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

from minimal_agent.cost import UsageRecord

if TYPE_CHECKING:
    from minimal_agent.run import RunError


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_CALL_STARTED = "model_call_started"
    PROVIDER_ATTEMPT_STARTED = "provider_attempt_started"
    PROVIDER_ATTEMPT_FAILED = "provider_attempt_failed"
    PROVIDER_RETRY_SCHEDULED = "provider_retry_scheduled"
    MODEL_USAGE_RECORDED = "model_usage_recorded"
    MODEL_RESPONSE = "model_response"
    MODEL_CONTENT_DELTA = "model_content_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    STEERING_MESSAGE_ACCEPTED = "steering_message_accepted"
    TOOL_CALL_REQUESTED = "tool_call_requested"
    TOOL_CONFIRMATION_REQUESTED = "tool_confirmation_requested"
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

    def content_delta(self) -> str | None:
        value = self.data.get("content_delta")
        return (
            value if self.kind is EventKind.MODEL_CONTENT_DELTA and isinstance(value, str) else None
        )

    def tool_call(self) -> TraceToolCall | None:
        if self.kind is not EventKind.TOOL_CALL_REQUESTED:
            return None
        call_id = self.data.get("tool_call_id")
        name = self.data.get("name")
        arguments = self.data.get("arguments")
        if not all(isinstance(value, str) for value in (call_id, name, arguments)):
            return None
        return TraceToolCall(cast(str, call_id), cast(str, name), cast(str, arguments))

    def confirmation_request(self) -> ToolConfirmationRequest | None:
        if self.kind is not EventKind.TOOL_CONFIRMATION_REQUESTED:
            return None
        run_id = self.data.get("run_id")
        call_id = self.data.get("tool_call_id")
        name = self.data.get("name")
        arguments = self.data.get("arguments")
        if not all(isinstance(value, str) for value in (run_id, call_id, name)) or not isinstance(
            arguments, Mapping
        ):
            return None
        return ToolConfirmationRequest(
            cast(str, run_id),
            cast(str, call_id),
            cast(str, name),
            MappingProxyType(dict(arguments)),
        )

    def tool_result(self) -> TraceToolResult | None:
        if self.kind is not EventKind.TOOL_RESULT_PRODUCED:
            return None
        call_id = self.data.get("tool_call_id")
        name = self.data.get("name")
        success = self.data.get("success")
        result = self.data.get("result")
        retryable = self.data.get("retryable", False)
        if (
            not isinstance(call_id, str)
            or not isinstance(name, str)
            or not isinstance(success, bool)
            or not isinstance(result, str)
            or not isinstance(retryable, bool)
        ):
            return None
        return TraceToolResult(call_id, name, success, result, retryable)

    def final_content(self) -> str | None:
        value = self.data.get("content")
        return value if self.kind is EventKind.FINAL_RESPONSE and isinstance(value, str) else None

    def stop_reason(self) -> str | None:
        value = self.data.get("stop_reason")
        return (
            value
            if self.kind in {EventKind.RUN_STOPPED, EventKind.RUN_ERROR} and isinstance(value, str)
            else None
        )

    def usage(self) -> UsageRecord | None:
        value = self.data.get("usage_record")
        return (
            value
            if self.kind is EventKind.MODEL_USAGE_RECORDED and isinstance(value, UsageRecord)
            else None
        )

    def terminal_error(self) -> RunError | None:
        value = self.data.get("error")
        return cast("RunError", value) if self.kind is EventKind.RUN_ERROR else None

    def rendered_details(self) -> str:
        details = []
        for key, value in self.data.items():
            rendered = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else str(value)
            )
            details.append(f"{key}={rendered}")
        return " ".join(details)


@dataclass(frozen=True)
class TraceToolCall:
    call_id: str
    name: str
    raw_arguments: str

    @property
    def arguments(self) -> Mapping[str, object] | None:
        try:
            decoded = json.loads(self.raw_arguments)
        except json.JSONDecodeError:
            return None
        return MappingProxyType(decoded) if isinstance(decoded, dict) else None


@dataclass(frozen=True)
class ToolConfirmationRequest:
    run_id: str
    tool_call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class TraceToolResult:
    tool_call_id: str
    name: str
    success: bool
    result: str
    retryable: bool


@dataclass(frozen=True)
class Trace(Sequence[AgentEvent]):
    events: tuple[AgentEvent, ...] = ()

    def __iter__(self) -> Iterator[AgentEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def kinds(self) -> tuple[EventKind, ...]:
        return tuple(event.kind for event in self.events)

    def tool_calls(self) -> tuple[TraceToolCall, ...]:
        return tuple(call for event in self.events if (call := event.tool_call()) is not None)

    def usage(self) -> tuple[UsageRecord, ...]:
        return tuple(usage for event in self.events if (usage := event.usage()) is not None)

    def terminal_error(self) -> RunError | None:
        return next(
            (
                error
                for event in reversed(self.events)
                if (error := event.terminal_error()) is not None
            ),
            None,
        )

    def steering_count(self) -> int:
        return sum(event.kind is EventKind.STEERING_MESSAGE_ACCEPTED for event in self.events)

    def rendered_lines(self) -> tuple[str, ...]:
        lines = []
        for event in self.events:
            details = event.rendered_details()
            suffix = f" {details}" if details else ""
            lines.append(f"  {event.sequence} {event.kind}{suffix}")
        return tuple(lines)


class AgentEventListener(Protocol):
    def __call__(self, event: AgentEvent) -> None: ...
