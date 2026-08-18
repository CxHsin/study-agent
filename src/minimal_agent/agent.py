import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

type ChatMessage = dict[str, object]

MAX_AGENT_STEPS = 8
TRACE_SCHEMA_VERSION = 1


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    data: dict[str, object]
    run_id: str
    sequence: int
    timestamp: str
    duration_ms: float | None = None
    schema_version: int = TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("Trace event kind must not be empty.")
        if not self.run_id:
            raise ValueError("Trace event run_id must not be empty.")
        if self.sequence < 1:
            raise ValueError("Trace event sequence must be positive.")
        if self.schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported Trace schema version: {self.schema_version}")


class TraceSink(Protocol):
    def __call__(self, event: TraceEvent) -> None: ...


class MessageStore(Protocol):
    def create_session(self) -> str: ...

    def append_message(
        self,
        session_id: str,
        message: ChatMessage,
        *,
        run_id: str,
    ) -> None: ...

    def load_messages(self, session_id: str) -> list[ChatMessage]: ...


class TaskStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StopReason(StrEnum):
    FINAL_RESPONSE = "final_response"
    MAX_STEPS = "max_steps"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    MODEL_ERROR = "model_error"
    USER_CANCELLED = "user_cancelled"
    TOOL_ERROR = "tool_error"


@dataclass(frozen=True)
class TaskResult:
    status: TaskStatus
    stop_reason: StopReason
    final_response: str | None
    steps_used: int


@dataclass(frozen=True)
class ModelResponse:
    content: str | None = None
    tool_calls: tuple["ToolCall", ...] = ()


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


class ToolExecutor(Protocol):
    def execute(self, tool_call: ToolCall) -> str: ...


class ModelAdapter(Protocol):
    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse: ...


class AgentSession:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolExecutor | None = None,
        trace_sink: TraceSink | None = None,
        message_store: MessageStore | None = None,
        session_id: str | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._trace_sink = trace_sink
        self._message_store = message_store
        self._session_id = (
            message_store.create_session()
            if message_store is not None and session_id is None
            else session_id
        )
        self._messages = (
            message_store.load_messages(self._session_id)
            if message_store is not None and self._session_id is not None
            else []
        )
        self._trace_run_id = ""
        self._trace_sequence = 0

    def submit(self, user_input: str) -> TaskResult:
        self._trace_run_id = str(uuid4())
        self._trace_sequence = 0
        self._append_message({"role": "user", "content": user_input})
        steps_used = 0
        previous_tool_batch: tuple[tuple[str, str], ...] | None = None

        while True:
            self._emit_trace("agent_step", {"step": steps_used + 1})
            model_started = time.perf_counter()
            try:
                response = self._model.complete(self._messages)
            except KeyboardInterrupt:
                self._emit_task_end(TaskStatus.CANCELLED, StopReason.USER_CANCELLED, steps_used)
                raise
            except ModelError:
                self._emit_trace(
                    "model_call",
                    {"status": "error"},
                    duration_ms=_elapsed_ms(model_started),
                )
                self._emit_task_end(TaskStatus.FAILED, StopReason.MODEL_ERROR, steps_used)
                return TaskResult(
                    status=TaskStatus.FAILED,
                    stop_reason=StopReason.MODEL_ERROR,
                    final_response=None,
                    steps_used=steps_used,
                )
            self._emit_trace(
                "model_call",
                {"status": "ok", "tool_calls": len(response.tool_calls)},
                duration_ms=_elapsed_ms(model_started),
            )
            steps_used += 1

            if not response.tool_calls:
                self._append_message({"role": "assistant", "content": response.content})
                self._emit_task_end(TaskStatus.COMPLETED, StopReason.FINAL_RESPONSE, steps_used)
                return TaskResult(
                    status=TaskStatus.COMPLETED,
                    stop_reason=StopReason.FINAL_RESPONSE,
                    final_response=response.content,
                    steps_used=steps_used,
                )

            tool_batch = _tool_batch_fingerprint(response.tool_calls)
            if tool_batch == previous_tool_batch:
                self._emit_task_end(TaskStatus.FAILED, StopReason.REPEATED_TOOL_CALL, steps_used)
                return TaskResult(
                    status=TaskStatus.FAILED,
                    stop_reason=StopReason.REPEATED_TOOL_CALL,
                    final_response=None,
                    steps_used=steps_used,
                )
            if steps_used >= MAX_AGENT_STEPS:
                self._emit_task_end(TaskStatus.FAILED, StopReason.MAX_STEPS, steps_used)
                return TaskResult(
                    status=TaskStatus.FAILED,
                    stop_reason=StopReason.MAX_STEPS,
                    final_response=None,
                    steps_used=steps_used,
                )

            previous_tool_batch = tool_batch
            self._append_message(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            )
            if self._tools is None:
                self._emit_task_end(TaskStatus.FAILED, StopReason.TOOL_ERROR, steps_used)
                return TaskResult(
                    status=TaskStatus.FAILED,
                    stop_reason=StopReason.TOOL_ERROR,
                    final_response=None,
                    steps_used=steps_used,
                )

            for tool_call in response.tool_calls:
                self._emit_trace(
                    "tool_call",
                    {
                        "name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "arguments": _safe_tool_arguments(tool_call.arguments),
                    },
                )
                tool_started = time.perf_counter()
                try:
                    tool_result = self._tools.execute(tool_call)
                except KeyboardInterrupt:
                    self._emit_task_end(TaskStatus.CANCELLED, StopReason.USER_CANCELLED, steps_used)
                    raise
                except Exception:  # noqa: BLE001 - tool failures must not escape the harness
                    self._emit_trace(
                        "tool_result",
                        {"tool_call_id": tool_call.id, "status": "exception"},
                        duration_ms=_elapsed_ms(tool_started),
                    )
                    self._emit_task_end(TaskStatus.FAILED, StopReason.TOOL_ERROR, steps_used)
                    return TaskResult(
                        status=TaskStatus.FAILED,
                        stop_reason=StopReason.TOOL_ERROR,
                        final_response=None,
                        steps_used=steps_used,
                    )
                self._emit_trace(
                    "tool_result",
                    {
                        "tool_call_id": tool_call.id,
                        "status": _tool_result_status(tool_result),
                    },
                    duration_ms=_elapsed_ms(tool_started),
                )
                self._append_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

    def reset(self) -> None:
        self._messages.clear()
        if self._message_store is not None:
            self._session_id = self._message_store.create_session()

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _append_message(self, message: ChatMessage) -> None:
        self._messages.append(message)
        if self._message_store is not None and self._session_id is not None:
            self._message_store.append_message(
                self._session_id,
                message,
                run_id=self._trace_run_id,
            )

    def _emit_trace(
        self,
        kind: str,
        data: dict[str, object],
        duration_ms: float | None = None,
    ) -> None:
        if self._trace_sink is None:
            return
        self._trace_sequence += 1
        self._trace_sink(
            TraceEvent(
                kind=kind,
                data=data,
                run_id=self._trace_run_id,
                sequence=self._trace_sequence,
                timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                duration_ms=duration_ms,
            )
        )

    def _emit_task_end(self, status: TaskStatus, stop_reason: StopReason, steps_used: int) -> None:
        self._emit_trace(
            "task_end",
            {
                "status": status.value,
                "stop_reason": stop_reason.value,
                "steps_used": steps_used,
            },
        )


def _tool_batch_fingerprint(tool_calls: Sequence[ToolCall]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (tool_call.name, _canonical_arguments(tool_call.arguments)) for tool_call in tool_calls
    )


def _canonical_arguments(arguments: str) -> str:
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    return json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _safe_tool_arguments(arguments: str) -> dict[str, object] | str:
    try:
        decoded = json.loads(arguments)
    except json.JSONDecodeError:
        return "<invalid-json>"
    if not isinstance(decoded, dict):
        return "<non-object>"
    safe: dict[str, object] = {}
    for key, value in decoded.items():
        if key == "path" and isinstance(value, str):
            normalized = value.replace("\\", "/")
            safe[key] = "<absolute-path>" if _is_absolute_path(normalized) else normalized
        elif key != "content":
            safe[key] = value
    return safe


def _tool_result_status(tool_result: str) -> str:
    try:
        decoded = json.loads(tool_result)
    except json.JSONDecodeError:
        return "invalid-result"
    return "ok" if decoded.get("ok") is True else str(decoded.get("error", {}).get("code", "error"))


def _is_absolute_path(path: str) -> bool:
    return path.startswith("/") or (len(path) >= 3 and path[1] == ":" and path[2] == "/")


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
