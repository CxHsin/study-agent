import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from minimal_agent.events import AgentEvent, AgentEventListener, EventKind
from minimal_agent.protocol import ModelAdapter, ModelError, ToolCall
from minimal_agent.session import AgentSession
from minimal_agent.tools import ToolRegistry


class StopReason(StrEnum):
    FINAL = "final"
    MAX_STEPS = "max_steps"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class RunError:
    code: str
    message: str
    error_type: str
    step: int
    tool_call_id: str | None = None


@dataclass(frozen=True)
class RunResult:
    final_response: str | None
    stop_reason: StopReason
    steps_used: int
    run_id: str
    error: RunError | None = None
    events: tuple[AgentEvent, ...] = ()


class RunControl:
    def __init__(self) -> None:
        self._stop_reason: StopReason | None = None

    def abort(self) -> None:
        self._stop_reason = StopReason.ABORTED

    def cancel(self) -> None:
        if self._stop_reason is None:
            self._stop_reason = StopReason.CANCELLED

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason


class AgentCore:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry | None = None,
        session: AgentSession | None = None,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._session = session or AgentSession()
        self._max_steps = max_steps
        self._listeners: list[AgentEventListener] = []
        self._sequence = 0

    def subscribe(self, listener: AgentEventListener) -> None:
        self._listeners.append(listener)

    @property
    def session(self) -> AgentSession:
        return self._session

    def prompt(self, user_input: str, control: RunControl | None = None) -> RunResult:
        run_id = str(uuid4())
        control = control or RunControl()
        self._sequence = 0
        trace: list[AgentEvent] = []
        started_at = time.perf_counter()
        disabled_listeners: set[int] = set()
        self._session.append({"role": "user", "content": user_input})
        emit = lambda kind, data: self._emit(
            run_id, kind, data, trace, started_at, disabled_listeners
        )
        emit(
            EventKind.RUN_STARTED,
            {
                "query": user_input,
                "max_steps": self._max_steps,
                "message_count": len(self._session.messages),
            },
        )
        last_fingerprint: str | None = None
        pending_repeat: str | None = None
        try:
            for step in range(1, self._max_steps + 1):
                stop = _control_stop(control)
                if stop:
                    return self._result(run_id, stop, step - 1, trace, emit)
                emit(
                    EventKind.MODEL_CALL_STARTED,
                    {"step": step, "message_count": len(self._session.messages)},
                )
                try:
                    response = self._model.complete(self._session.messages)
                except ModelError as error:
                    return self._error(
                        run_id,
                        "MODEL_ERROR",
                        str(error) or "Model request failed.",
                        "model_error",
                        step,
                        trace=trace,
                        emit=emit,
                    )
                emit(
                    EventKind.MODEL_RESPONSE,
                    {
                        "step": step,
                        "content": response.content,
                        "tool_calls": tuple(_tool_data(call) for call in response.tool_calls),
                    },
                )
                stop = _control_stop(control)
                if stop:
                    return self._result(run_id, stop, step, trace, emit)
                if not response.tool_calls:
                    if not response.content:
                        return self._error(
                            run_id,
                            "INVALID_MODEL_RESPONSE",
                            "Model returned no final content.",
                            "model_error",
                            step,
                            trace=trace,
                            emit=emit,
                        )
                    self._session.append({"role": "assistant", "content": response.content})
                    emit(EventKind.FINAL_RESPONSE, {"step": step, "content": response.content})
                    return self._finish(
                        RunResult(response.content, StopReason.FINAL, step, run_id), trace
                    )

                self._session.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": response.tool_calls,
                    }
                )
                fingerprints = [_fingerprint(call) for call in response.tool_calls]
                if pending_repeat is not None and pending_repeat in fingerprints:
                    return self._error(
                        run_id,
                        "REPEATED_TOOL_CALL",
                        "Model repeated a blocked tool call.",
                        "control_error",
                        step,
                        StopReason.REPEATED_TOOL_CALL,
                        trace=trace,
                        emit=emit,
                    )
                for tool_call in response.tool_calls:
                    stop = _control_stop(control)
                    if stop:
                        return self._result(run_id, stop, step, trace, emit)
                    fingerprint = _fingerprint(tool_call)
                    if fingerprint == last_fingerprint:
                        result = json.dumps(
                            {
                                "ok": False,
                                "error": {
                                    "code": "REPEATED_TOOL_CALL",
                                    "message": "Repeated tool call blocked.",
                                },
                            }
                        )
                        self._session.append(
                            {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                        )
                        emit(EventKind.TOOL_RESULT_PRODUCED, _tool_result_data(tool_call, result))
                        pending_repeat = fingerprint
                        last_fingerprint = fingerprint
                        continue
                    pending_repeat = None
                    last_fingerprint = fingerprint
                    emit(EventKind.TOOL_CALL_REQUESTED, _tool_data(tool_call))
                    result = self._tools.execute(tool_call)
                    self._session.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    )
                    emit(EventKind.TOOL_RESULT_PRODUCED, _tool_result_data(tool_call, result))
                    stop = _control_stop(control)
                    if stop:
                        return self._result(run_id, stop, step, trace, emit)
            return self._result(run_id, StopReason.MAX_STEPS, self._max_steps, trace, emit)
        except Exception as error:  # noqa: BLE001 - run boundary returns structured errors
            return self._error(
                run_id,
                "INTERNAL_ERROR",
                str(error) or "Internal error.",
                "internal_error",
                locals().get("step", 0),
                trace=trace,
                emit=emit,
            )

    def _finish(self, result: RunResult, trace: list[AgentEvent]) -> RunResult:
        return RunResult(
            result.final_response,
            result.stop_reason,
            result.steps_used,
            result.run_id,
            result.error,
            tuple(trace),
        )

    def _result(
        self,
        run_id: str,
        reason: StopReason,
        steps: int,
        trace: list[AgentEvent],
        emit,
    ) -> RunResult:
        emit(EventKind.RUN_STOPPED, {"stop_reason": reason.value, "steps_used": steps})
        return self._finish(RunResult(None, reason, steps, run_id), trace)

    def _error(
        self,
        run_id: str,
        code: str,
        message: str,
        error_type: str,
        step: int,
        reason: StopReason = StopReason.ERROR,
        *,
        trace: list[AgentEvent],
        emit,
    ) -> RunResult:
        error = RunError(code, message, error_type, step)
        emit(
            EventKind.RUN_ERROR,
            {"stop_reason": reason.value, "steps_used": step, "error": error},
        )
        return self._finish(RunResult(None, reason, step, run_id, error), trace)

    def _emit(
        self,
        run_id: str,
        kind: EventKind,
        data: dict[str, object],
        trace: list[AgentEvent],
        started_at: float,
        disabled_listeners: set[int],
    ) -> None:
        self._sequence += 1
        event = AgentEvent(
            run_id,
            kind,
            data,
            self._sequence,
            datetime.now(UTC),
            (time.perf_counter() - started_at) * 1000,
        )
        trace.append(event)
        for index, listener in enumerate(self._listeners):
            if index in disabled_listeners:
                continue
            try:
                listener(event)
            except Exception as error:  # noqa: BLE001 - observer failures are isolated
                disabled_listeners.add(index)
                self._emit(
                    run_id,
                    EventKind.LISTENER_ERROR,
                    {"listener": type(listener).__name__, "message": str(error)},
                    trace,
                    started_at,
                    disabled_listeners,
                )


def _tool_data(tool_call: ToolCall) -> dict[str, object]:
    return {"tool_call_id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}


def _tool_result_data(tool_call: ToolCall, result: str) -> dict[str, object]:
    success: bool | None = None
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and isinstance(parsed.get("ok"), bool):
            success = parsed["ok"]
    except json.JSONDecodeError:
        pass
    return {
        "tool_call_id": tool_call.id,
        "name": tool_call.name,
        "result": result,
        "success": success,
    }


def _control_stop(control: RunControl) -> StopReason | None:
    return control.stop_reason


def _fingerprint(tool_call: ToolCall) -> str:
    try:
        arguments = json.dumps(
            json.loads(tool_call.arguments), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, json.JSONDecodeError):
        arguments = tool_call.arguments.strip()
    return f"{tool_call.name}\0{arguments}"
