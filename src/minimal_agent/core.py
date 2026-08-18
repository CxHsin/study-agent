import json
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from minimal_agent.events import AgentEvent, AgentEventListener
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
        self._session.append({"role": "user", "content": user_input})
        self._emit("run_started", {"run_id": run_id})
        last_fingerprint: str | None = None
        pending_repeat: str | None = None
        try:
            for step in range(1, self._max_steps + 1):
                stop = _control_stop(control)
                if stop:
                    return self._result(run_id, stop, step - 1)
                self._emit("model_call_started", {"step": step})
                try:
                    response = self._model.complete(self._session.messages)
                except ModelError as error:
                    return self._error(
                        run_id,
                        "MODEL_ERROR",
                        str(error) or "Model request failed.",
                        "model_error",
                        step,
                    )
                self._emit("model_response", {"step": step, "tool_calls": len(response.tool_calls)})
                stop = _control_stop(control)
                if stop:
                    return self._result(run_id, stop, step)
                if not response.tool_calls:
                    if not response.content:
                        return self._error(
                            run_id,
                            "INVALID_MODEL_RESPONSE",
                            "Model returned no final content.",
                            "model_error",
                            step,
                        )
                    self._session.append({"role": "assistant", "content": response.content})
                    self._emit("final_response", {"step": step})
                    return RunResult(response.content, StopReason.FINAL, step, run_id)

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
                    )
                for tool_call in response.tool_calls:
                    stop = _control_stop(control)
                    if stop:
                        return self._result(run_id, stop, step)
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
                        self._emit("tool_result_produced", {"tool_call_id": tool_call.id})
                        pending_repeat = fingerprint
                        last_fingerprint = fingerprint
                        continue
                    pending_repeat = None
                    last_fingerprint = fingerprint
                    self._emit("tool_call_requested", _tool_data(tool_call))
                    result = self._tools.execute(tool_call)
                    self._session.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                    )
                    self._emit("tool_result_produced", {"tool_call_id": tool_call.id})
                    stop = _control_stop(control)
                    if stop:
                        return self._result(run_id, stop, step)
            return self._result(run_id, StopReason.MAX_STEPS, self._max_steps)
        except Exception as error:  # noqa: BLE001 - run boundary returns structured errors
            return self._error(
                run_id,
                "INTERNAL_ERROR",
                str(error) or "Internal error.",
                "internal_error",
                locals().get("step", 0),
            )

    def _result(self, run_id: str, reason: StopReason, steps: int) -> RunResult:
        try:
            self._emit("run_failed", {"reason": reason.value})
        except Exception:  # noqa: BLE001 - listener failures must not replace the run result
            return RunResult(None, reason, steps, run_id)
        return RunResult(None, reason, steps, run_id)

    def _error(
        self,
        run_id: str,
        code: str,
        message: str,
        error_type: str,
        step: int,
        reason: StopReason = StopReason.ERROR,
    ) -> RunResult:
        error = RunError(code, message, error_type, step)
        try:
            self._emit("run_failed", {"reason": reason.value, "error": error})
        except Exception:  # noqa: BLE001 - listener failures must not replace the run result
            return RunResult(None, reason, step, run_id, error)
        return RunResult(None, reason, step, run_id, error)

    def _emit(self, kind: str, data: dict[str, object]) -> None:
        self._sequence += 1
        event = AgentEvent(kind, data, self._sequence)
        for listener in self._listeners:
            listener(event)


def _tool_data(tool_call: ToolCall) -> dict[str, object]:
    return {"tool_call_id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}


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
