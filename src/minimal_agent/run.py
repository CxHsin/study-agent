"""Run outcomes and cooperative control shared by Agent and AgentLoop."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from minimal_agent.events import AgentEvent


class StopReason(StrEnum):
    FINAL = "final"
    MAX_STEPS = "max_steps"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    ERROR = "error"


class RunPhase(StrEnum):
    READY = "ready"
    MODEL = "model"
    TOOL = "tool"
    TERMINAL = "terminal"


class LoopStateMachine:
    """Enforce in-process Agent Loop phase transitions."""

    _allowed: ClassVar[dict[RunPhase, frozenset[RunPhase]]] = {
        RunPhase.READY: frozenset((RunPhase.MODEL, RunPhase.TERMINAL)),
        RunPhase.MODEL: frozenset((RunPhase.TOOL, RunPhase.TERMINAL)),
        RunPhase.TOOL: frozenset((RunPhase.MODEL, RunPhase.TERMINAL)),
        RunPhase.TERMINAL: frozenset(),
    }

    def __init__(self) -> None:
        self._phase = RunPhase.READY
        self._step = 0

    @property
    def phase(self) -> RunPhase:
        return self._phase

    @property
    def step(self) -> int:
        return self._step

    def transition(self, phase: RunPhase, *, step: int | None = None) -> None:
        if phase not in self._allowed[self._phase]:
            raise RuntimeError(f"Invalid Run transition: {self._phase.value} -> {phase.value}")
        if step is not None:
            if step < self._step or (phase is RunPhase.MODEL and step <= self._step):
                raise RuntimeError("Run steps must advance at model boundaries.")
            self._step = step
        self._phase = phase


@dataclass(frozen=True)
class RunError:
    code: str
    message: str
    error_type: str
    step: int
    tool_call_id: str | None = None
    retryable: bool = False
    status_code: int | None = None
    provider_request_id: str | None = None
    retry_after: float | None = None


@dataclass(frozen=True)
class RunResult:
    final_response: str | None
    stop_reason: StopReason
    steps_used: int
    run_id: str
    error: RunError | None = None
    events: tuple[AgentEvent, ...] = ()
    context_metadata: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class LoopOutcome:
    final_response: str | None
    stop_reason: StopReason
    steps_used: int
    error: RunError | None = None
    context_metadata: tuple[dict[str, object], ...] = ()


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
