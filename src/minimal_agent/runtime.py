from dataclasses import dataclass
from enum import StrEnum


class RunState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    RECOVERABLE = "recoverable"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunEvent(StrEnum):
    RUN_STARTED = "run_started"
    MODEL_TOOL_CALLS_PERSISTED = "model_tool_calls_persisted"
    MODEL_FINAL_RESPONSE_PERSISTED = "model_final_response_persisted"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    ALL_TOOLS_SUCCEEDED = "all_tools_succeeded"
    RETRYABLE_TOOL_FAILURE = "retryable_tool_failure"
    UNCERTAIN_NON_IDEMPOTENT_TOOL = "uncertain_non_idempotent_tool"
    TOOL_EXECUTION_INTERRUPTED = "tool_execution_interrupted"
    USER_CANCELLED = "user_cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RESOLUTION_RECORDED = "resolution_recorded"


class InvalidTransitionError(ValueError):
    pass


@dataclass(frozen=True)
class RunStateMachine:
    state: RunState = RunState.PENDING

    def transition(self, event: RunEvent) -> "RunStateMachine":
        next_state = _NEXT_STATES.get((self.state, event))
        if next_state is None:
            raise InvalidTransitionError(
                f"Event {event.value!r} is invalid while Run is {self.state.value!r}."
            )
        return RunStateMachine(next_state)


_NEXT_STATES: dict[tuple[RunState, RunEvent], RunState] = {
    (RunState.PENDING, RunEvent.RUN_STARTED): RunState.RUNNING,
    (RunState.RUNNING, RunEvent.MODEL_TOOL_CALLS_PERSISTED): RunState.WAITING_FOR_TOOL,
    (RunState.RUNNING, RunEvent.MODEL_FINAL_RESPONSE_PERSISTED): RunState.COMPLETED,
    (RunState.RUNNING, RunEvent.MODEL_ERROR): RunState.FAILED,
    (RunState.RUNNING, RunEvent.TOOL_ERROR): RunState.FAILED,
    (RunState.RUNNING, RunEvent.REPEATED_TOOL_CALL): RunState.FAILED,
    (RunState.WAITING_FOR_TOOL, RunEvent.ALL_TOOLS_SUCCEEDED): RunState.RUNNING,
    (RunState.WAITING_FOR_TOOL, RunEvent.RETRYABLE_TOOL_FAILURE): RunState.WAITING_FOR_TOOL,
    (RunState.WAITING_FOR_TOOL, RunEvent.TOOL_ERROR): RunState.FAILED,
    (RunState.WAITING_FOR_TOOL, RunEvent.UNCERTAIN_NON_IDEMPOTENT_TOOL): RunState.BLOCKED,
    (RunState.WAITING_FOR_TOOL, RunEvent.TOOL_EXECUTION_INTERRUPTED): RunState.RECOVERABLE,
    (RunState.RUNNING, RunEvent.USER_CANCELLED): RunState.CANCELLED,
    (RunState.WAITING_FOR_TOOL, RunEvent.USER_CANCELLED): RunState.CANCELLED,
    (RunState.RUNNING, RunEvent.BUDGET_EXHAUSTED): RunState.FAILED,
    (RunState.WAITING_FOR_TOOL, RunEvent.BUDGET_EXHAUSTED): RunState.FAILED,
    (RunState.BLOCKED, RunEvent.RESOLUTION_RECORDED): RunState.WAITING_FOR_TOOL,
}
