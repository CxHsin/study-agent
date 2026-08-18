import pytest

from minimal_agent.runtime import (
    InvalidTransitionError,
    RunEvent,
    RunState,
    RunStateMachine,
)


def test_run_state_machine_accepts_the_normal_tool_path() -> None:
    machine = RunStateMachine()

    machine = machine.transition(RunEvent.RUN_STARTED)
    machine = machine.transition(RunEvent.MODEL_TOOL_CALLS_PERSISTED)
    machine = machine.transition(RunEvent.ALL_TOOLS_SUCCEEDED)
    machine = machine.transition(RunEvent.MODEL_FINAL_RESPONSE_PERSISTED)

    assert machine.state is RunState.COMPLETED


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (RunEvent.UNCERTAIN_NON_IDEMPOTENT_TOOL, RunState.BLOCKED),
        (RunEvent.TOOL_EXECUTION_INTERRUPTED, RunState.RECOVERABLE),
        (RunEvent.USER_CANCELLED, RunState.CANCELLED),
        (RunEvent.BUDGET_EXHAUSTED, RunState.FAILED),
    ],
)
def test_waiting_for_tool_records_distinct_stop_conditions(
    event: RunEvent, expected: RunState
) -> None:
    machine = RunStateMachine(RunState.WAITING_FOR_TOOL)

    assert machine.transition(event).state is expected


def test_invalid_transition_is_explicit() -> None:
    with pytest.raises(InvalidTransitionError, match="run_started"):
        RunStateMachine(RunState.COMPLETED).transition(RunEvent.RUN_STARTED)
