import pytest

from minimal_agent.run import LoopStateMachine, RunPhase


def test_loop_state_machine_follows_model_tool_lifecycle() -> None:
    state = LoopStateMachine()

    state.transition(RunPhase.MODEL, step=1)
    state.transition(RunPhase.TOOL)
    state.transition(RunPhase.MODEL, step=2)
    state.transition(RunPhase.TERMINAL)

    assert state.phase is RunPhase.TERMINAL
    assert state.step == 2


def test_loop_state_machine_rejects_invalid_or_repeated_transitions() -> None:
    state = LoopStateMachine()

    with pytest.raises(RuntimeError, match="ready -> tool"):
        state.transition(RunPhase.TOOL)

    state.transition(RunPhase.MODEL, step=1)
    with pytest.raises(RuntimeError, match="model -> model"):
        state.transition(RunPhase.MODEL, step=2)
