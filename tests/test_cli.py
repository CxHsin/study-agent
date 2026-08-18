from collections.abc import Sequence

from minimal_agent.cli import run_console
from minimal_agent.core import AgentCore
from minimal_agent.protocol import ChatMessage, ModelResponse


class FinalModel:
    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return ModelResponse(content=f"Answer: {messages[-1]['content']}")


def test_console_submits_tasks_resets_and_exits() -> None:
    inputs = iter(["hello", "/reset", "", "/exit"])
    outputs: list[str] = []
    core = AgentCore(FinalModel())

    exit_code = run_console(
        core,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert exit_code == 0
    assert core.session.messages == []
    assert "Agent> Answer: hello" in outputs
    assert "Session reset." in outputs


def test_console_displays_last_core_events() -> None:
    inputs = iter(["hello", "/trace last", "/exit"])
    outputs: list[str] = []

    run_console(
        AgentCore(FinalModel()),
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert "Trace> Last run:" in outputs
    assert any("final_response" in output for output in outputs)
