from collections.abc import Sequence

from minimal_agent.cli import ConsoleConfirmation, run_console
from minimal_agent.core import AgentCore
from minimal_agent.protocol import ChatMessage, ModelResponse, ModelStreamChunk, ToolCall
from minimal_agent.tools import ToolDefinition, ToolRegistry


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


def test_console_renders_streamed_text_and_tool_progress() -> None:
    class StreamingModel:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, messages):
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamChunk(
                    tool_call_id="call", tool_name="echo", arguments_delta='{"value":"ok"}'
                )
                yield ModelStreamChunk(done=True)
            else:
                yield ModelStreamChunk(content_delta="done", done=True)

    tools = ToolRegistry(
        (
            ToolDefinition(
                "echo",
                "Echo",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                lambda arguments: arguments,
            ),
        )
    )
    outputs: list[str] = []
    inputs = iter(["hello", "/exit"])

    run_console(
        AgentCore(StreamingModel(), tools),
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert 'Tool> echo {"value":"ok"}' in outputs
    assert "Tool> echo ok" in outputs
    assert "Agent> done" in outputs


def test_console_confirmation_requires_explicit_yes() -> None:
    outputs: list[str] = []
    confirmation = ConsoleConfirmation(input_fn=lambda _prompt: "yes", output_fn=outputs.append)
    context = type(
        "Context",
        (),
        {"arguments": {"path": "file.txt"}, "tool_call": ToolCall("call", "write_file", "{}")},
    )()

    assert confirmation.confirm(context) is True
    assert outputs == ['Confirm> write_file {"path":"file.txt"}']


def test_console_coordinates_confirmation_on_the_input_thread() -> None:
    class ToolModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(tool_calls=(ToolCall("call", "change", "{}"),))
            return ModelResponse(content="done")

    changed: list[bool] = []
    confirmation = ConsoleConfirmation(coordinated=True)
    tools = ToolRegistry(
        (
            ToolDefinition(
                "change",
                "Change",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: changed.append(True),
                requires_confirmation=True,
            ),
        ),
        confirmation=confirmation,
    )
    inputs = iter(["change it", "yes", "/exit"])

    run_console(
        AgentCore(ToolModel(), tools),
        input_fn=lambda _prompt: next(inputs),
        output_fn=lambda _text: None,
        confirmation=confirmation,
    )

    assert changed == [True]


def test_malformed_tool_arguments_cannot_leave_stale_confirmation() -> None:
    class MalformedModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(tool_calls=(ToolCall("call", "change", "not-json"),))
            if self.calls == 3:
                return ModelResponse(tool_calls=(ToolCall("call", "change", "{}"),))
            return ModelResponse(content="recovered")

    confirmation = ConsoleConfirmation(coordinated=True)
    changed: list[bool] = []
    tools = ToolRegistry(
        (
            ToolDefinition(
                "change",
                "Change",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: changed.append(True),
                requires_confirmation=True,
            ),
        ),
        confirmation=confirmation,
    )
    inputs = iter(["malformed", "/reset", "valid", "yes", "/exit"])
    outputs: list[str] = []

    run_console(
        AgentCore(MalformedModel(), tools),
        input_fn=lambda _prompt: next(inputs),
        output_fn=outputs.append,
        confirmation=confirmation,
    )

    assert changed == [True]
    assert sum(output.startswith("Confirm>") for output in outputs) == 1


def test_console_interrupt_cancels_only_the_current_run() -> None:
    class ToolModel:
        def complete(self, _messages):
            return ModelResponse(tool_calls=(ToolCall("call", "change", "{}"),))

    confirmation = ConsoleConfirmation(coordinated=True)
    changed: list[bool] = []
    tools = ToolRegistry(
        (
            ToolDefinition(
                "change",
                "Change",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: changed.append(True),
                requires_confirmation=True,
            ),
        ),
        confirmation=confirmation,
    )
    answers = iter(("stop this", KeyboardInterrupt(), "/exit"))

    def input_fn(_prompt: str) -> str:
        answer = next(answers)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    outputs: list[str] = []

    exit_code = run_console(
        AgentCore(ToolModel(), tools),
        input_fn=input_fn,
        output_fn=outputs.append,
        confirmation=confirmation,
    )

    assert exit_code == 0
    assert changed == []
    assert outputs[-1] == "Agent> Run cancelled."
