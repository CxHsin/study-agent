import json
from collections.abc import Sequence

from minimal_agent.core import AgentCore, RunControl, StopReason
from minimal_agent.protocol import ChatMessage, ModelResponse, ToolCall
from minimal_agent.tools import ToolDefinition, ToolRegistry
from minimal_agent.workspace_tools import WorkspaceTools


class ReadThenAnswer:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(tool_calls=(ToolCall("call-1", "read", "{}"),))
        assert json.loads(str(messages[-1]["content"]))["ok"] is True
        return ModelResponse(content="done")


def test_core_runs_tool_loop_and_publishes_ordered_events() -> None:
    registry = ToolRegistry(
        [ToolDefinition("read", "Read a value", {"type": "object"}, lambda call: '{"ok":true}')]
    )
    core = AgentCore(ReadThenAnswer(), registry)
    events = []
    core.subscribe(events.append)

    result = core.prompt("read it")

    assert result.stop_reason is StopReason.FINAL
    assert result.final_response == "done"
    assert result.steps_used == 2
    assert result.run_id
    assert [event.kind for event in events] == [
        "run_started",
        "model_call_started",
        "model_response",
        "tool_call_requested",
        "tool_result_produced",
        "model_call_started",
        "model_response",
        "final_response",
    ]
    assert [event.sequence for event in events] == list(range(1, 9))


def test_registry_turns_unknown_tool_into_structured_result() -> None:
    result = ToolRegistry().execute(ToolCall("call-1", "missing", "{}"))

    assert json.loads(result)["error"]["code"] == "UNKNOWN_TOOL"


def test_core_reads_a_real_workspace_file(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("The answer is 42.", encoding="utf-8")

    class ReadFileThenAnswer:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    tool_calls=(ToolCall("call-1", "read_file", '{"path":"notes.txt"}'),)
                )
            assert "The answer is 42." in str(messages[-1]["content"])
            return ModelResponse(content="The file says 42.")

    result = AgentCore(ReadFileThenAnswer(), WorkspaceTools(tmp_path).registry()).prompt(
        "Read notes.txt."
    )

    assert result.stop_reason is StopReason.FINAL
    assert result.final_response == "The file says 42."


def test_max_steps_executes_last_tool_call() -> None:
    calls = []

    class AlwaysTool:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(
                tool_calls=(ToolCall(f"call-{len(calls)}", "read", f'{{"step": {len(calls)}}}'),)
            )

    registry = ToolRegistry(
        [
            ToolDefinition(
                "read", "Read", {"type": "object"}, lambda call: calls.append(call) or '{"ok":true}'
            )
        ]
    )
    result = AgentCore(AlwaysTool(), registry, max_steps=2).prompt("go")

    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps_used == 2
    assert len(calls) == 2


def test_repeated_tool_call_is_reported_then_observed() -> None:
    class RepeatsThenAnswers:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            self.calls += 1
            if self.calls <= 2:
                return ModelResponse(
                    tool_calls=(ToolCall(f"call-{self.calls}", "read", '{"x": 1}'),)
                )
            return ModelResponse(content="recovered")

    result = AgentCore(
        RepeatsThenAnswers(),
        ToolRegistry([ToolDefinition("read", "Read", {}, lambda call: '{"ok":true}')]),
    ).prompt("go")

    assert result.stop_reason is StopReason.FINAL
    assert result.final_response == "recovered"


def test_repeated_tool_call_stops_on_second_observation() -> None:
    class AlwaysRepeats:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(tool_calls=(ToolCall("call", "read", '{"x": 1}'),))

    result = AgentCore(
        AlwaysRepeats(),
        ToolRegistry([ToolDefinition("read", "Read", {}, lambda call: '{"ok":true}')]),
    ).prompt("go")

    assert result.stop_reason is StopReason.REPEATED_TOOL_CALL
    assert result.error is not None
    assert result.error.code == "REPEATED_TOOL_CALL"


def test_control_can_abort_before_first_model_call() -> None:
    class NeverCalled:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            raise AssertionError("model should not be called")

    control = RunControl()
    control.abort()
    result = AgentCore(NeverCalled()).prompt("stop", control)

    assert result.stop_reason is StopReason.ABORTED
    assert result.steps_used == 0


def test_model_error_becomes_structured_run_error() -> None:
    from minimal_agent.protocol import ModelError

    class Fails:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            raise ModelError("provider unavailable")

    result = AgentCore(Fails()).prompt("go")

    assert result.stop_reason is StopReason.ERROR
    assert result.error is not None
    assert result.error.code == "MODEL_ERROR"
