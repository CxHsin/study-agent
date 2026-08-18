import json
from collections.abc import Sequence

from minimal_agent.core import AgentCore, AgentResult, AgentStatus
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

    assert result == AgentResult(AgentStatus.COMPLETED, "done", 2)
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

    assert result.status is AgentStatus.COMPLETED
    assert result.final_response == "The file says 42."
