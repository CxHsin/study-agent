from types import SimpleNamespace

from openai import OpenAIError

from minimal_agent.agent import AgentSession, StopReason, ToolCall
from minimal_agent.deepseek import DeepSeekAdapter
from minimal_agent.workspace_tools import WorkspaceTools


class FakeCompletions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, object]] = []

    def create(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_adapter_sends_deepseek_tool_call_request(tmp_path) -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Done.", tool_calls=None))]
    )
    completions = FakeCompletions(response=completion)
    tools = WorkspaceTools(tmp_path)
    adapter = DeepSeekAdapter(
        api_key="test-key",
        tool_definitions=tools.definitions(),
        client=FakeClient(completions),
    )
    messages = [
        {"role": "user", "content": "Read notes.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": (
                ToolCall(id="call-1", name="read_file", arguments='{"path":"notes.txt"}'),
            ),
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
    ]

    result = adapter.complete(messages)

    assert result.content == "Done."
    request = completions.requests[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "list_files",
        "read_file",
    ]
    assistant_message = request["messages"][1]
    assert assistant_message["tool_calls"][0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"notes.txt"}'},
    }


def test_adapter_converts_deepseek_tool_calls(tmp_path) -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(
                                name="list_files",
                                arguments='{"path":"."}',
                            ),
                        )
                    ],
                )
            )
        ]
    )
    tools = WorkspaceTools(tmp_path)
    adapter = DeepSeekAdapter(
        api_key="test-key",
        tool_definitions=tools.definitions(),
        client=FakeClient(FakeCompletions(response=completion)),
    )

    result = adapter.complete([{"role": "user", "content": "List files"}])

    assert result.tool_calls == (
        ToolCall(id="call-1", name="list_files", arguments='{"path":"."}'),
    )


def test_model_errors_become_failed_task_results(tmp_path) -> None:
    tools = WorkspaceTools(tmp_path)
    adapter = DeepSeekAdapter(
        api_key="test-key",
        tool_definitions=tools.definitions(),
        client=FakeClient(FakeCompletions(error=OpenAIError("network failed"))),
    )

    result = AgentSession(model=adapter, tools=tools).submit("List files")

    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.steps_used == 0
