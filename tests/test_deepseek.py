from types import SimpleNamespace

from minimal_agent.deepseek import DeepSeekAdapter
from minimal_agent.protocol import (
    AssistantMessage,
    ModelRequest,
    ProviderError,
    ProviderErrorKind,
    RequestOptions,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from minimal_agent.provider_client import ProviderClient


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


def test_adapter_sends_deepseek_typed_tool_call_request() -> None:
    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Done.", tool_calls=None))],
        usage=None,
    )
    completions = FakeCompletions(response=completion)
    adapter = DeepSeekAdapter(api_key="test-key", client=FakeClient(completions))
    tool = ToolDefinition(
        "read_file",
        "Read text",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    messages = (
        UserMessage("Read notes.txt"),
        AssistantMessage(
            None,
            (ToolCall(id="call-1", name="read_file", arguments='{"path":"notes.txt"}'),),
        ),
        ToolResultMessage(ToolResult("call-1", "read_file", True, data={"ok": True})),
    )

    result = ProviderClient(adapter).complete(ModelRequest(messages, RequestOptions(tools=(tool,))))

    assert result.content == "Done."
    request = completions.requests[0]
    assert request["model"] == "deepseek-v4-flash"
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["tool_choice"] == "auto"
    assert request["messages"][1]["tool_calls"][0]["function"] == {
        "name": "read_file",
        "arguments": '{"path":"notes.txt"}',
    }


def test_adapter_converts_deepseek_tool_calls() -> None:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            function=SimpleNamespace(name="read_file", arguments='{"path":"x"}'),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )
    client = ProviderClient(
        DeepSeekAdapter(api_key="test-key", client=FakeClient(FakeCompletions(response=completion)))
    )

    result = client.complete(ModelRequest((UserMessage("Read x"),)))

    assert result.tool_calls == (ToolCall(id="call-1", name="read_file", arguments='{"path":"x"}'),)


def test_deepseek_profile_declares_model_capabilities() -> None:
    client = ProviderClient(DeepSeekAdapter(api_key="test-key", client=object()))

    assert client.profile.tool_calls is True
    assert client.profile.streaming is True
    assert client.profile.prompt_cache is True
    assert client.profile.context_window_tokens == 128_000


def test_sdk_errors_are_classified() -> None:
    class RateLimitError(Exception):
        status_code = 429
        request_id = "request-1"

    client = ProviderClient(
        DeepSeekAdapter(
            api_key="test-key", client=FakeClient(FakeCompletions(error=RateLimitError()))
        ),
        retry_policy=None,
        sleep=lambda _delay: None,
    )

    try:
        client.complete(ModelRequest((UserMessage("List files"),)))
    except ProviderError as error:
        assert error.kind is ProviderErrorKind.RATE_LIMIT
        assert error.retryable is True
        assert error.request_id == "request-1"
    else:
        raise AssertionError("Expected ProviderError")
