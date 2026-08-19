from types import SimpleNamespace

from minimal_agent.protocol import (
    AssistantMessage,
    ContextSummaryMessage,
    ModelRequest,
    RequestOptions,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolResultMessage,
    UserMessage,
)
from minimal_agent.provider_client import ProviderClient, aggregate_stream
from minimal_agent.providers import (
    AnthropicAdapter,
    AnthropicMessageCodec,
    OpenAIAdapter,
    OpenAIMessageCodec,
)


class FakeOpenAI:
    def __init__(self, response) -> None:
        self.requests: list[dict[str, object]] = []

        def create(**kwargs):
            self.requests.append(kwargs)
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


class FakeAnthropic:
    def __init__(self, response) -> None:
        self.requests: list[dict[str, object]] = []

        def create(**kwargs):
            self.requests.append(kwargs)
            return response

        self.messages = SimpleNamespace(create=create)


def test_openai_client_maps_text_and_tool_calls() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1", function=SimpleNamespace(name="read", arguments="{}")
                        )
                    ],
                )
            )
        ],
        usage=None,
    )
    sdk = FakeOpenAI(response)
    client = ProviderClient(OpenAIAdapter("key", client=sdk))

    result = client.complete(ModelRequest((UserMessage("read"),)))

    assert result.tool_calls == (ToolCall("c1", "read", "{}"),)
    assert client.profile.tool_calls is True
    assert sdk.requests[0]["stream"] is True


def test_anthropic_client_maps_text_tool_use_and_usage() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="done"),
            SimpleNamespace(type="tool_use", id="c1", name="read", input={"path": "x"}),
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=2),
    )
    client = ProviderClient(AnthropicAdapter("key", client=FakeAnthropic(response)))

    result = client.complete(ModelRequest((UserMessage("read"),)))

    assert result.content == "done"
    assert result.tool_calls[0].arguments == '{"path":"x"}'
    assert result.usage is not None
    assert result.usage.input_tokens == 5


def test_openai_codec_normalizes_complete_tool_history_and_summary() -> None:
    result = ToolResult("c1", "read", True, data={"ok": True})
    request = ModelRequest(
        (
            SystemMessage("system"),
            ContextSummaryMessage("facts", "summary-1"),
            AssistantMessage(None, (ToolCall("c1", "read", '{"path":"x"}'),)),
            ToolResultMessage(result),
        )
    )

    payload = OpenAIMessageCodec().request(request)

    assert payload["messages"][1]["content"].startswith("[Context summary")
    assert payload["messages"][2]["tool_calls"][0]["function"]["name"] == "read"
    assert payload["messages"][3]["role"] == "tool"


def test_anthropic_codec_combines_system_messages_and_structures_tools() -> None:
    tool = ToolDefinition(
        "read",
        "Read a file",
        {"type": "object", "properties": {}, "additionalProperties": False},
    )
    request = ModelRequest(
        (SystemMessage("first"), SystemMessage("second"), UserMessage("read")),
        RequestOptions(tools=(tool,)),
    )

    payload = AnthropicMessageCodec().request(request)

    assert payload["system"] == "first\n\nsecond"
    assert payload["tools"][0]["input_schema"]["type"] == "object"
    assert payload["tool_choice"] == {"type": "auto"}


def test_openai_codec_aggregates_streamed_tool_call_fragments() -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="c1",
                                function=SimpleNamespace(name="read", arguments='{"path"'),
                            )
                        ],
                    )
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(name=None, arguments=':"x"}'),
                            )
                        ],
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        ),
    ]

    response = aggregate_stream(OpenAIMessageCodec().events(iter(chunks)))

    assert response.tool_calls == (ToolCall("c1", "read", '{"path":"x"}'),)
    assert response.usage is not None
    assert response.usage.output_tokens == 2


def test_tool_choice_none_omits_tools_from_both_provider_payloads() -> None:
    tool = ToolDefinition("read", "Read", {"type": "object"})
    request = ModelRequest(
        (UserMessage("summarize"),),
        RequestOptions(tools=(tool,), tool_choice="none"),
    )

    assert "tools" not in OpenAIMessageCodec().request(request)
    assert "tools" not in AnthropicMessageCodec().request(request)


def test_unknown_model_requires_an_explicit_profile() -> None:
    try:
        OpenAIAdapter("key", model="future-model", client=object())
    except ValueError as error:
        assert "explicit ModelProfile" in str(error)
    else:
        raise AssertionError("Expected an unknown model to require a profile")


def test_anthropic_codec_groups_parallel_tool_results() -> None:
    request = ModelRequest(
        (
            AssistantMessage(
                None,
                (ToolCall("c1", "first", "{}"), ToolCall("c2", "second", "{}")),
            ),
            ToolResultMessage(ToolResult("c1", "first", True, data=1)),
            ToolResultMessage(ToolResult("c2", "second", True, data=2)),
        )
    )

    payload = AnthropicMessageCodec().request(request)

    assert len(payload["messages"]) == 2
    assert [block["tool_use_id"] for block in payload["messages"][1]["content"]] == [
        "c1",
        "c2",
    ]
