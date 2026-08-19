from types import SimpleNamespace

from minimal_agent.protocol import ToolCall
from minimal_agent.providers import AnthropicAdapter, OpenAIAdapter


class FakeOpenAI:
    def __init__(self, response) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))


class FakeAnthropic:
    def __init__(self, response) -> None:
        self.messages = SimpleNamespace(create=lambda **kwargs: response)


def test_openai_adapter_maps_text_and_tool_calls() -> None:
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
    adapter = OpenAIAdapter("key", [], client=FakeOpenAI(response))

    result = adapter.complete([{"role": "user", "content": "read"}])

    assert result.tool_calls == (ToolCall("c1", "read", "{}"),)
    assert adapter.capabilities().tool_calls is True


def test_anthropic_adapter_maps_text_and_tool_use() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="done"),
            SimpleNamespace(type="tool_use", id="c1", name="read", input={"path": "x"}),
        ]
    )
    adapter = AnthropicAdapter("key", [], client=FakeAnthropic(response))

    result = adapter.complete([{"role": "user", "content": "read"}])

    assert result.content == "done"
    assert result.tool_calls[0].arguments == '{"path":"x"}'
