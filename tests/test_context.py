from collections.abc import Sequence

from minimal_agent.context import ContextBuilder, ContextConfig, ContextError
from minimal_agent.core import AgentCore, StopReason
from minimal_agent.protocol import (
    ChatMessage,
    ContextSummaryMessage,
    ModelRequest,
    ModelResponse,
    SystemMessage,
    UserMessage,
)
from minimal_agent.providers import OpenAIMessageCodec
from minimal_agent.session import AgentSession


class EchoModel:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        self.messages = list(messages)
        return ModelResponse(content="ok")


class FixedSummarizer:
    def summarize(self, messages: Sequence[ChatMessage]) -> str:
        return "Earlier goals and facts were retained."


def test_context_builder_keeps_system_prompt_and_compresses_old_history() -> None:
    session = AgentSession(
        [
            {"role": "user", "content": "old " * 40},
            {"role": "assistant", "content": "answer " * 40},
        ],
        system_prompt="You are precise.",
    )
    builder = ContextBuilder(
        ContextConfig(
            context_window_tokens=120,
            reserved_output_tokens=10,
            keep_recent_messages=1,
            max_compression_passes=1,
        ),
        summarizer=FixedSummarizer(),
    )

    built = builder.build(session.messages, system_prompt=session.system_prompt)

    assert built.compressed is True
    assert built.messages[0] == SystemMessage("You are precise.")
    assert isinstance(built.messages[1], ContextSummaryMessage)
    assert isinstance(session.messages[0], UserMessage)
    assert built.estimated_tokens_before > built.estimated_tokens


def test_core_reports_context_error_without_calling_model() -> None:
    model = EchoModel()
    builder = ContextBuilder(
        ContextConfig(context_window_tokens=10, reserved_output_tokens=2),
    )
    result = AgentCore(
        model, session=AgentSession(system_prompt="system " * 20), context_builder=builder
    ).prompt("question")

    assert result.stop_reason is StopReason.ERROR
    assert result.error is not None
    assert result.error.code == "CONTEXT_COMPRESSION_UNAVAILABLE"
    assert model.messages == []
    assert result.context_metadata == ()


def test_context_error_is_structured_for_invalid_budget() -> None:
    try:
        ContextBuilder(ContextConfig(context_window_tokens=4, reserved_output_tokens=4)).build([])
    except ContextError as error:
        assert error.code == "CONTEXT_BUDGET_INVALID"
    else:
        raise AssertionError("expected ContextError")


def test_provider_translates_internal_context_summary() -> None:
    payload = OpenAIMessageCodec().request(
        ModelRequest((ContextSummaryMessage("facts", "summary-1"),))
    )
    message = payload["messages"][0]
    assert message == {
        "role": "user",
        "content": "[Context summary of earlier conversation]\nfacts",
    }
