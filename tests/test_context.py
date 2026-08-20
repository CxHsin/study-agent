from collections.abc import Sequence

from minimal_agent.context import (
    ContextBuilder,
    ContextConfig,
    ContextError,
    ContextSummarySections,
    ModelSummarizer,
)
from minimal_agent.core import AgentCore, StopReason
from minimal_agent.cost import ModelCallPurpose, UsageStatus
from minimal_agent.protocol import (
    ChatMessage,
    ContextSummaryMessage,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    StreamEnd,
    SystemMessage,
    TextDelta,
    UserMessage,
)
from minimal_agent.provider_client import ProviderClient
from minimal_agent.providers import OpenAIMessageCodec
from minimal_agent.session import AgentSession


class EchoModel:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        self.messages = list(messages)
        return ModelResponse(content="ok")


class FixedSummarizer:
    def summarize(self, messages: Sequence[ChatMessage]) -> ContextSummarySections:
        return ContextSummarySections(
            "Complete task.",
            constraints=("Scoped.",),
            progress=("Facts retained.",),
            files=("a.py: read",),
            next_steps=("Continue.",),
            facts=("History complete.",),
        )


def test_context_builder_keeps_system_prompt_and_compresses_old_history() -> None:
    session = AgentSession(
        [
            {"role": "user", "content": "old " * 80},
            {"role": "assistant", "content": "answer " * 40},
        ],
        system_prompt="You are precise.",
    )
    builder = ContextBuilder(
        ContextConfig(
            context_window_tokens=150,
            reserved_output_tokens=10,
            compression_trigger_ratio=0.5,
            keep_recent_messages=1,
            max_compression_passes=1,
        ),
        summarizer=FixedSummarizer(),
    )

    built = builder.build(session.messages, system_prompt=session.system_prompt)

    assert built.compressed is True
    assert built.messages[0] == SystemMessage("You are precise.")
    assert isinstance(built.messages[1], ContextSummaryMessage)
    assert '"objective":"Complete task."' in built.messages[1].content
    assert built.summaries[0].sections is not None
    assert isinstance(session.messages[0], UserMessage)
    assert built.estimated_tokens_before > built.estimated_tokens


def test_context_summary_versions_and_cache_are_deterministic() -> None:
    messages = (UserMessage("old " * 80), UserMessage("recent"))
    builder = ContextBuilder(
        ContextConfig(
            context_window_tokens=120,
            reserved_output_tokens=10,
            compression_trigger_ratio=0.5,
            keep_recent_messages=1,
            max_compression_passes=1,
        ),
        summarizer=FixedSummarizer(),
    )

    first = builder.build(messages)
    second = builder.build(messages)

    assert first.summaries[0].version == second.summaries[0].version
    assert second.cache_hits == 1

    with_system = builder.build(messages, system_prompt="System")
    changed_source = builder.build((UserMessage("changed " * 80), UserMessage("recent")))

    assert with_system.cache_hits == 1
    assert with_system.summaries[0].start_index == 1
    assert with_system.summaries[0].end_index == 2
    assert with_system.summaries[0].source_ids == ("1",)
    assert with_system.summaries[0].version == first.summaries[0].version
    assert changed_source.summaries[0].version != first.summaries[0].version


def test_model_summarizer_requires_structured_json() -> None:
    class SummaryAdapter:
        profile = ModelProfile("test", "summary", 8192, 1024)

        def __init__(self) -> None:
            self.request = None

        def stream(self, request):
            self.request = request
            yield TextDelta(
                '{"objective":"Ship","constraints":["Scoped"],"progress":["Read"],'
                '"files":["a.py: read"],"next_steps":["Edit"],"facts":["Tests pass"]}'
            )
            yield StreamEnd()

    adapter = SummaryAdapter()
    sections = ModelSummarizer(ProviderClient(adapter)).summarize((UserMessage("work"),))

    assert sections.objective == "Ship"
    assert sections.next_steps == ("Edit",)
    assert adapter.request.options.tools == ()


def test_compaction_and_primary_model_calls_have_distinct_usage_records() -> None:
    class SummaryThenAnswerAdapter:
        profile = ModelProfile("test", "summary-and-agent", 150, 10)

        def stream(self, request):
            if "Summarize the delimited conversation" in request.messages[0].content:
                yield TextDelta(
                    '{"objective":"Ship","constraints":[],"progress":[],"files":[],'
                    '"next_steps":["Continue"],"facts":[]}'
                )
            else:
                yield TextDelta("done")
            yield StreamEnd()

    client = ProviderClient(SummaryThenAnswerAdapter())
    builder = ContextBuilder(
        ContextConfig(
            context_window_tokens=150,
            reserved_output_tokens=10,
            compression_trigger_ratio=0.5,
            keep_recent_messages=1,
            max_compression_passes=1,
        ),
        summarizer=ModelSummarizer(client),
    )
    session = AgentSession(
        (UserMessage("old " * 80), UserMessage("recent")),
    )

    result = AgentCore(client, session=session, context_builder=builder).prompt("finish")

    usage = result.trace.usage()
    assert result.final_response == "done"
    assert [item.purpose for item in usage] == [
        ModelCallPurpose.CONTEXT_SUMMARY,
        ModelCallPurpose.AGENT,
    ]
    assert [item.step for item in usage] == [1, 1]
    assert len({item.call_id for item in usage}) == 2
    assert all(item.status is UsageStatus.SUCCEEDED for item in usage)


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
