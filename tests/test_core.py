import json
import threading
from collections.abc import Sequence

from minimal_agent.core import AgentCore, RunControl, StopReason
from minimal_agent.events import EventKind
from minimal_agent.protocol import ChatMessage, ModelResponse, ModelStreamChunk, ToolCall
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
        [
            ToolDefinition(
                "read",
                "Read a value",
                {"type": "object", "additionalProperties": False},
                lambda args: {"ok": True},
            )
        ]
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


def test_stream_yields_incremental_content_and_preserves_terminal_order() -> None:
    class StreamingModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            raise AssertionError("streaming model should use stream")

        def stream(self, messages: Sequence[ChatMessage]):
            yield ModelStreamChunk(content_delta="hel")
            yield ModelStreamChunk(content_delta="lo", done=True)

    events = list(AgentCore(StreamingModel()).stream("say hello"))

    assert [event.kind for event in events] == [
        EventKind.RUN_STARTED,
        EventKind.MODEL_CALL_STARTED,
        EventKind.MODEL_CONTENT_DELTA,
        EventKind.MODEL_CONTENT_DELTA,
        EventKind.MODEL_RESPONSE,
        EventKind.FINAL_RESPONSE,
    ]
    assert [
        event.data["content_delta"]
        for event in events
        if event.kind is EventKind.MODEL_CONTENT_DELTA
    ] == [
        "hel",
        "lo",
    ]
    assert events[-1].data["content"] == "hello"


def test_completed_stream_does_not_cancel_caller_control() -> None:
    class FinalStreamingModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            raise AssertionError("streaming model should use stream")

        def stream(self, messages: Sequence[ChatMessage]):
            yield ModelStreamChunk(content_delta="done", done=True)

    control = RunControl()
    assert (
        list(AgentCore(FinalStreamingModel()).stream("say", control))[-1].kind
        is EventKind.FINAL_RESPONSE
    )
    assert control.stop_reason is None


def test_stream_buffers_tool_call_fragments_before_execution() -> None:
    class StreamingToolModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            raise AssertionError("streaming model should use stream")

        def stream(self, messages: Sequence[ChatMessage]):
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamChunk(tool_call_id="call-1", tool_name="echo")
                yield ModelStreamChunk(tool_call_id="call-1", arguments_delta='{"value":')
                yield ModelStreamChunk(tool_call_id="call-1", arguments_delta=" 7}", done=True)
            else:
                yield ModelStreamChunk(content_delta="done", done=True)

    registry = ToolRegistry(
        [
            ToolDefinition(
                "echo",
                "Echo a value",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                lambda args: args["value"],
            )
        ]
    )
    events = list(AgentCore(StreamingToolModel(), registry).stream("echo"))

    assert any(event.kind is EventKind.TOOL_CALL_DELTA for event in events)
    tool_result = next(event for event in events if event.kind is EventKind.TOOL_RESULT_PRODUCED)
    assert '"ok":true' in str(tool_result.data["result"])
    assert events[-1].kind is EventKind.FINAL_RESPONSE


def test_stream_rejects_incomplete_tool_call_fragments() -> None:
    class IncompleteModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            raise AssertionError("streaming model should use stream")

        def stream(self, messages: Sequence[ChatMessage]):
            yield ModelStreamChunk(tool_call_id="call-1", arguments_delta='{"value": 7}')

    events = list(AgentCore(IncompleteModel()).stream("echo"))

    error = next(event for event in events if event.kind is EventKind.RUN_ERROR)
    assert error.data["error"].code == "MODEL_ERROR"


def test_steering_is_applied_at_the_next_model_boundary() -> None:
    first_call_started = threading.Event()
    release_first_call = threading.Event()

    class SteerableModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            self.calls += 1
            if self.calls == 1:
                first_call_started.set()
                release_first_call.wait(timeout=2)
                return ModelResponse(tool_calls=(ToolCall("call-1", "echo", '{"value": 1}'),))
            assert messages[-1]["content"] == "please use the other value"
            return ModelResponse(content="done")

    registry = ToolRegistry(
        [
            ToolDefinition(
                "echo",
                "Echo",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                lambda args: args["value"],
            )
        ]
    )
    core = AgentCore(SteerableModel(), registry)
    result_holder = []
    worker = threading.Thread(target=lambda: result_holder.append(core.prompt("start")))
    worker.start()
    assert first_call_started.wait(timeout=2)
    assert core.steer("please use the other value") is True
    release_first_call.set()
    worker.join(timeout=2)

    assert result_holder[0].final_response == "done"
    assert any(
        event.kind is EventKind.STEERING_MESSAGE_ACCEPTED for event in result_holder[0].events
    )
    assert core.steer("too late") is False


def test_concurrent_prompt_is_rejected_for_one_session() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            started.set()
            release.wait(timeout=2)
            return ModelResponse(content="done")

    core = AgentCore(SlowModel())
    first_result = []
    worker = threading.Thread(target=lambda: first_result.append(core.prompt("first")))
    worker.start()
    assert started.wait(timeout=2)
    second = core.prompt("second")
    release.set()
    worker.join(timeout=2)

    assert second.error is not None
    assert second.error.code == "SESSION_BUSY"
    assert first_result[0].final_response == "done"


def test_concurrent_stream_exposes_busy_error_event() -> None:
    started = threading.Event()
    release = threading.Event()

    class SlowModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            started.set()
            release.wait(timeout=2)
            return ModelResponse(content="done")

    core = AgentCore(SlowModel())
    first = threading.Thread(target=lambda: core.prompt("first"))
    first.start()
    assert started.wait(timeout=2)
    events = list(core.stream("second"))
    release.set()
    first.join(timeout=2)

    assert len(events) == 1
    assert events[0].kind is EventKind.RUN_ERROR
    assert events[0].data["error"].code == "SESSION_BUSY"


def test_registry_turns_unknown_tool_into_structured_result() -> None:
    result = ToolRegistry().execute(ToolCall("call-1", "missing", "{}"))

    assert result.error is not None
    assert result.error.code == "UNKNOWN_TOOL"


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
                "read",
                "Read",
                {
                    "type": "object",
                    "properties": {"step": {"type": "integer"}},
                    "required": ["step"],
                    "additionalProperties": False,
                },
                lambda args: calls.append(args) or {"ok": True},
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
        ToolRegistry(
            [
                ToolDefinition(
                    "read",
                    "Read",
                    {"type": "object", "additionalProperties": False},
                    lambda args: {"ok": True},
                )
            ]
        ),
    ).prompt("go")

    assert result.stop_reason is StopReason.FINAL
    assert result.final_response == "recovered"


def test_repeated_tool_call_stops_on_second_observation() -> None:
    class AlwaysRepeats:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(tool_calls=(ToolCall("call", "read", '{"x": 1}'),))

    result = AgentCore(
        AlwaysRepeats(),
        ToolRegistry(
            [
                ToolDefinition(
                    "read",
                    "Read",
                    {"type": "object", "additionalProperties": False},
                    lambda args: {"ok": True},
                )
            ]
        ),
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


def test_run_result_contains_complete_trace_snapshot() -> None:
    result = AgentCore(
        ReadThenAnswer(),
        ToolRegistry(
            [
                ToolDefinition(
                    "read",
                    "Read",
                    {"type": "object", "additionalProperties": False},
                    lambda args: {"ok": True},
                )
            ]
        ),
    ).prompt("read it")

    assert result.events[0].kind is EventKind.RUN_STARTED
    assert result.events[-1].kind is EventKind.FINAL_RESPONSE
    assert [event.sequence for event in result.events] == list(range(1, len(result.events) + 1))
    assert {event.run_id for event in result.events} == {result.run_id}
    assert all(event.elapsed_ms >= 0 for event in result.events)
    assert all(event.occurred_at.tzinfo is not None for event in result.events)
    try:
        result.events[0].data["changed"] = True
    except TypeError:
        pass
    else:
        raise AssertionError("Trace event data must be immutable")


def test_listener_failure_is_isolated_and_recorded() -> None:
    observed: list[EventKind] = []

    def broken_listener(event) -> None:
        raise RuntimeError("display failed")

    def healthy_listener(event) -> None:
        observed.append(event.kind)

    core = AgentCore(
        ReadThenAnswer(),
        ToolRegistry(
            [
                ToolDefinition(
                    "read",
                    "Read",
                    {"type": "object", "additionalProperties": False},
                    lambda args: {"ok": True},
                )
            ]
        ),
    )
    core.subscribe(broken_listener)
    core.subscribe(healthy_listener)
    result = core.prompt("read it")

    assert result.stop_reason is StopReason.FINAL
    assert EventKind.LISTENER_ERROR in [event.kind for event in result.events]
    assert EventKind.FINAL_RESPONSE in observed


def test_tool_result_trace_includes_result_and_success() -> None:
    result = AgentCore(
        ReadThenAnswer(),
        ToolRegistry(
            [
                ToolDefinition(
                    "read",
                    "Read",
                    {"type": "object", "additionalProperties": False},
                    lambda args: {"ok": True},
                )
            ]
        ),
    ).prompt("read it")

    tool_event = next(
        event for event in result.events if event.kind is EventKind.TOOL_RESULT_PRODUCED
    )
    assert tool_event.data["success"] is True
    assert json.loads(str(tool_event.data["result"]))["data"] == {"ok": True}


def test_registry_validates_arguments_before_execution() -> None:
    called = False

    def execute(arguments):
        nonlocal called
        called = True
        return {"value": arguments["value"]}

    registry = ToolRegistry(
        [
            ToolDefinition(
                "echo",
                "Echo",
                {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                execute,
            )
        ]
    )
    result = registry.execute(ToolCall("call", "echo", '{"value": 1}'))

    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "INVALID_ARGUMENTS"
    assert called is False


def test_confirmation_and_authorization_happen_before_execution() -> None:
    called = False

    class Deny:
        def authorize(self, context) -> bool:
            return False

    registry = ToolRegistry(
        [
            ToolDefinition(
                "write",
                "Write",
                {"type": "object", "additionalProperties": False},
                lambda arguments: called,
                requires_confirmation=True,
            )
        ],
        authorizer=Deny(),
    )
    result = registry.execute(ToolCall("call", "write", "{}"))

    assert result.error is not None
    assert result.error.code == "PERMISSION_DENIED"
    assert called is False


def test_tool_errors_and_result_size_are_structured() -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                "tiny",
                "Tiny",
                {"type": "object", "additionalProperties": False},
                lambda arguments: "0123456789",
                max_result_bytes=4,
            )
        ]
    )
    result = registry.execute(ToolCall("call", "tiny", "{}"))

    assert result.error is not None
    assert result.error.code == "TOOL_RESULT_TOO_LARGE"
