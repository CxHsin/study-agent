import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from minimal_agent.agent import (
    AgentSession,
    ChatMessage,
    ModelResponse,
    StopReason,
    TaskResult,
    TaskStatus,
    ToolCall,
    TraceEvent,
)
from minimal_agent.experimental_tools import IdempotentIntentRecorder
from minimal_agent.session_store import SQLiteSessionStore
from minimal_agent.workspace_tools import WorkspaceTools


class FinalResponseModel:
    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return ModelResponse(content="The task is complete.")


class ReadFileThenAnswerModel:
    def __init__(self) -> None:
        self._step = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        if self._step == 0:
            self._step += 1
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="read_file",
                        arguments='{"path":"notes.txt"}',
                    ),
                )
            )

        tool_message = messages[-1]
        tool_result = json.loads(cast(str, tool_message["content"]))
        if (
            tool_message["role"] == "tool"
            and tool_message["tool_call_id"] == "call-1"
            and tool_result["ok"] is True
            and tool_result["data"]["content"] == "The answer is 42."
        ):
            return ModelResponse(content="The file says the answer is 42.")

        raise AssertionError("The model did not receive the expected Tool Result.")


class ToolErrorThenAnswerModel:
    def __init__(self, tool_call: ToolCall, expected_error_code: str) -> None:
        self._tool_call = tool_call
        self._expected_error_code = expected_error_code
        self._step = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        if self._step == 0:
            self._step += 1
            return ModelResponse(tool_calls=(self._tool_call,))

        tool_message = messages[-1]
        tool_result = json.loads(cast(str, tool_message["content"]))
        if (
            tool_message["role"] == "tool"
            and tool_message["tool_call_id"] == self._tool_call.id
            and tool_result["ok"] is False
            and tool_result["error"]["code"] == self._expected_error_code
        ):
            return ModelResponse(content=f"Recovered from {self._expected_error_code}.")

        raise AssertionError("The model did not receive the expected tool error.")


class ListFilesThenAnswerModel:
    def __init__(self) -> None:
        self._step = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        if self._step == 0:
            self._step += 1
            return ModelResponse(
                tool_calls=(ToolCall(id="call-list", name="list_files", arguments='{"path":"."}'),)
            )

        tool_result = json.loads(cast(str, messages[-1]["content"]))
        entries = tool_result["data"]["entries"]
        if entries == [
            {"name": "notes.txt", "type": "file"},
            {"name": "sources", "type": "directory"},
        ]:
            return ModelResponse(content="I found notes.txt and the sources directory.")

        raise AssertionError("The model did not receive the expected directory listing.")


class MultipleFilesThenAnswerModel:
    def __init__(self) -> None:
        self._step = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        if self._step == 0:
            self._step += 1
            return ModelResponse(
                tool_calls=(
                    ToolCall(id="call-a", name="read_file", arguments='{"path":"a.txt"}'),
                    ToolCall(id="call-b", name="read_file", arguments='{"path":"b.txt"}'),
                )
            )

        tool_messages = messages[-2:]
        tool_ids = [message["tool_call_id"] for message in tool_messages]
        contents = [
            json.loads(cast(str, message["content"]))["data"]["content"]
            for message in tool_messages
        ]
        if tool_ids == ["call-a", "call-b"] and contents == ["A", "B"]:
            return ModelResponse(content="I read A, then B.")

        raise AssertionError("Tool Results were not appended in request order.")


class HistoryAwareModel:
    def __init__(self) -> None:
        self._task = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        if self._task == 0:
            self._task += 1
            return ModelResponse(content="First response.")

        assert list(messages) == [
            {"role": "user", "content": "First task."},
            {"role": "assistant", "content": "First response."},
            {"role": "user", "content": "Second task."},
        ]
        return ModelResponse(content="Second response with history.")


class ResetAwareModel:
    def __init__(self) -> None:
        self._task = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        if self._task == 0:
            self._task += 1
            return ModelResponse(content="Before reset.")

        assert list(messages) == [{"role": "user", "content": "After reset task."}]
        return ModelResponse(content="After reset.")


class NineToolStepsThenFinalModel:
    def __init__(self) -> None:
        self._step = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        self._step += 1
        if self._step <= 8:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{self._step}",
                        name="read_file",
                        arguments=f'{{"path":"missing-{self._step}.txt"}}',
                    ),
                )
            )
        return ModelResponse(content="Too late.")


class RepeatedToolModel:
    def __init__(self) -> None:
        self._step = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        self._step += 1
        if self._step <= 2:
            return ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"call-{self._step}",
                        name="read_file",
                        arguments='{"path":"notes.txt"}',
                    ),
                )
            )
        return ModelResponse(content="Repeated calls were allowed.")


def test_task_completes_when_model_returns_final_response() -> None:
    session = AgentSession(model=FinalResponseModel())

    result = session.submit("Complete this task without using a tool.")

    assert result == TaskResult(
        status=TaskStatus.COMPLETED,
        stop_reason=StopReason.FINAL_RESPONSE,
        final_response="The task is complete.",
        steps_used=1,
    )


def test_task_uses_read_file_result_before_completing(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The answer is 42.", encoding="utf-8")
    session = AgentSession(
        model=ReadFileThenAnswerModel(),
        tools=WorkspaceTools(tmp_path),
    )

    result = session.submit("Read notes.txt and report the answer.")

    assert result == TaskResult(
        status=TaskStatus.COMPLETED,
        stop_reason=StopReason.FINAL_RESPONSE,
        final_response="The file says the answer is 42.",
        steps_used=2,
    )
    assert session.run_state.value == "completed"


def test_task_records_tool_execution_status(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The answer is 42.", encoding="utf-8")
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    session = AgentSession(
        model=ReadFileThenAnswerModel(),
        tools=WorkspaceTools(tmp_path),
        message_store=store,
        execution_store=store,
    )

    session.submit("Read notes.txt and report the answer.")

    execution = store.get_tool_execution("call-1")
    assert execution is not None
    assert execution["status"] == "succeeded"
    assert json.loads(execution["result"])["ok"] is True


def test_experimental_intent_recorder_is_idempotent() -> None:
    recorder = IdempotentIntentRecorder()
    call = ToolCall("call-1", "record_intent", '{"intent_id":"payment-1","value":"send"}')

    first = json.loads(recorder.execute(call))
    second = json.loads(recorder.execute(ToolCall("call-2", call.name, call.arguments)))

    assert first["data"]["created"] is True
    assert second["data"]["created"] is False
    assert recorder.records == {"payment-1": "send"}
    assert recorder.execution_count == 2


def test_trace_records_file_task_in_execution_order(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("The answer is 42.", encoding="utf-8")
    events: list[TraceEvent] = []
    session = AgentSession(
        model=ReadFileThenAnswerModel(),
        tools=WorkspaceTools(tmp_path),
        trace_sink=events.append,
    )

    session.submit("Read notes.txt and report the answer.")

    assert [event.kind for event in events] == [
        "agent_step",
        "model_call",
        "tool_call",
        "tool_result",
        "agent_step",
        "model_call",
        "task_end",
    ]
    tool_call = events[2]
    assert tool_call.data == {
        "name": "read_file",
        "tool_call_id": "call-1",
        "arguments": {"path": "notes.txt"},
    }
    tool_result = events[3]
    assert tool_result.data["status"] == "ok"
    assert tool_result.data["tool_call_id"] == "call-1"
    assert events[0].data["run_state"] == "running"
    assert events[4].data["run_state"] == "running"
    assert events[-1].data == {
        "status": "completed",
        "stop_reason": "final_response",
        "steps_used": 2,
    }
    assert len({event.run_id for event in events}) == 1
    assert [event.sequence for event in events] == list(range(1, 8))
    assert all(event.timestamp.endswith("Z") for event in events)
    assert all(event.duration_ms >= 0 for event in events if event.duration_ms is not None)


def test_task_uses_directory_listing_before_completing(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "sources").mkdir()
    (tmp_path / ".secret").write_text("hidden", encoding="utf-8")
    session = AgentSession(
        model=ListFilesThenAnswerModel(),
        tools=WorkspaceTools(tmp_path),
    )

    result = session.submit("List the files in the workspace.")

    assert result.final_response == "I found notes.txt and the sources directory."
    assert result.steps_used == 2


@pytest.mark.parametrize(
    ("tool_call", "expected_error_code"),
    [
        (
            ToolCall(id="call-unknown", name="delete_file", arguments='{"path":"notes.txt"}'),
            "UNKNOWN_TOOL",
        ),
        (
            ToolCall(id="call-invalid", name="read_file", arguments="not-json"),
            "INVALID_ARGUMENTS",
        ),
        (
            ToolCall(id="call-missing", name="read_file", arguments='{"path":"missing.txt"}'),
            "FILE_NOT_FOUND",
        ),
        (
            ToolCall(id="call-outside", name="read_file", arguments='{"path":"../secret.txt"}'),
            "PATH_OUTSIDE_WORKSPACE",
        ),
    ],
)
def test_task_can_recover_from_structured_tool_errors(
    tmp_path: Path,
    tool_call: ToolCall,
    expected_error_code: str,
) -> None:
    session = AgentSession(
        model=ToolErrorThenAnswerModel(tool_call, expected_error_code),
        tools=WorkspaceTools(tmp_path),
    )

    result = session.submit("Try the requested tool and handle any error.")

    assert result.final_response == f"Recovered from {expected_error_code}."
    assert result.steps_used == 2


def test_multiple_tool_calls_execute_in_request_order(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")
    session = AgentSession(
        model=MultipleFilesThenAnswerModel(),
        tools=WorkspaceTools(tmp_path),
    )

    result = session.submit("Read a.txt and b.txt in that order.")

    assert result.final_response == "I read A, then B."
    assert result.steps_used == 2


def test_consecutive_tasks_share_conversation_history() -> None:
    session = AgentSession(model=HistoryAwareModel())

    session.submit("First task.")
    result = session.submit("Second task.")

    assert result.final_response == "Second response with history."
    assert result.steps_used == 1


def test_reset_clears_conversation_history() -> None:
    session = AgentSession(model=ResetAwareModel())
    session.submit("Before reset task.")

    session.reset()
    result = session.submit("After reset task.")

    assert result.final_response == "After reset."
    assert result.steps_used == 1


def test_session_history_survives_recreating_agent_session(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    first_model = FinalResponseModel()
    first = AgentSession(model=first_model, message_store=store)

    first.submit("First task.")

    class RestartAwareModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            assert list(messages) == [
                {"role": "user", "content": "First task."},
                {"role": "assistant", "content": "The task is complete."},
                {"role": "user", "content": "After restart task."},
            ]
            return ModelResponse(content="History restored.")

    second_model = RestartAwareModel()
    second = AgentSession(
        model=second_model,
        message_store=store,
        session_id=first.session_id,
    )

    result = second.submit("After restart task.")

    assert result.final_response == "History restored."


def test_reset_starts_a_new_persistent_session(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    session = AgentSession(model=FinalResponseModel(), message_store=store)
    first_session_id = session.session_id

    session.submit("Keep this history.")
    session.reset()

    assert session.session_id != first_session_id
    assert store.load_messages(first_session_id or "") == [
        {"role": "user", "content": "Keep this history."},
        {"role": "assistant", "content": "The task is complete."},
    ]
    assert store.load_messages(session.session_id or "") == []


def test_task_fails_when_model_exceeds_step_limit(tmp_path: Path) -> None:
    session = AgentSession(
        model=NineToolStepsThenFinalModel(),
        tools=WorkspaceTools(tmp_path),
    )

    result = session.submit("Keep trying files forever.")

    assert result == TaskResult(
        status=TaskStatus.FAILED,
        stop_reason=StopReason.MAX_STEPS,
        final_response=None,
        steps_used=8,
    )


def test_task_fails_on_consecutive_repeated_tool_calls(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    session = AgentSession(
        model=RepeatedToolModel(),
        tools=WorkspaceTools(tmp_path),
    )

    result = session.submit("Read notes.txt without looping.")

    assert result == TaskResult(
        status=TaskStatus.FAILED,
        stop_reason=StopReason.REPEATED_TOOL_CALL,
        final_response=None,
        steps_used=2,
    )


class RaisingTool:
    def execute(self, tool_call: ToolCall) -> str:
        raise OSError("tool process failed")


class InterruptingTool:
    def execute(self, tool_call: ToolCall) -> str:
        raise KeyboardInterrupt


def test_tool_exception_emits_terminal_trace_event() -> None:
    events: list[TraceEvent] = []

    class ToolRequestModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(tool_calls=(ToolCall("call-error", "x", "{}"),))

    session = AgentSession(model=ToolRequestModel(), tools=RaisingTool(), trace_sink=events.append)
    result = session.submit("run the tool")

    assert result.stop_reason is StopReason.TOOL_ERROR
    assert [event.kind for event in events] == [
        "agent_step",
        "model_call",
        "tool_call",
        "tool_result",
        "task_end",
    ]
    assert events[3].data["status"] == "exception"
    assert events[-1].data["stop_reason"] == "tool_error"
    assert session.run_state.value == "failed"


def test_tool_exception_leaves_execution_status_unknown(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")

    class ToolRequestModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(tool_calls=(ToolCall("call-error", "x", "{}"),))

    session = AgentSession(
        model=ToolRequestModel(),
        tools=RaisingTool(),
        message_store=store,
        execution_store=store,
    )

    result = session.submit("run the tool")

    assert result.stop_reason is StopReason.TOOL_ERROR
    execution = store.get_tool_execution("call-error")
    assert execution is not None
    assert execution["status"] == "unknown"


def test_interrupting_tool_emits_cancelled_trace_event() -> None:
    events: list[TraceEvent] = []

    class ToolRequestModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(tool_calls=(ToolCall("call-cancel", "x", "{}"),))

    session = AgentSession(
        model=ToolRequestModel(), tools=InterruptingTool(), trace_sink=events.append
    )

    with pytest.raises(KeyboardInterrupt):
        session.submit("cancel the tool")

    assert events[-1].data == {
        "status": "cancelled",
        "stop_reason": "user_cancelled",
        "steps_used": 1,
    }
