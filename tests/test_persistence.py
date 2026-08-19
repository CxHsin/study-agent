from collections.abc import Sequence

from minimal_agent.core import AgentCore
from minimal_agent.persistence import InMemoryRepository, SQLiteRepository
from minimal_agent.protocol import ChatMessage, ModelResponse, ToolCall
from minimal_agent.session import AgentSession


def test_sqlite_repository_preserves_events_and_unresolved_tool(tmp_path) -> None:
    path = tmp_path / "agent.sqlite"
    repository = SQLiteRepository(path)

    repository.save_session("session-1", "system", [{"role": "user", "content": "key=secret"}])
    repository.start_run("run-1", "session-1")
    repository.append_event("run-1", 1, "run_started", {"authorization": "Bearer secret"})
    repository.record_tool("run-1", "call-1", "write", '{"value":1}', "started", idempotent=False)
    repository.record_tool("run-1", "call-1", "write", '{"value":1}', "completed", '{"ok":true}')
    repository.record_tool("run-1", "call-2", "write", '{"value":2}', "started", idempotent=False)

    assert repository.unresolved_tools()[0].call_id == "call-2"
    assert repository.event_payloads("run-1")[0]["authorization"] == "[REDACTED]"

    repository.close()
    reopened = SQLiteRepository(path)
    assert reopened.unresolved_tools()[0].run_id == "run-1"


def test_core_can_persist_a_completed_run() -> None:
    class FinalModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            return ModelResponse(content="done")

    repository = InMemoryRepository()
    result = AgentCore(FinalModel(), repository=repository).prompt("hello")

    assert result.final_response == "done"
    assert repository.run_status(result.run_id) == "finished"
    assert repository.event_count(result.run_id) >= 4


def test_recovery_requires_idempotency_and_links_continuation(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "recovery.sqlite")
    repository.start_run("parent", "session")
    repository.record_tool("parent", "unsafe", "write", "{}", "started", idempotent=False)
    repository.record_tool("parent", "safe", "read", "{}", "started", idempotent=True)

    try:
        repository.retry_tool("parent", "unsafe", "child-unsafe", "session")
    except RuntimeError as error:
        assert "idempotent" in str(error)
    else:
        raise AssertionError("Expected unsafe retry to be rejected")

    repository.retry_tool("parent", "safe", "child-safe", "session")
    repository.resolve_tool("parent", "unsafe", '{"ok":true}', "child-resolved", "session")

    assert repository.run_status("child-safe") == "running"
    assert repository.run_status("child-resolved") == "running"
    assert repository.event_payloads("child-resolved")[1]["call_id"] == "unsafe"


def test_redacts_secrets_inside_json_strings(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "secrets.sqlite")
    repository.save_session("s", None, [{"role": "user", "content": '{"password":"supersecret"}'}])

    loaded = repository.load_session("s")

    assert "supersecret" not in str(loaded)


def test_tool_lifecycle_is_committed_as_one_transaction(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "lifecycle.sqlite")
    repository.record_tool_lifecycle("run", "call", "read", "{}", '{"ok":true}')

    assert repository.unresolved_tools() == ()
    statuses = repository._connection.execute(
        "SELECT status FROM tool_executions WHERE run_id='run' ORDER BY id"
    ).fetchall()
    assert [item[0] for item in statuses] == ["requested", "started", "completed"]


def test_core_continues_a_persisted_continuation_run(tmp_path) -> None:
    seen: list[tuple[dict[str, object], ...]] = []

    class FinalModel:
        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            seen.append(tuple(messages))
            return ModelResponse(content="continued")

    repository = SQLiteRepository(tmp_path / "continuation.sqlite")
    repository.save_session("s", None, [{"role": "user", "content": "old"}])
    repository.start_run("parent", "s")
    repository.create_continuation("parent", "child", "s")
    core = AgentCore(FinalModel(), session=AgentSession(session_id="s"), repository=repository)

    result = core.continue_run("child", "continue")

    assert result.final_response == "continued"
    assert seen[0][0]["content"] == "old"


def test_continuation_injects_resolved_tool_result(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "resolved-continuation.sqlite")
    repository.save_session(
        "s",
        None,
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": (ToolCall("call", "read", "{}"),),
            }
        ],
    )
    repository.start_run("parent", "s")
    repository.record_tool_started("parent", "call", "read", "{}")
    repository.resolve_tool("parent", "call", '{"ok":true}', "child", "s")

    restored = repository.continuation_session("child")

    assert restored is not None
    assert restored[2][-1] == {
        "role": "tool",
        "tool_call_id": "call",
        "content": '{"ok":true}',
    }


def test_repository_loads_session_and_marks_unknown_tool_resolution(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "restore.sqlite")
    repository.save_session("session", "system", [{"role": "user", "content": "hello"}])
    repository.start_run("run", "session")
    repository.record_tool("run", "call", "write", '{"password":"secret"}', "started")

    assert repository.load_session("session") == ("system", ({"role": "user", "content": "hello"},))
    repository.recover()
    assert repository.run_status("run") == "needs_resolution"


def test_session_round_trip_restores_nested_tool_calls(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "tool-history.sqlite")
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": (ToolCall("call", "read", '{"path":"notes"}'),),
        }
    ]

    repository.save_session("s", None, messages)

    loaded = repository.load_session("s")
    assert loaded is not None
    assert loaded[1][0]["tool_calls"] == (ToolCall("call", "read", '{"path":"notes"}'),)


def test_event_redaction_handles_nested_dataclass_values(tmp_path) -> None:
    from minimal_agent.core import RunError

    repository = SQLiteRepository(tmp_path / "event-redaction.sqlite")
    repository.append_event(
        "run",
        1,
        "run_error",
        {"error": RunError("X", "api_key=supersecret", "provider", 1)},
    )

    payload = repository.event_payloads("run")[0]
    assert payload["error"]["message"] == "api_key=[REDACTED]"


def test_repository_records_tool_started_before_external_execution(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "started-before-failure.sqlite")
    repository.record_tool_started("run", "call", "unstable", "{}")

    assert repository.unresolved_tools()[0].call_id == "call"
