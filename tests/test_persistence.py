from collections.abc import Sequence

from minimal_agent.core import AgentCore
from minimal_agent.persistence import InMemoryRepository, SQLiteRepository
from minimal_agent.protocol import ChatMessage, ModelResponse


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


def test_repository_loads_session_and_marks_unknown_tool_resolution(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "restore.sqlite")
    repository.save_session("session", "system", [{"role": "user", "content": "hello"}])
    repository.start_run("run", "session")
    repository.record_tool("run", "call", "write", '{"password":"secret"}', "started")

    assert repository.load_session("session") == ("system", ({"role": "user", "content": "hello"},))
    repository.recover()
    assert repository.run_status("run") == "needs_resolution"
