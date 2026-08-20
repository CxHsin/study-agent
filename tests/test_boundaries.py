from minimal_agent.persistence import SQLiteRepository


def test_sqlite_repository_exposes_split_adapters(tmp_path) -> None:
    repository = SQLiteRepository(tmp_path / "adapters.sqlite")

    repository.sessions.save_session("session", None, [{"role": "user", "content": "hello"}])
    repository.runs.start_run("run", "session")
    repository.runs.append_event("run", 1, "run_started", {})
    repository.tools.record_tool_started("run", "call", "read", "{}")

    assert repository.sessions.load_session("session") is not None
    assert repository.runs.continuation_session("run") is None
    assert repository.tools.unresolved_tools()[0].call_id == "call"

    repository.close()
