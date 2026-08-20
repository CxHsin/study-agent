from minimal_agent.agent_loop import AgentLoop
from minimal_agent.core import RunControl, RunResult, StopReason
from minimal_agent.persistence import SQLiteRepository


def test_agent_loop_exposes_a_small_execution_seam() -> None:
    seen = []

    def implementation(run_id, user_input, control, event_sink):
        seen.append((run_id, user_input, control, event_sink))
        return RunResult("done", StopReason.FINAL, 1, run_id)

    control = RunControl()
    result = AgentLoop(implementation).run("run-1", "hello", control)

    assert result.final_response == "done"
    assert seen[0][0:2] == ("run-1", "hello")
    assert seen[0][2] is control


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
