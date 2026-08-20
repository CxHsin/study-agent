from minimal_agent.core import AgentCore
from minimal_agent.persistence import SQLiteRepository
from minimal_agent.protocol import ModelResponse, ToolCall
from minimal_agent.tools import ToolDefinition, ToolRegistry


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


def test_agent_core_routes_work_through_independently_injected_repositories() -> None:
    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(tool_calls=(ToolCall("call", "read", "{}"),))
            return ModelResponse(content="done")

    class Sessions:
        def __init__(self) -> None:
            self.saved = 0

        def save_session(self, *args) -> None:
            self.saved += 1

    class Runs:
        def __init__(self) -> None:
            self.started = self.finished = 0
            self.events = []

        def start_run(self, *args) -> None:
            self.started += 1

        def append_event(self, run_id, sequence, kind, data) -> None:
            self.events.append(kind)

        def finish_run(self, *args) -> None:
            self.finished += 1

    class ToolExecutions:
        def __init__(self) -> None:
            self.started = self.completed = 0

        def record_tool_started(self, *args, **kwargs) -> None:
            self.started += 1

        def record_tool(self, *args, **kwargs) -> None:
            self.completed += 1

    sessions = Sessions()
    runs = Runs()
    executions = ToolExecutions()
    tools = ToolRegistry(
        (
            ToolDefinition(
                "read",
                "Read",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                lambda _arguments: "ok",
            ),
        )
    )

    result = AgentCore(
        Model(),
        tools,
        session_repository=sessions,
        run_repository=runs,
        tool_execution_repository=executions,
    ).prompt("read")

    assert result.final_response == "done"
    assert sessions.saved >= 2
    assert (runs.started, runs.finished) == (1, 1)
    assert (executions.started, executions.completed) == (1, 1)
