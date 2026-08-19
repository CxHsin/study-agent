import json

from minimal_agent.core import RunControl
from minimal_agent.protocol import ToolCall
from minimal_agent.workspace_tools import WorkspaceTools


class Confirm:
    def confirm(self, context) -> bool:
        return True


def call(registry, name, arguments):
    return registry.execute(ToolCall("call", name, json.dumps(arguments)))


def test_read_file_supports_inclusive_line_range(tmp_path):
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = call(WorkspaceTools(tmp_path).registry(), "read_file", {"path": "notes.txt", "start_line": 2, "end_line": 2})
    assert result.ok and result.data["content"] == "two\n"


def test_read_file_rejects_workspace_escape_and_non_utf8(tmp_path):
    (tmp_path / "bad.bin").write_bytes(b"\xff")
    registry = WorkspaceTools(tmp_path).registry()
    assert call(registry, "read_file", {"path": "../bad"}).error.code == "PATH_OUTSIDE_WORKSPACE"
    assert call(registry, "read_file", {"path": "bad.bin"}).error.code == "NOT_UTF8_TEXT"


def test_bash_requires_confirmation_and_returns_process_data(tmp_path):
    denied = call(WorkspaceTools(tmp_path).registry(), "bash", {"command": "printf ok"})
    assert denied.error.code == "PERMISSION_DENIED"
    result = call(WorkspaceTools(tmp_path, confirmation=Confirm()).registry(), "bash", {"command": "printf ok"})
    assert result.ok and result.data == {"exit_code": 0, "stdout": "ok", "stderr": "", "timed_out": False}


def test_bash_timeout_and_cancellation_are_distinct(tmp_path):
    registry = WorkspaceTools(tmp_path, confirmation=Confirm()).registry()
    timed = call(registry, "bash", {"command": "sleep 1", "timeout_ms": 10})
    assert timed.ok and timed.data["timed_out"] is True
    control = RunControl()
    control.cancel()
    cancelled = registry.execute(ToolCall("call", "bash", json.dumps({"command": "sleep 1"})), control=control)
    assert cancelled.error.code == "TOOL_CANCELLED"
