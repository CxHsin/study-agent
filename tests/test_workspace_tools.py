import json
import threading
import time

from minimal_agent.core import RunControl
from minimal_agent.protocol import ToolCall, ToolError
from minimal_agent.workspace_tools import WorkspaceTools


class Confirm:
    def confirm(self, context) -> bool:
        return True


def call(registry, name, arguments):
    return registry.execute(ToolCall("call", name, json.dumps(arguments)))


def test_read_file_supports_inclusive_line_range(tmp_path):
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    result = call(
        WorkspaceTools(tmp_path).registry(),
        "read_file",
        {"path": "notes.txt", "start_line": 2, "end_line": 2},
    )
    assert result.ok and result.data["content"] == "two\n"


def test_read_file_rejects_workspace_escape_and_non_utf8(tmp_path):
    (tmp_path / "bad.bin").write_bytes(b"\xff")
    registry = WorkspaceTools(tmp_path).registry()
    assert call(registry, "read_file", {"path": "../bad"}).error.code == "PATH_OUTSIDE_WORKSPACE"
    assert call(registry, "read_file", {"path": "bad.bin"}).error.code == "NOT_UTF8_TEXT"


def test_write_and_edit_file_complete_a_confirmed_change(tmp_path):
    registry = WorkspaceTools(tmp_path, confirmation=Confirm()).registry()

    written = call(
        registry,
        "write_file",
        {"path": "src/example.py", "content": "value = 1\n"},
    )
    edited = call(
        registry,
        "edit_file",
        {"path": "src/example.py", "old_text": "1", "new_text": "2"},
    )

    assert written.ok and written.data["created"] is True
    assert edited.ok and edited.data["replacements"] == 1
    assert (tmp_path / "src" / "example.py").read_text(encoding="utf-8") == "value = 2\n"


def test_edit_file_requires_one_exact_occurrence(tmp_path):
    (tmp_path / "notes.txt").write_text("same same", encoding="utf-8")
    registry = WorkspaceTools(tmp_path, confirmation=Confirm()).registry()

    missing = call(
        registry,
        "edit_file",
        {"path": "notes.txt", "old_text": "absent", "new_text": "new"},
    )
    repeated = call(
        registry,
        "edit_file",
        {"path": "notes.txt", "old_text": "same", "new_text": "new"},
    )

    assert missing.error.code == "TEXT_NOT_FOUND"
    assert repeated.error.code == "TEXT_NOT_UNIQUE"


def test_write_and_edit_require_confirmation(tmp_path):
    registry = WorkspaceTools(tmp_path).registry()

    result = call(registry, "write_file", {"path": "new.txt", "content": "content"})

    assert result.error.code == "PERMISSION_DENIED"
    assert not (tmp_path / "new.txt").exists()


def test_bash_requires_confirmation_and_returns_process_data(tmp_path):
    denied = call(WorkspaceTools(tmp_path).registry(), "bash", {"command": "printf ok"})
    assert denied.error.code == "PERMISSION_DENIED"
    result = call(
        WorkspaceTools(tmp_path, confirmation=Confirm()).registry(),
        "bash",
        {"command": "printf ok"},
    )
    assert result.ok and result.data == {
        "exit_code": 0,
        "stdout": "ok",
        "stderr": "",
        "timed_out": False,
    }


def test_bash_timeout_and_cancellation_are_distinct(tmp_path):
    registry = WorkspaceTools(tmp_path, confirmation=Confirm()).registry()
    timed = call(registry, "bash", {"command": "sleep 1", "timeout_ms": 10})
    assert timed.ok and timed.data["timed_out"] is True
    control = RunControl()
    control.cancel()
    cancelled = registry.execute(
        ToolCall("call", "bash", json.dumps({"command": "sleep 1"})), control=control
    )
    assert cancelled.error.code == "TOOL_CANCELLED"


def test_bash_drains_and_clips_large_output_without_timing_out(tmp_path):
    registry = WorkspaceTools(
        tmp_path,
        confirmation=Confirm(),
        max_command_output_bytes=4096,
    ).registry()

    result = call(
        registry,
        "bash",
        {"command": "printf %200000s x", "timeout_ms": 1000},
    )

    assert result.ok
    assert result.data["timed_out"] is False
    assert 0 < len(result.data["stdout"].encode("utf-8")) <= 4096


def test_bash_cancellation_terminates_the_process_tree(tmp_path):
    registry = WorkspaceTools(tmp_path, confirmation=Confirm()).registry()
    control = RunControl()
    outcome = []

    worker = threading.Thread(
        target=lambda: outcome.append(
            registry.execute(
                ToolCall("call", "bash", json.dumps({"command": "sleep 5 | cat"})),
                control=control,
            )
        ),
        daemon=True,
    )
    worker.start()
    time.sleep(0.05)
    control.cancel()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert outcome[0].error.code == "TOOL_CANCELLED"


def test_bash_reports_process_tree_termination_failure(tmp_path, monkeypatch):
    def fail_termination(process) -> None:
        raise ToolError(
            "PROCESS_TREE_TERMINATION_ERROR", "Could not terminate the Bash process tree."
        )

    monkeypatch.setattr("minimal_agent.workspace_tools._terminate_process_tree", fail_termination)
    control = RunControl()
    control.cancel()

    result = (
        WorkspaceTools(tmp_path, confirmation=Confirm())
        .registry()
        .execute(
            ToolCall("call", "bash", json.dumps({"command": "sleep 1 | cat"})),
            control=control,
        )
    )

    assert result.error.code == "PROCESS_TREE_TERMINATION_ERROR"
