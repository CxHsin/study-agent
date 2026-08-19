"""Provider-neutral tools for the local Agent Workspace."""

import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from io import BufferedReader
from pathlib import Path

from minimal_agent.tools import ConfirmationPolicy, ToolDefinition, ToolError, ToolRegistry

MAX_FILE_SIZE_BYTES = 64 * 1024
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024
DEFAULT_TIMEOUT_MS = 30_000
MAX_TIMEOUT_MS = 120_000


class WorkspaceTools:
    def __init__(
        self,
        workspace: Path,
        *,
        confirmation: ConfirmationPolicy | None = None,
        max_command_output_bytes: int = MAX_COMMAND_OUTPUT_BYTES,
        max_timeout_ms: int = MAX_TIMEOUT_MS,
    ) -> None:
        if max_command_output_bytes < 1 or max_timeout_ms < 1:
            raise ValueError("Workspace Tool limits must be positive.")
        self._workspace = workspace.resolve()
        self._confirmation = confirmation
        self._max_output = max_command_output_bytes
        self._max_timeout_ms = max_timeout_ms

    def definitions(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "name": "read_file",
                "description": "Read UTF-8 text from a file in the Agent Workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
            {
                "name": "bash",
                "description": "Run one Bash command in the Agent Workspace after confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "minLength": 1},
                        "timeout_ms": {"type": "integer", "minimum": 1},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
        )

    def registry(self) -> ToolRegistry:
        definitions = self.definitions()
        return ToolRegistry(
            [
                ToolDefinition(
                    "read_file",
                    str(definitions[0]["description"]),
                    definitions[0]["parameters"],
                    self._read_file,
                    read_only=True,
                    idempotent=True,
                    strict=False,
                ),
                ToolDefinition(
                    "bash",
                    str(definitions[1]["description"]),
                    definitions[1]["parameters"],
                    self._bash,
                    requires_confirmation=True,
                    strict=False,
                    max_result_bytes=self._max_output * 6 + 512,
                ),
            ],
            confirmation=self._confirmation,
        )

    def _read_file(self, arguments: Mapping[str, object]) -> object:
        requested = arguments["path"]
        if not isinstance(requested, str):
            raise ToolError("INVALID_ARGUMENTS", "Path must be a string.")
        path = self._resolve_path(requested)
        if not path.exists():
            raise ToolError("FILE_NOT_FOUND", f"File does not exist: {requested}")
        if not path.is_file():
            raise ToolError("NOT_A_FILE", f"Path is not a file: {requested}")
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ToolError("FILE_TOO_LARGE", f"File exceeds the size limit: {requested}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(
                "NOT_UTF8_TEXT", f"File is not valid UTF-8 text: {requested}"
            ) from error
        except OSError as error:
            raise ToolError("FILE_READ_ERROR", f"Could not read file: {requested}") from error
        start, end = arguments.get("start_line"), arguments.get("end_line")
        if start is not None or end is not None:
            lines = content.splitlines(keepends=True)
            first, last = int(start or 1) - 1, int(end) if end is not None else len(lines)
            if first >= last:
                raise ToolError("INVALID_ARGUMENTS", "end_line must be >= start_line.")
            content = "".join(lines[first:last])
        return {"path": requested, "content": content}

    def _bash(self, arguments: Mapping[str, object], control: object | None = None) -> object:
        command = arguments["command"]
        if not isinstance(command, str) or not command.strip():
            raise ToolError("INVALID_ARGUMENTS", "Command must be a non-empty string.")
        self._validate_command(command)
        executable = shutil.which("bash")
        if executable is None:
            raise ToolError("BASH_UNAVAILABLE", "No Bash executable was found.")
        timeout_ms = min(int(arguments.get("timeout_ms", DEFAULT_TIMEOUT_MS)), self._max_timeout_ms)
        env = {
            key: os.environ[key]
            for key in ("PATH", "SystemRoot", "USERPROFILE", "WSLENV")
            if key in os.environ
        }
        env["LANG"] = "C.UTF-8"
        process_options: dict[str, object] = {}
        if os.name == "nt":
            process_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            process_options["start_new_session"] = True
        try:
            process = subprocess.Popen(
                [executable, "-lc", command],
                cwd=self._workspace,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **process_options,
            )
        except OSError as error:
            raise ToolError("PROCESS_START_ERROR", "Could not start Bash.") from error
        capture = _ProcessOutput(self._max_output)
        readers = (
            threading.Thread(target=capture.drain_stdout, args=(process.stdout,), daemon=True),
            threading.Thread(target=capture.drain_stderr, args=(process.stderr,), daemon=True),
        )
        for reader in readers:
            reader.start()
        deadline = time.monotonic() + timeout_ms / 1000
        cancelled = False
        timed_out = False
        while process.poll() is None:
            if getattr(control, "stop_reason", None) is not None:
                cancelled = True
                _terminate_process_tree(process)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process_tree(process)
                break
            time.sleep(0.01)
        process.wait()
        for reader in readers:
            reader.join()
        if cancelled:
            raise ToolError("TOOL_CANCELLED", "Bash execution was cancelled.")
        return {
            "exit_code": None if timed_out else process.returncode,
            "stdout": capture.stdout,
            "stderr": capture.stderr,
            "timed_out": timed_out,
        }

    def _validate_command(self, command: str) -> None:
        if any(
            token in command
            for token in ("..", "/", "\\", ">", "<", "cd ", "curl ", "wget ", "ssh ")
        ):
            raise ToolError(
                "COMMAND_OUTSIDE_WORKSPACE",
                "Command contains a forbidden workspace escape or redirection.",
            )

    def _resolve_path(self, requested: str) -> Path:
        relative = Path(requested)
        if relative.is_absolute() or ".." in relative.parts:
            raise ToolError("PATH_OUTSIDE_WORKSPACE", f"Path is outside the Workspace: {requested}")
        if any(part.startswith(".") and part != "." for part in relative.parts):
            raise ToolError("HIDDEN_PATH", f"Hidden paths are not accessible: {requested}")
        current = self._workspace
        for part in relative.parts:
            if part not in {"", "."}:
                current /= part
                if current.is_symlink():
                    raise ToolError("LINK_NOT_ALLOWED", f"Links are not accessible: {requested}")
        resolved = (self._workspace / relative).resolve()
        if not resolved.is_relative_to(self._workspace):
            raise ToolError("PATH_OUTSIDE_WORKSPACE", f"Path is outside the Workspace: {requested}")
        return resolved


class _ProcessOutput:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._size = 0
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._lock = threading.Lock()

    def drain_stdout(self, stream: BufferedReader | None) -> None:
        self._drain(stream, self._stdout)

    def drain_stderr(self, stream: BufferedReader | None) -> None:
        self._drain(stream, self._stderr)

    def _drain(self, stream: BufferedReader | None, target: bytearray) -> None:
        if stream is None:
            return
        while chunk := stream.read(8192):
            with self._lock:
                remaining = self._limit - self._size
                if remaining > 0:
                    captured = chunk[:remaining]
                    target.extend(captured)
                    self._size += len(captured)

    @property
    def stdout(self) -> str:
        return self._stdout.decode("utf-8", errors="replace")

    @property
    def stderr(self) -> str:
        return self._stderr.decode("utf-8", errors="replace")


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        else:
            if completed.returncode != 0 and process.poll() is None:
                process.kill()
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()
