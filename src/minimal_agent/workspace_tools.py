import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from minimal_agent.agent import ToolCall

MAX_FILE_SIZE_BYTES = 64 * 1024


class _ReadFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)


class _ListFilesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", min_length=1)


class _ToolFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class WorkspaceTools:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    def execute(self, tool_call: ToolCall) -> str:
        try:
            if tool_call.name == "read_file":
                return self._read_file(tool_call.arguments)
            if tool_call.name == "list_files":
                return self._list_files(tool_call.arguments)
            raise _ToolFailure("UNKNOWN_TOOL", f"Unknown tool: {tool_call.name}")
        except _ToolFailure as error:
            return _error_result(error.code, error.message)

    def definitions(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List the direct children of a directory in the Agent Workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative directory path. Defaults to '.'.",
                            }
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read one UTF-8 text file from the Agent Workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative path of the text file to read.",
                            }
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def _read_file(self, raw_arguments: str) -> str:
        arguments = _parse_arguments(raw_arguments, _ReadFileArguments)
        file_path = self._resolve_path(arguments.path)

        if not file_path.exists():
            raise _ToolFailure("FILE_NOT_FOUND", f"File does not exist: {arguments.path}")
        if not file_path.is_file():
            raise _ToolFailure("NOT_A_FILE", f"Path is not a file: {arguments.path}")
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise _ToolFailure(
                "FILE_TOO_LARGE",
                f"File exceeds the {MAX_FILE_SIZE_BYTES}-byte limit: {arguments.path}",
            )

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise _ToolFailure(
                "NOT_UTF8_TEXT",
                f"File is not valid UTF-8 text: {arguments.path}",
            ) from error
        except OSError as error:
            raise _ToolFailure(
                "FILE_READ_ERROR", f"Could not read file: {arguments.path}"
            ) from error

        return _success_result({"path": arguments.path, "content": content})

    def _list_files(self, raw_arguments: str) -> str:
        arguments = _parse_arguments(raw_arguments, _ListFilesArguments)
        directory = self._resolve_path(arguments.path)

        if not directory.exists():
            raise _ToolFailure(
                "DIRECTORY_NOT_FOUND",
                f"Directory does not exist: {arguments.path}",
            )
        if not directory.is_dir():
            raise _ToolFailure("NOT_A_DIRECTORY", f"Path is not a directory: {arguments.path}")

        entries = []
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.casefold())
            for child in children:
                if child.name.startswith(".") or _is_link(child):
                    continue
                if child.is_file():
                    entry_type = "file"
                elif child.is_dir():
                    entry_type = "directory"
                else:
                    continue
                entries.append({"name": child.name, "type": entry_type})
        except OSError as error:
            raise _ToolFailure(
                "DIRECTORY_READ_ERROR",
                f"Could not list directory: {arguments.path}",
            ) from error

        return _success_result({"path": arguments.path, "entries": entries})

    def _resolve_path(self, requested_path: str) -> Path:
        relative_path = Path(requested_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise _ToolFailure(
                "PATH_OUTSIDE_WORKSPACE",
                f"Path is outside the Workspace: {requested_path}",
            )
        if any(part.startswith(".") and part != "." for part in relative_path.parts):
            raise _ToolFailure("HIDDEN_PATH", f"Hidden paths are not accessible: {requested_path}")

        unresolved = self._workspace / relative_path
        current = self._workspace
        for part in relative_path.parts:
            if part in {"", "."}:
                continue
            current /= part
            if _is_link(current):
                raise _ToolFailure(
                    "LINK_NOT_ALLOWED", f"Links are not accessible: {requested_path}"
                )

        resolved = unresolved.resolve()
        if not resolved.is_relative_to(self._workspace):
            raise _ToolFailure(
                "PATH_OUTSIDE_WORKSPACE",
                f"Path is outside the Workspace: {requested_path}",
            )
        return resolved


def _parse_arguments(
    raw_arguments: str,
    arguments_type: type[_ReadFileArguments] | type[_ListFilesArguments],
) -> _ReadFileArguments | _ListFilesArguments:
    try:
        decoded = json.loads(raw_arguments)
        return arguments_type.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError) as error:
        raise _ToolFailure(
            "INVALID_ARGUMENTS", "Tool arguments do not match the schema."
        ) from error


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _success_result(data: object) -> str:
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False)


def _error_result(code: str, message: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {
                "code": code,
                "message": message,
            },
        },
        ensure_ascii=False,
    )
