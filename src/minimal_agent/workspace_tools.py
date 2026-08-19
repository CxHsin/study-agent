from collections.abc import Mapping
from pathlib import Path

from minimal_agent.tools import ToolDefinition, ToolError, ToolRegistry

MAX_FILE_SIZE_BYTES = 64 * 1024


class WorkspaceTools:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    def definitions(self) -> tuple[dict[str, object], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List direct children of a directory in the Agent Workspace. Use null for the workspace root.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": ["string", "null"]}},
                        "required": ["path"],
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
                        "properties": {"path": {"type": "string", "minLength": 1}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def registry(self) -> ToolRegistry:
        definitions = []
        for provider_definition in self.definitions():
            function = provider_definition["function"]
            name = str(function["name"])
            executor = self._list_files if name == "list_files" else self._read_file
            definitions.append(
                ToolDefinition(
                    name=name,
                    description=str(function["description"]),
                    parameters=function["parameters"],
                    execute=executor,
                    read_only=True,
                    idempotent=True,
                )
            )
        return ToolRegistry(definitions)

    def _read_file(self, arguments: Mapping[str, object]) -> object:
        requested_path = arguments["path"]
        if not isinstance(requested_path, str):
            raise ToolError("INVALID_ARGUMENTS", "Path must be a string.")
        file_path = self._resolve_path(requested_path)
        if not file_path.exists():
            raise ToolError("FILE_NOT_FOUND", f"File does not exist: {requested_path}")
        if not file_path.is_file():
            raise ToolError("NOT_A_FILE", f"Path is not a file: {requested_path}")
        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ToolError("FILE_TOO_LARGE", f"File exceeds the size limit: {requested_path}")
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError(
                "NOT_UTF8_TEXT", f"File is not valid UTF-8 text: {requested_path}"
            ) from error
        except OSError as error:
            raise ToolError("FILE_READ_ERROR", f"Could not read file: {requested_path}") from error
        return {"path": requested_path, "content": content}

    def _list_files(self, arguments: Mapping[str, object]) -> object:
        requested_path = arguments["path"]
        path = "." if requested_path is None else requested_path
        if not isinstance(path, str):
            raise ToolError("INVALID_ARGUMENTS", "Path must be a string or null.")
        directory = self._resolve_path(path)
        if not directory.exists():
            raise ToolError("DIRECTORY_NOT_FOUND", f"Directory does not exist: {path}")
        if not directory.is_dir():
            raise ToolError("NOT_A_DIRECTORY", f"Path is not a directory: {path}")
        try:
            entries = []
            for child in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
                if child.name.startswith(".") or _is_link(child):
                    continue
                if child.is_file():
                    entries.append({"name": child.name, "type": "file"})
                elif child.is_dir():
                    entries.append({"name": child.name, "type": "directory"})
        except OSError as error:
            raise ToolError("DIRECTORY_READ_ERROR", f"Could not list directory: {path}") from error
        return {"path": path, "entries": entries}

    def _resolve_path(self, requested_path: str) -> Path:
        relative_path = Path(requested_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ToolError(
                "PATH_OUTSIDE_WORKSPACE", f"Path is outside the Workspace: {requested_path}"
            )
        if any(part.startswith(".") and part != "." for part in relative_path.parts):
            raise ToolError("HIDDEN_PATH", f"Hidden paths are not accessible: {requested_path}")
        current = self._workspace
        for part in relative_path.parts:
            if part not in {"", "."}:
                current /= part
                if _is_link(current):
                    raise ToolError(
                        "LINK_NOT_ALLOWED", f"Links are not accessible: {requested_path}"
                    )
        resolved = (self._workspace / relative_path).resolve()
        if not resolved.is_relative_to(self._workspace):
            raise ToolError(
                "PATH_OUTSIDE_WORKSPACE", f"Path is outside the Workspace: {requested_path}"
            )
        return resolved


def _is_link(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()
