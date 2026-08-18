import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from minimal_agent.protocol import ToolCall

ToolExecutor = Callable[[ToolCall], str]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]
    execute: ToolExecutor

    def provider_definition(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, definitions: Iterable[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if not definition.name:
            raise ValueError("Tool name must not be empty.")
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def definitions(self) -> tuple[dict[str, object], ...]:
        return tuple(item.provider_definition() for item in self._definitions.values())

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def execute(self, tool_call: ToolCall) -> str:
        definition = self._definitions.get(tool_call.name)
        if definition is None:
            return _error_result("UNKNOWN_TOOL", f"Unknown tool: {tool_call.name}")
        try:
            return definition.execute(tool_call)
        except Exception as error:  # noqa: BLE001 - tool failures become model-visible results
            return _error_result("TOOL_EXCEPTION", str(error) or "Tool execution failed.")


def _error_result(code: str, message: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code, "message": message}})
