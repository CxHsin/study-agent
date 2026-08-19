import inspect
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from minimal_agent.protocol import ToolCall


class ToolError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    ok: bool
    data: object = None
    error: ToolError | None = None
    retryable: bool = False

    def to_json(self) -> str:
        payload: dict[str, object] = {"ok": self.ok}
        if self.ok:
            payload["data"] = self.data
        elif self.error:
            payload["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "retryable": self.retryable,
            }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ToolAuthorizer(Protocol):
    def authorize(self, context: "ToolExecutionContext") -> bool: ...


class ConfirmationPolicy(Protocol):
    def confirm(self, context: "ToolExecutionContext") -> bool: ...


@dataclass(frozen=True)
class ToolExecutionContext:
    run_id: str
    user_input: str
    tool_call: ToolCall
    definition: "ToolDefinition"
    arguments: Mapping[str, object]


ToolExecutor = Callable[..., object]
MAX_TOOL_RESULT_BYTES = 64 * 1024
_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, object]
    execute: ToolExecutor
    read_only: bool = False
    idempotent: bool = False
    requires_confirmation: bool = False
    strict: bool = True
    max_result_bytes: int = MAX_TOOL_RESULT_BYTES

    def __post_init__(self) -> None:
        if not _TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("Tool name must match ^[a-zA-Z0-9_-]{1,64}$.")
        if self.max_result_bytes < 1:
            raise ValueError("max_result_bytes must be positive.")
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as error:
            raise ValueError("Tool parameters must be a valid JSON Schema.") from error
        if self.parameters.get("type") != "object":
            raise ValueError("Tool parameters schema must have type=object.")
        if self.strict:
            properties = self.parameters.get("properties", {})
            required = set(self.parameters.get("required", ()))
            if self.parameters.get("additionalProperties") is not False:
                raise ValueError("Strict tool schemas must set additionalProperties=false.")
            if set(properties) != required:
                raise ValueError("Strict tool schemas must mark every property as required.")

    def definition(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
            "read_only": self.read_only,
            "idempotent": self.idempotent,
            "requires_confirmation": self.requires_confirmation,
        }


class ToolRegistry:
    def __init__(
        self,
        definitions: Iterable[ToolDefinition] = (),
        authorizer: ToolAuthorizer | None = None,
        confirmation: ConfirmationPolicy | None = None,
    ) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._authorizer = authorizer
        self._confirmation = confirmation
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def definitions(self) -> tuple[dict[str, object], ...]:
        return tuple(item.definition() for item in self._definitions.values())

    def get(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def all_read_only(self) -> bool:
        return all(definition.read_only for definition in self._definitions.values())

    def execute(self, tool_call: ToolCall, *, run_id: str = "", user_input: str = "", control: object | None = None) -> ToolResult:
        definition = self._definitions.get(tool_call.name)
        if definition is None:
            return _failure(tool_call, "UNKNOWN_TOOL", f"Unknown tool: {tool_call.name}")
        try:
            decoded = json.loads(tool_call.arguments)
            if not isinstance(decoded, dict):
                raise TypeError
            Draft202012Validator(definition.parameters).validate(decoded)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
            return _failure(
                tool_call, "INVALID_ARGUMENTS", "Tool arguments do not match the schema."
            )
        context = ToolExecutionContext(run_id, user_input, tool_call, definition, decoded)
        if self._authorizer is not None and not self._authorizer.authorize(context):
            return _failure(tool_call, "PERMISSION_DENIED", "Tool operation is not permitted.")
        if definition.requires_confirmation:
            if self._confirmation is None:
                return _failure(tool_call, "PERMISSION_DENIED", "Tool confirmation is required.")
            try:
                confirmed = self._confirmation.confirm(context)
            except Exception as error:  # noqa: BLE001 - confirmation is a safety boundary
                return _failure(
                    tool_call, "CONFIRMATION_ERROR", str(error) or "Confirmation failed."
                )
            if not confirmed:
                return _failure(tool_call, "PERMISSION_DENIED", "Tool operation was not confirmed.")
        try:
            if control is not None and len(inspect.signature(definition.execute).parameters) >= 2:
                data = definition.execute(decoded, control)
            else:
                data = definition.execute(decoded)
            encoded = json.dumps(data, ensure_ascii=False)
            if len(encoded.encode("utf-8")) > definition.max_result_bytes:
                return _failure(
                    tool_call, "TOOL_RESULT_TOO_LARGE", "Tool result exceeds the size limit."
                )
            return ToolResult(tool_call.id, tool_call.name, True, data=data)
        except ToolError as error:
            return ToolResult(
                tool_call.id, tool_call.name, False, error=error, retryable=error.retryable
            )
        except Exception as error:  # noqa: BLE001 - unknown tool failures become model-visible results
            return _failure(tool_call, "TOOL_EXCEPTION", str(error) or "Tool execution failed.")


def _failure(tool_call: ToolCall, code: str, message: str) -> ToolResult:
    return ToolResult(tool_call.id, tool_call.name, False, error=ToolError(code, message))
