import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from minimal_agent.agent import ToolCall
from minimal_agent.recovery import ToolRetryPolicy


class _RecordIntentArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_id: str = Field(min_length=1)
    value: str = Field(min_length=1)


class IdempotentIntentRecorder:
    """A deterministic side-effect simulator for Runtime recovery experiments."""

    is_idempotent = True
    retry_allowed = True

    def __init__(self) -> None:
        self.records: dict[str, str] = {}
        self.execution_count = 0

    def retry_policy(self, tool_name: str) -> ToolRetryPolicy:
        return ToolRetryPolicy(
            is_idempotent=tool_name == "record_intent",
            retry_allowed=tool_name == "record_intent",
        )

    def execute(self, tool_call: ToolCall) -> str:
        if tool_call.name != "record_intent":
            return _error("UNKNOWN_TOOL")
        try:
            arguments = _RecordIntentArguments.model_validate(json.loads(tool_call.arguments))
        except (json.JSONDecodeError, ValidationError):
            return _error("INVALID_ARGUMENTS")

        self.execution_count += 1
        created = arguments.intent_id not in self.records
        self.records.setdefault(arguments.intent_id, arguments.value)
        return json.dumps(
            {
                "ok": True,
                "data": {
                    "intent_id": arguments.intent_id,
                    "created": created,
                },
            }
        )


def _error(code: str) -> str:
    return json.dumps({"ok": False, "error": {"code": code}})
