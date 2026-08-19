from collections.abc import Mapping, Sequence
from typing import Any, cast

from openai import OpenAI, OpenAIError

from minimal_agent.protocol import (
    ChatMessage,
    ModelError,
    ModelResponse,
    ModelUsage,
    ProviderCapabilities,
    ToolCall,
)
from minimal_agent.tools import ToolRegistry

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekAdapter:
    model_name = DEEPSEEK_MODEL

    def __init__(
        self,
        api_key: str,
        tool_definitions: Sequence[dict[str, object]] | ToolRegistry,
        client: Any | None = None,
    ) -> None:
        self._tool_definitions = (
            tuple(_to_openai_tool(item) for item in tool_definitions.definitions())
            if isinstance(tool_definitions, ToolRegistry)
            else tuple(_to_openai_tool(item) for item in tool_definitions)
        )
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=30.0,
            max_retries=2,
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=False,
            tool_calls=True,
            cancellation=False,
            usage=True,
            prompt_cache=False,
        )

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        try:
            completion = self._client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[_to_sdk_message(message) for message in messages],
                tools=self._tool_definitions,
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )
            message = completion.choices[0].message
            tool_calls = tuple(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in (message.tool_calls or ())
            )
            if message.content is None and not tool_calls:
                raise ModelError("DeepSeek returned neither text nor Tool Calls.")
            usage = _usage_from_completion(completion)
            return ModelResponse(content=message.content, tool_calls=tool_calls, usage=usage)
        except ModelError:
            raise
        except OpenAIError as error:
            raise ModelError("DeepSeek request failed.") from error
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelError("DeepSeek returned an invalid response.") from error


def _to_sdk_message(message: ChatMessage) -> dict[str, object]:
    sdk_message = dict(message)
    if message.get("role") == "context_summary":
        sdk_message["role"] = "user"
        sdk_message["content"] = "[Context summary of earlier conversation]\n" + str(
            message["content"]
        )
        sdk_message.pop("summary_id", None)
    if message.get("role") == "assistant" and "tool_calls" in message:
        tool_calls = cast(Sequence[object], message["tool_calls"])
        sdk_message["tool_calls"] = [
            {
                "id": _tool_call_field(tool_call, "id"),
                "type": "function",
                "function": {
                    "name": _tool_call_field(tool_call, "name"),
                    "arguments": _tool_call_field(tool_call, "arguments"),
                },
            }
            for tool_call in tool_calls
        ]
    return sdk_message


def _usage_from_completion(completion: Any) -> ModelUsage | None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return ModelUsage(
        input_tokens=_int_or_none(getattr(usage, "prompt_tokens", None)),
        output_tokens=_int_or_none(getattr(usage, "completion_tokens", None)),
        cached_tokens=_int_or_none(getattr(usage, "prompt_cache_hit_tokens", None)),
        estimated=False,
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _tool_call_field(call: object, field: str) -> str:
    if isinstance(call, Mapping):
        return str(call.get(field, ""))
    return str(getattr(call, field, ""))


def _to_openai_tool(definition: dict[str, object]) -> dict[str, object]:
    function = definition.get("function", definition)
    if not isinstance(function, Mapping):
        function = {}
    return {"type": "function", "function": {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object"}),
        "strict": function.get("strict", False),
    }}
