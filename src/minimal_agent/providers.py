"""Optional Provider Adapter implementations using injected SDK clients."""

from collections.abc import Mapping
from typing import Any, cast

from minimal_agent.deepseek import _usage_from_completion
from minimal_agent.protocol import (
    ChatMessage,
    ModelError,
    ModelResponse,
    ModelUsage,
    ProviderCapabilities,
    ToolCall,
)
from minimal_agent.tools import ToolRegistry


class OpenAIAdapter:
    def __init__(
        self,
        api_key: str,
        tool_definitions: list[dict[str, object]] | ToolRegistry,
        *,
        model: str = "gpt-4o-mini",
        client: Any | None = None,
    ) -> None:
        self.model_name = model
        self._tools = (
            tuple(_to_openai_tool(item) for item in tool_definitions.definitions())
            if isinstance(tool_definitions, ToolRegistry)
            else tuple(_to_openai_tool(item) for item in tool_definitions)
        )
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
        self._client = client

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=False, tool_calls=True, usage=True, prompt_cache=False
        )

    def complete(self, messages: list[ChatMessage] | tuple[ChatMessage, ...]) -> ModelResponse:
        try:
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=[_to_openai_message(message) for message in messages],
                tools=self._tools,
                tool_choice="auto",
            )
            message = completion.choices[0].message
            calls = tuple(
                ToolCall(call.id, call.function.name, call.function.arguments)
                for call in (message.tool_calls or ())
            )
            if message.content is None and not calls:
                raise ModelError("OpenAI returned neither text nor Tool Calls.")
            return ModelResponse(message.content, calls, usage=_usage_from_completion(completion))
        except ModelError:
            raise
        except Exception as error:
            raise ModelError("OpenAI request failed.") from error


class AnthropicAdapter:
    def __init__(
        self,
        api_key: str,
        tool_definitions: list[dict[str, object]] | ToolRegistry,
        *,
        model: str = "claude-3-5-sonnet-latest",
        client: Any | None = None,
    ) -> None:
        self.model_name = model
        definitions = (
            tool_definitions.definitions()
            if isinstance(tool_definitions, ToolRegistry)
            else tuple(tool_definitions)
        )
        self._tools = tuple(_to_anthropic_tool(item) for item in definitions)
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as error:
                raise ModelError("Anthropic SDK is not installed.") from error
            client = Anthropic(api_key=api_key)
        self._client = client

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=False, tool_calls=True, usage=True, prompt_cache=False
        )

    def complete(self, messages: list[ChatMessage] | tuple[ChatMessage, ...]) -> ModelResponse:
        try:
            system, converted = _to_anthropic_messages(messages)
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system,
                messages=converted,
                tools=self._tools,
            )
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    text_parts.append(str(getattr(block, "text", "")))
                elif getattr(block, "type", None) == "tool_use":
                    calls.append(
                        ToolCall(
                            str(block.id),
                            str(block.name),
                            _json_arguments(getattr(block, "input", {})),
                        )
                    )
            if not text_parts and not calls:
                raise ModelError("Anthropic returned neither text nor Tool Calls.")
            usage_data = getattr(response, "usage", None)
            usage = (
                ModelUsage(
                    input_tokens=getattr(usage_data, "input_tokens", None),
                    output_tokens=getattr(usage_data, "output_tokens", None),
                )
                if usage_data is not None
                else None
            )
            return ModelResponse("".join(text_parts) or None, tuple(calls), usage=usage)
        except ModelError:
            raise
        except Exception as error:
            raise ModelError("Anthropic request failed.") from error


def _to_openai_message(message: ChatMessage) -> dict[str, object]:
    converted = dict(message)
    if converted.get("role") == "context_summary":
        converted["role"] = "user"
        converted.pop("summary_id", None)
    if converted.get("role") == "assistant" and "tool_calls" in converted:
        calls = converted["tool_calls"]
        converted["tool_calls"] = [
            {
                "id": _tool_call_field(call, "id"),
                "type": "function",
                "function": {
                    "name": _tool_call_field(call, "name"),
                    "arguments": _tool_call_field(call, "arguments"),
                },
            }
            for call in calls
        ]
    return converted


def _to_anthropic_tool(definition: dict[str, object]) -> dict[str, object]:
    function = cast(dict[str, object], definition.get("function", definition))
    return {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {"type": "object"}),
    }


def _to_openai_tool(definition: dict[str, object]) -> dict[str, object]:
    function = cast(dict[str, object], definition.get("function", definition))
    return {"type": "function", "function": {
        "name": function.get("name", ""),
        "description": function.get("description", ""),
        "parameters": function.get("parameters", {"type": "object"}),
        "strict": function.get("strict", False),
    }}


def _to_anthropic_messages(
    messages: tuple[ChatMessage, ...] | list[ChatMessage],
) -> tuple[str | None, list[dict[str, object]]]:
    system: str | None = None
    converted: list[dict[str, object]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "system":
            system = str(message.get("content", ""))
            continue
        if role == "context_summary":
            role = "user"
        if role == "assistant" and "tool_calls" in message:
            blocks: list[dict[str, object]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": _tool_call_field(call, "id"),
                    "name": _tool_call_field(call, "name"),
                    "input": _decoded_arguments(_tool_call_field(call, "arguments")),
                }
                for call in cast(tuple[ToolCall, ...], message["tool_calls"])
            )
            converted.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": message.get("content", ""),
                        }
                    ],
                }
            )
        else:
            converted.append({"role": role, "content": message.get("content", "")})
    return system, converted


def _json_arguments(value: object) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _decoded_arguments(value: str) -> object:
    import json

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _tool_call_field(call: object, field: str) -> str:
    if isinstance(call, Mapping):
        return str(call.get(field, ""))
    return str(getattr(call, field, ""))
