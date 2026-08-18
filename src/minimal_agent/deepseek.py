from collections.abc import Sequence
from typing import Any, cast

from openai import OpenAI, OpenAIError

from minimal_agent.protocol import ChatMessage, ModelError, ModelResponse, ToolCall
from minimal_agent.tools import ToolRegistry

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


class DeepSeekAdapter:
    def __init__(
        self,
        api_key: str,
        tool_definitions: Sequence[dict[str, object]] | ToolRegistry,
        client: Any | None = None,
    ) -> None:
        self._tool_definitions = (
            tool_definitions.definitions()
            if isinstance(tool_definitions, ToolRegistry)
            else tuple(tool_definitions)
        )
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            timeout=30.0,
            max_retries=2,
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
            return ModelResponse(content=message.content, tool_calls=tool_calls)
        except ModelError:
            raise
        except OpenAIError as error:
            raise ModelError("DeepSeek request failed.") from error
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelError("DeepSeek returned an invalid response.") from error


def _to_sdk_message(message: ChatMessage) -> dict[str, object]:
    sdk_message = dict(message)
    if message.get("role") == "assistant" and "tool_calls" in message:
        tool_calls = cast(Sequence[ToolCall], message["tool_calls"])
        sdk_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in tool_calls
        ]
    return sdk_message
