from collections.abc import Iterable

from minimal_agent.protocol import ChatMessage


class AgentSession:
    """In-memory conversation context used by AgentCore."""

    def __init__(
        self, messages: Iterable[ChatMessage] = (), *, system_prompt: str | None = None
    ) -> None:
        self._messages = list(messages)
        self._system_prompt = system_prompt

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def append(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()
