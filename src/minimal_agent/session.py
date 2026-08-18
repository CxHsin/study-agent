from collections.abc import Iterable

from minimal_agent.protocol import ChatMessage


class AgentSession:
    """In-memory conversation context used by AgentCore."""

    def __init__(self, messages: Iterable[ChatMessage] = ()) -> None:
        self._messages = list(messages)

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def append(self, message: ChatMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages.clear()
