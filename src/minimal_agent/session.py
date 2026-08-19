from collections.abc import Iterable
from threading import Lock
from uuid import uuid4

from minimal_agent.protocol import ChatMessage, LegacyMessage, normalize_messages


class AgentSession:
    """In-memory conversation context used by AgentCore."""

    def __init__(
        self,
        messages: Iterable[ChatMessage | LegacyMessage] = (),
        *,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._messages = list(normalize_messages(messages))
        self._system_prompt = system_prompt
        self.session_id = session_id or str(uuid4())
        self._run_lock = Lock()

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    @property
    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def append(self, message: ChatMessage) -> None:
        self._messages = list(normalize_messages((*self._messages, message)))

    def clear(self) -> None:
        self._messages.clear()

    def restore(
        self,
        messages: Iterable[ChatMessage | LegacyMessage],
        *,
        system_prompt: str | None = None,
    ) -> None:
        """Replace in-memory history with a repository snapshot."""
        self._messages = list(normalize_messages(messages))
        self._system_prompt = system_prompt

    def try_acquire_run(self) -> bool:
        return self._run_lock.acquire(blocking=False)

    def release_run(self) -> None:
        self._run_lock.release()
