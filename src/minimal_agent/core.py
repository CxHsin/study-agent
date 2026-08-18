from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from minimal_agent.events import AgentEvent, AgentEventListener
from minimal_agent.protocol import ModelAdapter, ToolCall
from minimal_agent.session import AgentSession
from minimal_agent.tools import ToolRegistry


class AgentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentResult:
    status: AgentStatus
    final_response: str | None
    steps_used: int


class AgentCore:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry | None = None,
        session: AgentSession | None = None,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._session = session or AgentSession()
        self._max_steps = max_steps
        self._listeners: list[AgentEventListener] = []
        self._sequence = 0

    def subscribe(self, listener: AgentEventListener) -> None:
        self._listeners.append(listener)

    @property
    def session(self) -> AgentSession:
        return self._session

    def prompt(self, user_input: str) -> AgentResult:
        run_id = str(uuid4())
        self._sequence = 0
        self._session.append({"role": "user", "content": user_input})
        self._emit("run_started", {"run_id": run_id})

        for step in range(1, self._max_steps + 1):
            self._emit("model_call_started", {"step": step})
            response = self._model.complete(self._session.messages)
            self._emit("model_response", {"step": step, "tool_calls": len(response.tool_calls)})
            if not response.tool_calls:
                self._session.append({"role": "assistant", "content": response.content})
                self._emit("final_response", {"step": step})
                return AgentResult(AgentStatus.COMPLETED, response.content, step)

            self._session.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls,
                }
            )
            for tool_call in response.tool_calls:
                self._emit("tool_call_requested", _tool_data(tool_call))
                result = self._tools.execute(tool_call)
                self._session.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )
                self._emit("tool_result_produced", {"tool_call_id": tool_call.id})

        self._emit("run_failed", {"reason": "max_steps"})
        return AgentResult(AgentStatus.FAILED, None, self._max_steps)

    def _emit(self, kind: str, data: dict[str, object]) -> None:
        self._sequence += 1
        event = AgentEvent(kind, data, self._sequence)
        for listener in self._listeners:
            listener(event)


def _tool_data(tool_call: ToolCall) -> dict[str, object]:
    return {"tool_call_id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}
