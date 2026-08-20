import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Condition

from dotenv import load_dotenv

from minimal_agent.core import AgentCore
from minimal_agent.deepseek import DeepSeekAdapter
from minimal_agent.events import AgentEvent, EventKind
from minimal_agent.persistence import SQLiteRepository
from minimal_agent.provider_client import ProviderClient
from minimal_agent.recovery import RecoveryCoordinator, RecoveryPlan
from minimal_agent.workspace_tools import WorkspaceTools

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConsoleRuntime:
    core: AgentCore
    recovery: RecoveryCoordinator
    recovery_plan: RecoveryPlan


class ConsoleConfirmation:
    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
        *,
        coordinated: bool = False,
    ) -> None:
        self._input = input_fn
        self._output = output_fn
        self._coordinated = coordinated
        self._condition = Condition()
        self._decisions: dict[tuple[str, str], bool] = {}

    def confirm(self, context) -> bool:
        if self._coordinated:
            key = (context.run_id, context.tool_call.id)
            with self._condition:
                self._condition.wait_for(lambda: key in self._decisions)
                return self._decisions.pop(key)
        return self._prompt(context.tool_call.name, context.arguments)

    def prompt_and_resolve(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: object,
        input_fn: Callable[[str], str],
        output_fn: Callable[[str], None],
    ) -> bool:
        try:
            allowed = _ask_confirmation(name, arguments, input_fn, output_fn)
        except EOFError:
            allowed = False
            self.resolve(run_id, call_id, allowed)
            return allowed
        except KeyboardInterrupt:
            self.resolve(run_id, call_id, False)
            raise
        self.resolve(run_id, call_id, allowed)
        return allowed

    def resolve(self, run_id: str, call_id: str, allowed: bool) -> None:
        with self._condition:
            self._decisions[(run_id, call_id)] = allowed
            self._condition.notify_all()

    def _prompt(self, name: str, arguments: object) -> bool:
        try:
            return _ask_confirmation(name, arguments, self._input, self._output)
        except (EOFError, KeyboardInterrupt):
            return False


def _ask_confirmation(
    name: str,
    arguments: object,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> bool:
    rendered = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    )
    output_fn(f"Confirm> {name} {rendered}")
    return input_fn("Allow? [y/N] ").strip().lower() in {"y", "yes"}


def run_console(
    core: AgentCore,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    *,
    confirmation: ConsoleConfirmation | None = None,
) -> int:
    events: list[AgentEvent] = []
    core.subscribe(events.append)
    while True:
        try:
            user_input = input_fn("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0

        if not user_input:
            continue
        if user_input == "/exit":
            return 0
        if user_input == "/reset":
            core.session.clear()
            events.clear()
            output_fn("Session reset.")
            continue
        if user_input == "/trace last":
            _display_last_trace(events, output_fn)
            continue

        events.clear()
        streamed_content = False
        try:
            for event in core.stream(user_input):
                if event.kind is EventKind.MODEL_CONTENT_DELTA:
                    streamed_content = True
                    output_fn(f"Agent> {event.data['content_delta']}")
                elif event.kind is EventKind.TOOL_CALL_REQUESTED:
                    output_fn(f"Tool> {event.data['name']} {event.data['arguments']}")
                elif event.kind is EventKind.TOOL_CONFIRMATION_REQUESTED:
                    if confirmation is not None:
                        confirmation.prompt_and_resolve(
                            str(event.data["run_id"]),
                            str(event.data["tool_call_id"]),
                            str(event.data["name"]),
                            event.data["arguments"],
                            input_fn,
                            output_fn,
                        )
                elif event.kind is EventKind.TOOL_RESULT_PRODUCED:
                    status = "ok" if event.data["success"] else "failed"
                    output_fn(f"Tool> {event.data['name']} {status}")
                elif event.kind is EventKind.FINAL_RESPONSE and not streamed_content:
                    output_fn(f"Agent> {event.data['content']}")
                elif event.kind in {EventKind.RUN_STOPPED, EventKind.RUN_ERROR}:
                    output_fn(f"Agent> Task stopped: {event.data['stop_reason']}")
        except KeyboardInterrupt:
            output_fn("Agent> Run cancelled.")
            continue


def create_core(confirmation: ConsoleConfirmation | None = None) -> AgentCore:
    return create_runtime(confirmation).core


def create_runtime(confirmation: ConsoleConfirmation | None = None) -> ConsoleRuntime:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        raise ConfigurationError("DEEPSEEK_API_KEY is missing from the environment or .env file.")

    workspace = PROJECT_ROOT / "workspace"
    workspace.mkdir(exist_ok=True)
    tools = WorkspaceTools(workspace, confirmation=confirmation or ConsoleConfirmation()).registry()
    model = ProviderClient(DeepSeekAdapter(api_key=api_key))
    storage = PROJECT_ROOT / ".minimal-agent"
    storage.mkdir(exist_ok=True)
    repository = SQLiteRepository(storage / "agent.sqlite")
    recovery = RecoveryCoordinator(repository)
    recovery_plan = recovery.recover()
    saved = repository.latest_session()
    session = None
    if saved is not None:
        from minimal_agent.session import AgentSession

        session = AgentSession(saved[2], system_prompt=saved[1], session_id=saved[0])
    core = AgentCore(model=model, tools=tools, session=session, repository=repository)
    return ConsoleRuntime(core, recovery, recovery_plan)


def _display_last_trace(events: list[AgentEvent], output_fn: Callable[[str], None]) -> None:
    if not events:
        output_fn("Trace> No runs recorded.")
        return

    output_fn("Trace> Last run:")
    for event in events:
        details = []
        for key, value in event.data.items():
            rendered = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else str(value)
            )
            details.append(f"{key}={rendered}")
        suffix = f" {' '.join(details)}" if details else ""
        output_fn(f"  {event.sequence} {event.kind}{suffix}")


def main() -> int:
    confirmation = ConsoleConfirmation(coordinated=True)
    try:
        runtime = create_runtime(confirmation)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    for tool in runtime.recovery_plan.unresolved:
        print(f"Recovery> {tool.run_id}/{tool.call_id} {tool.name} requires Tool Resolution.")
    return run_console(runtime.core, confirmation=confirmation)
