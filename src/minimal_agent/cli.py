import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from minimal_agent.core import AgentCore
from minimal_agent.deepseek import DeepSeekAdapter
from minimal_agent.events import AgentEvent
from minimal_agent.persistence import SQLiteRepository
from minimal_agent.workspace_tools import WorkspaceTools

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigurationError(RuntimeError):
    pass


def run_console(
    core: AgentCore,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
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
        try:
            result = core.prompt(user_input)
        except KeyboardInterrupt:
            return 0
        if result.stop_reason.value == "final":
            output_fn(f"Agent> {result.final_response or ''}")
        else:
            output_fn(f"Agent> Task stopped: {result.stop_reason.value}")


def create_core() -> AgentCore:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        raise ConfigurationError("DEEPSEEK_API_KEY is missing from the environment or .env file.")

    workspace = PROJECT_ROOT / "workspace"
    workspace.mkdir(exist_ok=True)
    tools = WorkspaceTools(workspace).registry()
    model = DeepSeekAdapter(api_key=api_key, tool_definitions=tools)
    repository = SQLiteRepository(PROJECT_ROOT / ".minimal-agent" / "agent.sqlite")
    repository.recover()
    return AgentCore(model=model, tools=tools, repository=repository)


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
    try:
        core = create_core()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    return run_console(core)
