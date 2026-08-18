import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv

from minimal_agent.agent import AgentSession, TaskStatus, TraceEvent, TraceSink
from minimal_agent.deepseek import DeepSeekAdapter
from minimal_agent.session_store import SQLiteSessionStore
from minimal_agent.trace import JsonlTraceStore
from minimal_agent.workspace_tools import WorkspaceTools

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACE_FILE = PROJECT_ROOT / ".minimal-agent" / "traces.jsonl"
SESSION_DATABASE = PROJECT_ROOT / ".minimal-agent" / "sessions.sqlite3"


class ConfigurationError(RuntimeError):
    pass


class TraceReader(Protocol):
    def read_last_run(self) -> list[TraceEvent]: ...


def run_console(
    session: AgentSession,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    trace_reader: TraceReader | None = None,
) -> int:
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
            session.reset()
            output_fn("Session reset.")
            continue
        if user_input == "/trace last":
            _display_last_trace(trace_reader, output_fn)
            continue

        try:
            result = session.submit(user_input)
        except KeyboardInterrupt:
            return 0
        if result.status is TaskStatus.COMPLETED:
            output_fn(f"Agent> {result.final_response or ''}")
        else:
            output_fn(f"Agent> Task failed: {result.stop_reason.value}")


def create_session(trace_sink: TraceSink | None = None) -> AgentSession:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key is None or not api_key.strip():
        raise ConfigurationError("DEEPSEEK_API_KEY is missing from the environment or .env file.")

    workspace = PROJECT_ROOT / "workspace"
    workspace.mkdir(exist_ok=True)
    tools = WorkspaceTools(workspace)
    session_store = SQLiteSessionStore(SESSION_DATABASE)
    session_id = session_store.latest_session_id()
    model = DeepSeekAdapter(
        api_key=api_key,
        tool_definitions=tools.definitions(),
    )
    return AgentSession(
        model=model,
        tools=tools,
        trace_sink=trace_sink,
        message_store=session_store,
        session_id=session_id,
    )


def _display_last_trace(
    trace_reader: TraceReader | None,
    output_fn: Callable[[str], None],
) -> None:
    events = trace_reader.read_last_run() if trace_reader is not None else []
    if not events:
        output_fn("Trace> No runs recorded.")
        return

    output_fn("Trace> Last run:")
    for event in events:
        output_fn(_format_trace_event(event))


def _format_trace_event(event: TraceEvent) -> str:
    details = []
    for key, value in event.data.items():
        if key == "tool_call_id":
            continue
        rendered = (
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            if isinstance(value, (dict, list))
            else str(value)
        )
        details.append(f"{key}={rendered}")
    if event.duration_ms is not None:
        details.append(f"duration_ms={event.duration_ms:.2f}")
    suffix = f" {' '.join(details)}" if details else ""
    return f"  {event.sequence} {event.kind}{suffix}"


def main() -> int:
    trace_store = JsonlTraceStore(TRACE_FILE)
    try:
        session = create_session(trace_sink=trace_store)
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1
    return run_console(session, trace_reader=trace_store)
