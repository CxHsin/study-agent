"""Versioned local persistence and conservative Run recovery."""

import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from minimal_agent.protocol import ToolCall

SCHEMA_VERSION = 1


class Repository(Protocol):
    def save_session(
        self, session_id: str, system_prompt: str | None, messages: object
    ) -> None: ...

    def start_run(self, run_id: str, session_id: str, parent_run_id: str | None = None) -> None: ...

    def append_event(
        self, run_id: str, sequence: int, kind: str, data: Mapping[str, object]
    ) -> None: ...

    def finish_run(self, run_id: str, stop_reason: str, steps_used: int) -> None: ...

    def record_tool(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        status: str,
        result: str | None = None,
        *,
        idempotent: bool = False,
    ) -> None: ...

    def record_tool_lifecycle(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        result: str,
        *,
        idempotent: bool = False,
    ) -> None: ...

    def record_tool_started(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        *,
        idempotent: bool = False,
    ) -> None: ...


class Redactor:
    _secret_key = re.compile(r"(?i)(api[_-]?key|authorization|cookie|token|secret|password)")
    _secret = re.compile(
        r"(?i)(api[_-]?key|authorization|cookie|token|secret|password)\s*[:=]\s*[^,;\s]+"
    )

    def redact(self, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): "[REDACTED]"
                if self._secret_key.fullmatch(str(key))
                else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return self._secret.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
            if decoded != value:
                return json.dumps(self.redact(decoded), ensure_ascii=False, separators=(",", ":"))
            return self._secret.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        return value


def _json(value: object, redactor: Redactor) -> str:
    value = _jsonable(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
    return json.dumps(
        redactor.redact(value), ensure_ascii=False, default=repr, separators=(",", ":")
    )


def _jsonable(value: object) -> object:
    """Convert nested protocol records to JSON-compatible primitives."""
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _restore_messages(messages: object) -> tuple[dict[str, object], ...]:
    restored: list[dict[str, object]] = []
    for raw in messages if isinstance(messages, list) else ():
        if not isinstance(raw, dict):
            continue
        message = dict(raw)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            message["tool_calls"] = tuple(
                ToolCall(str(call["id"]), str(call["name"]), str(call["arguments"]))
                for call in calls
                if isinstance(call, dict) and {"id", "name", "arguments"}.issubset(call)
            )
        restored.append(message)
    return tuple(restored)


@dataclass(frozen=True)
class UnresolvedTool:
    run_id: str
    call_id: str
    name: str
    arguments: str
    idempotent: bool


class SQLiteRepository:
    def __init__(self, path: str | Path, *, redactor: Redactor | None = None) -> None:
        self.path = str(path)
        self.redactor = redactor or Redactor()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            row = self._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO metadata VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
            elif int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError(f"Unsupported repository schema version: {row[0]}")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, system_prompt TEXT, messages_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, parent_run_id TEXT,
                    status TEXT NOT NULL, stop_reason TEXT, steps_used INTEGER,
                    started_at TEXT NOT NULL, ended_at TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL, kind TEXT NOT NULL, data_json TEXT NOT NULL,
                    format_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_executions (
                    run_id TEXT NOT NULL, call_id TEXT NOT NULL, name TEXT NOT NULL,
                    arguments TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT,
                    idempotent INTEGER NOT NULL DEFAULT 0,
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                );
                """
            )

    def save_session(self, session_id: str, system_prompt: str | None, messages: object) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET system_prompt=excluded.system_prompt, messages_json=excluded.messages_json",
                (
                    session_id,
                    _json(system_prompt, self.redactor) if system_prompt is not None else None,
                    _json(messages, self.redactor),
                ),
            )

    def load_session(
        self, session_id: str
    ) -> tuple[str | None, tuple[dict[str, object], ...]] | None:
        row = self._connection.execute(
            "SELECT system_prompt, messages_json FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        system_prompt = row[0]
        if isinstance(system_prompt, str):
            try:
                system_prompt = json.loads(system_prompt)
            except json.JSONDecodeError:
                pass
        messages = json.loads(row[1])
        return system_prompt, _restore_messages(messages)

    def latest_session(self) -> tuple[str, str | None, tuple[dict[str, object], ...]] | None:
        row = self._connection.execute(
            "SELECT session_id, system_prompt, messages_json FROM sessions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        system_prompt = row[1]
        if isinstance(system_prompt, str):
            try:
                system_prompt = json.loads(system_prompt)
            except json.JSONDecodeError:
                pass
        return row[0], system_prompt, _restore_messages(json.loads(row[2]))

    def continuation_session(
        self, run_id: str
    ) -> tuple[str, str | None, tuple[dict[str, object], ...]] | None:
        row = self._connection.execute(
            "SELECT session_id, parent_run_id FROM runs WHERE run_id=? AND parent_run_id IS NOT NULL",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        loaded = self.load_session(row[0])
        if loaded is None:
            return None
        system_prompt, messages = loaded
        messages = self._inject_resolved_tool_result(row[1], run_id, messages)
        return row[0], system_prompt, messages

    def _inject_resolved_tool_result(
        self,
        parent_run_id: str,
        continuation_run_id: str,
        messages: tuple[dict[str, object], ...],
    ) -> tuple[dict[str, object], ...]:
        event = self._connection.execute(
            "SELECT data_json FROM events WHERE run_id=? AND kind='tool_resolved' ORDER BY sequence DESC LIMIT 1",
            (continuation_run_id,),
        ).fetchone()
        if event is None:
            return messages
        data = json.loads(event[0])
        call_id = data.get("call_id") if isinstance(data, dict) else None
        if not isinstance(call_id, str):
            return messages
        if any(
            item.get("role") == "tool" and item.get("tool_call_id") == call_id for item in messages
        ):
            return messages
        row = self._connection.execute(
            "SELECT result_json FROM tool_executions WHERE run_id=? AND call_id=? AND status='completed' ORDER BY id DESC LIMIT 1",
            (parent_run_id, call_id),
        ).fetchone()
        if row is None:
            return messages
        decoded = json.loads(row[0])
        content = (
            decoded if isinstance(decoded, str) else json.dumps(decoded, separators=(",", ":"))
        )
        return (*messages, {"role": "tool", "tool_call_id": call_id, "content": content})

    def recover(self) -> tuple[UnresolvedTool, ...]:
        unresolved = self.unresolved_tools()
        with self._connection:
            self._connection.execute(
                "UPDATE runs SET status='needs_resolution' WHERE status='running'"
            )
        return unresolved

    def start_run(self, run_id: str, session_id: str, parent_run_id: str | None = None) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, 'running', NULL, NULL, ?, NULL)",
                (run_id, session_id, parent_run_id, datetime.now(UTC).isoformat()),
            )

    def append_event(
        self, run_id: str, sequence: int, kind: str, data: Mapping[str, object]
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO events(run_id, sequence, kind, data_json, format_version) VALUES (?, ?, ?, ?, ?)",
                (run_id, sequence, kind, _json(data, self.redactor), SCHEMA_VERSION),
            )

    def finish_run(self, run_id: str, stop_reason: str, steps_used: int) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE runs SET status='finished', stop_reason=?, steps_used=?, ended_at=? WHERE run_id=?",
                (stop_reason, steps_used, datetime.now(UTC).isoformat(), run_id),
            )

    def record_tool(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        status: str,
        result: str | None = None,
        *,
        idempotent: bool = False,
    ) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO tool_executions(run_id, call_id, name, arguments, status, result_json, idempotent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    call_id,
                    name,
                    _json(arguments, self.redactor),
                    status,
                    _json(result, self.redactor) if result else None,
                    int(idempotent),
                ),
            )

    def record_tool_lifecycle(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        result: str,
        *,
        idempotent: bool = False,
    ) -> None:
        with self._connection:
            encoded_arguments = _json(arguments, self.redactor)
            encoded_result = _json(result, self.redactor)
            for status, payload in (
                ("requested", None),
                ("started", None),
                ("completed", encoded_result),
            ):
                self._connection.execute(
                    "INSERT INTO tool_executions(run_id, call_id, name, arguments, status, result_json, idempotent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, call_id, name, encoded_arguments, status, payload, int(idempotent)),
                )

    def record_tool_started(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        *,
        idempotent: bool = False,
    ) -> None:
        with self._connection:
            encoded_arguments = _json(arguments, self.redactor)
            for status in ("requested", "started"):
                self._connection.execute(
                    "INSERT INTO tool_executions(run_id, call_id, name, arguments, status, result_json, idempotent) VALUES (?, ?, ?, ?, ?, NULL, ?)",
                    (run_id, call_id, name, encoded_arguments, status, int(idempotent)),
                )

    def unresolved_tools(self) -> tuple[UnresolvedTool, ...]:
        rows = self._connection.execute(
            """
            SELECT run_id, call_id, name, arguments, idempotent
            FROM tool_executions AS current
            WHERE status='started'
              AND NOT EXISTS (
                SELECT 1 FROM tool_executions AS newer
                WHERE newer.run_id=current.run_id
                  AND newer.call_id=current.call_id
                  AND newer.id > current.id
              )
            """
        ).fetchall()
        return tuple(UnresolvedTool(*row) for row in rows)

    def run_status(self, run_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT status FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        return row[0] if row else None

    def event_count(self, run_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id=?", (run_id,)
        ).fetchone()
        return int(row[0])

    def event_payloads(self, run_id: str) -> tuple[dict[str, object], ...]:
        rows = self._connection.execute(
            "SELECT data_json FROM events WHERE run_id=? ORDER BY sequence", (run_id,)
        ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def create_continuation(self, parent_run_id: str, run_id: str, session_id: str) -> None:
        self.start_run(run_id, session_id, parent_run_id=parent_run_id)
        self.append_event(run_id, 1, "continuation_started", {"parent_run_id": parent_run_id})

    def retry_tool(
        self, run_id: str, call_id: str, continuation_run_id: str, session_id: str
    ) -> UnresolvedTool:
        tool = next(
            (
                item
                for item in self.unresolved_tools()
                if item.run_id == run_id and item.call_id == call_id
            ),
            None,
        )
        if tool is None:
            raise KeyError(f"Unknown unresolved Tool Execution: {run_id}/{call_id}")
        if not tool.idempotent:
            raise RuntimeError("Only idempotent Tool Executions may be retried automatically.")
        self.create_continuation(run_id, continuation_run_id, session_id)
        with self._connection:
            self._connection.execute(
                "UPDATE tool_executions SET status='retry_scheduled' WHERE run_id=? AND call_id=?",
                (run_id, call_id),
            )
        self.append_event(
            continuation_run_id,
            2,
            "tool_retry_scheduled",
            {"parent_run_id": run_id, "call_id": call_id},
        )
        return tool

    def resolve_tool(
        self,
        run_id: str,
        call_id: str,
        result: str,
        continuation_run_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if continuation_run_id is not None:
            if session_id is None:
                raise ValueError("session_id is required for a continuation Run.")
            self.create_continuation(run_id, continuation_run_id, session_id)
        with self._connection:
            self._connection.execute(
                "UPDATE tool_executions SET status='completed', result_json=? WHERE run_id=? AND call_id=?",
                (_json(result, self.redactor), run_id, call_id),
            )
        if continuation_run_id is not None:
            self.append_event(
                continuation_run_id,
                2,
                "tool_resolved",
                {"parent_run_id": run_id, "call_id": call_id, "result": result},
            )

    def close(self) -> None:
        self._connection.close()


class InMemoryRepository(SQLiteRepository):
    """Test repository with the same public behavior and no filesystem state."""

    def __init__(self, *, redactor: Redactor | None = None) -> None:
        super().__init__(":memory:", redactor=redactor)
