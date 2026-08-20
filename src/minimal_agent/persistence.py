"""Versioned local persistence and conservative Run recovery."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from minimal_agent.cost import UsageRecord

if TYPE_CHECKING:
    from minimal_agent.recovery import ContinuationRun

from minimal_agent.protocol import (
    AssistantMessage,
    ChatMessage,
    ContextSummaryMessage,
    SystemMessage,
    ToolResult,
    ToolResultMessage,
    UserMessage,
    message_to_dict,
    normalize_messages,
)

SCHEMA_VERSION = 1


class SessionRepository(Protocol):
    def save_session(
        self, session_id: str, system_prompt: str | None, messages: object
    ) -> None: ...

    def load_session(
        self, session_id: str
    ) -> tuple[str | None, tuple[ChatMessage, ...]] | None: ...

    def latest_session(
        self,
    ) -> tuple[str, str | None, tuple[ChatMessage, ...]] | None: ...


class RunRepository(Protocol):
    def start_run(self, run_id: str, session_id: str, parent_run_id: str | None = None) -> None: ...

    def append_event(
        self, run_id: str, sequence: int, kind: str, data: Mapping[str, object]
    ) -> None: ...

    def finish_run(self, run_id: str, stop_reason: str, steps_used: int) -> None: ...

    def record_usage(self, run_id: str, usage: UsageRecord) -> None: ...

    def usage(self, run_id: str) -> tuple[UsageRecord, ...]: ...

    def continuation_session(
        self, run_id: str
    ) -> tuple[str, str | None, tuple[ChatMessage, ...]] | None: ...


class ToolExecutionRepository(Protocol):
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

    def record_tool_started(
        self,
        run_id: str,
        call_id: str,
        name: str,
        arguments: str,
        *,
        idempotent: bool = False,
    ) -> None: ...


class ToolLedgerRepository(ToolExecutionRepository, Protocol):
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

    def unresolved_tools(self) -> tuple[UnresolvedTool, ...]: ...

    def retry_tool(
        self, run_id: str, call_id: str, continuation_run_id: str, session_id: str
    ) -> UnresolvedTool: ...

    def resolve_tool(
        self,
        run_id: str,
        call_id: str,
        result: str,
        continuation_run_id: str | None = None,
        session_id: str | None = None,
    ) -> None: ...


class Repository(SessionRepository, RunRepository, ToolLedgerRepository, Protocol):
    """Composite persistence seam used by the compatibility facade.

    The interfaces are split by responsibility even though SQLite currently
    provides all three adapters through one connection.
    """


@dataclass(frozen=True)
class RepositoryAdapters:
    sessions: SessionRepository | None
    runs: RunRepository | None
    tools: ToolExecutionRepository | None


def repository_adapters(repository: Repository | None) -> RepositoryAdapters:
    if repository is None:
        return RepositoryAdapters(None, None, None)
    return RepositoryAdapters(
        getattr(repository, "sessions", None)
        or (repository if callable(getattr(repository, "save_session", None)) else None),
        getattr(repository, "runs", None)
        or (repository if callable(getattr(repository, "start_run", None)) else None),
        getattr(repository, "tools", None)
        or (repository if callable(getattr(repository, "record_tool", None)) else None),
    )


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
    if isinstance(
        value,
        (SystemMessage, UserMessage, AssistantMessage, ToolResultMessage, ContextSummaryMessage),
    ):
        return _jsonable(message_to_dict(value))
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _restore_messages(messages: object) -> tuple[ChatMessage, ...]:
    if not isinstance(messages, list):
        raise TypeError("Session messages must be a JSON array.")
    if not all(isinstance(message, dict) for message in messages):
        raise TypeError("Session messages must contain only objects.")
    return normalize_messages(messages)


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
        self._sessions = SQLiteSessionRepository(self)
        self._runs = SQLiteRunRepository(self)
        self._tools = SQLiteToolLedgerRepository(self)

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
                CREATE TABLE IF NOT EXISTS model_usages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL, call_id TEXT NOT NULL UNIQUE,
                    step INTEGER NOT NULL, purpose TEXT NOT NULL,
                    status TEXT NOT NULL, error_code TEXT,
                    input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER,
                    latency_ms REAL, cost REAL, source TEXT NOT NULL,
                    cache_hit_source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_executions (
                    run_id TEXT NOT NULL, call_id TEXT NOT NULL, name TEXT NOT NULL,
                    arguments TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT,
                    idempotent INTEGER NOT NULL DEFAULT 0,
                    id INTEGER PRIMARY KEY AUTOINCREMENT
                );
                """
            )

    @property
    def sessions(self) -> SQLiteSessionRepository:
        return self._sessions

    @property
    def runs(self) -> SQLiteRunRepository:
        return self._runs

    @property
    def tools(self) -> SQLiteToolLedgerRepository:
        return self._tools

    def save_session(self, session_id: str, system_prompt: str | None, messages: object) -> None:
        stored_messages = (
            normalize_messages(messages) if isinstance(messages, (list, tuple)) else messages
        )
        with self._connection:
            self._connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET system_prompt=excluded.system_prompt, messages_json=excluded.messages_json",
                (
                    session_id,
                    _json(system_prompt, self.redactor) if system_prompt is not None else None,
                    _json(stored_messages, self.redactor),
                ),
            )

    def load_session(self, session_id: str) -> tuple[str | None, tuple[ChatMessage, ...]] | None:
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

    def latest_session(self) -> tuple[str, str | None, tuple[ChatMessage, ...]] | None:
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
    ) -> tuple[str, str | None, tuple[ChatMessage, ...]] | None:
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
        messages: tuple[ChatMessage, ...],
    ) -> tuple[ChatMessage, ...]:
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
            isinstance(item, ToolResultMessage) and item.result.tool_call_id == call_id
            for item in messages
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
        tool_name = next(
            (
                call.name
                for message in messages
                if isinstance(message, AssistantMessage)
                for call in message.tool_calls
                if call.id == call_id
            ),
            "",
        )
        decoded_result = json.loads(content)
        ok = bool(decoded_result.get("ok", True)) if isinstance(decoded_result, dict) else True
        data = decoded_result.get("data") if isinstance(decoded_result, dict) else decoded_result
        return (*messages, ToolResultMessage(ToolResult(call_id, tool_name, ok, data=data)))

    def recover(self) -> tuple[UnresolvedTool, ...]:
        from minimal_agent.recovery import RecoveryCoordinator

        return RecoveryCoordinator(self).recover().unresolved

    def mark_interrupted_runs(self) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE runs SET status='needs_resolution' WHERE status='running'"
            )

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

    def record_usage(self, run_id: str, usage: UsageRecord) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO model_usages(
                    run_id, call_id, step, purpose, status, error_code,
                    input_tokens, output_tokens, cached_tokens,
                    latency_ms, cost, source, cache_hit_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    step=excluded.step,
                    purpose=excluded.purpose,
                    status=excluded.status,
                    error_code=excluded.error_code,
                    input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens,
                    cached_tokens=excluded.cached_tokens,
                    latency_ms=excluded.latency_ms,
                    cost=excluded.cost,
                    source=excluded.source,
                    cache_hit_source=excluded.cache_hit_source
                """,
                (
                    run_id,
                    usage.call_id,
                    usage.step,
                    usage.purpose.value,
                    usage.status.value,
                    usage.error_code,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cached_tokens,
                    usage.latency_ms,
                    usage.cost,
                    usage.source,
                    usage.cache_hit_source,
                ),
            )

    def usage(self, run_id: str) -> tuple[UsageRecord, ...]:
        rows = self._connection.execute(
            """
            SELECT input_tokens, output_tokens, cached_tokens, latency_ms, cost,
                   source, cache_hit_source, step, call_id, purpose, status, error_code
            FROM model_usages WHERE run_id=? ORDER BY id
            """,
            (run_id,),
        ).fetchall()
        return tuple(UsageRecord(*row) for row in rows)

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
        with self._connection:
            self._insert_continuation(parent_run_id, run_id, session_id)

    def retry_tool(
        self, run_id: str, call_id: str, continuation_run_id: str, session_id: str
    ) -> UnresolvedTool:
        from minimal_agent.recovery import ContinuationRun, RecoveryCoordinator

        return RecoveryCoordinator(self).retry_tool(
            run_id,
            call_id,
            ContinuationRun(continuation_run_id, session_id),
        )

    def schedule_tool_retry(
        self,
        tool: UnresolvedTool,
        continuation: ContinuationRun,
    ) -> None:
        with self._connection:
            self._insert_continuation(
                tool.run_id,
                continuation.run_id,
                continuation.session_id,
            )
            self._connection.execute(
                "UPDATE tool_executions SET status='retry_scheduled' WHERE run_id=? AND call_id=?",
                (tool.run_id, tool.call_id),
            )
            self._connection.execute(
                "INSERT INTO events(run_id, sequence, kind, data_json, format_version) VALUES (?, 2, 'tool_retry_scheduled', ?, ?)",
                (
                    continuation.run_id,
                    _json({"parent_run_id": tool.run_id, "call_id": tool.call_id}, self.redactor),
                    SCHEMA_VERSION,
                ),
            )

    def resolve_tool(
        self,
        run_id: str,
        call_id: str,
        result: str,
        continuation_run_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        from minimal_agent.recovery import ContinuationRun, RecoveryCoordinator

        if (continuation_run_id is None) != (session_id is None):
            raise ValueError("continuation_run_id and session_id must be provided together.")
        RecoveryCoordinator(self).resolve_tool(
            run_id,
            call_id,
            result,
            ContinuationRun(continuation_run_id, session_id)
            if continuation_run_id is not None and session_id is not None
            else None,
        )

    def store_tool_resolution(
        self,
        tool: UnresolvedTool,
        result: str,
        continuation: ContinuationRun | None,
    ) -> None:
        with self._connection:
            if continuation is not None:
                self._insert_continuation(
                    tool.run_id,
                    continuation.run_id,
                    continuation.session_id,
                )
            self._connection.execute(
                "UPDATE tool_executions SET status='completed', result_json=? WHERE run_id=? AND call_id=?",
                (_json(result, self.redactor), tool.run_id, tool.call_id),
            )
            if continuation is not None:
                self._connection.execute(
                    "INSERT INTO events(run_id, sequence, kind, data_json, format_version) VALUES (?, 2, 'tool_resolved', ?, ?)",
                    (
                        continuation.run_id,
                        _json(
                            {
                                "parent_run_id": tool.run_id,
                                "call_id": tool.call_id,
                                "result": result,
                            },
                            self.redactor,
                        ),
                        SCHEMA_VERSION,
                    ),
                )

    def _insert_continuation(
        self,
        parent_run_id: str,
        run_id: str,
        session_id: str,
    ) -> None:
        self._connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, 'running', NULL, NULL, ?, NULL)",
            (run_id, session_id, parent_run_id, datetime.now(UTC).isoformat()),
        )
        self._connection.execute(
            "INSERT INTO events(run_id, sequence, kind, data_json, format_version) VALUES (?, 1, 'continuation_started', ?, ?)",
            (
                run_id,
                _json({"parent_run_id": parent_run_id}, self.redactor),
                SCHEMA_VERSION,
            ),
        )

    def close(self) -> None:
        self._connection.close()


class SQLiteSessionRepository:
    """Session adapter backed by an existing SQLite repository connection."""

    def __init__(self, backend: SQLiteRepository) -> None:
        self._backend = backend

    def save_session(self, session_id: str, system_prompt: str | None, messages: object) -> None:
        self._backend.save_session(session_id, system_prompt, messages)

    def load_session(self, session_id: str) -> tuple[str | None, tuple[ChatMessage, ...]] | None:
        return self._backend.load_session(session_id)

    def latest_session(self) -> tuple[str, str | None, tuple[ChatMessage, ...]] | None:
        return self._backend.latest_session()


class SQLiteRunRepository:
    """Run and event adapter backed by an existing SQLite repository connection."""

    def __init__(self, backend: SQLiteRepository) -> None:
        self._backend = backend

    def start_run(self, run_id: str, session_id: str, parent_run_id: str | None = None) -> None:
        self._backend.start_run(run_id, session_id, parent_run_id)

    def append_event(
        self, run_id: str, sequence: int, kind: str, data: Mapping[str, object]
    ) -> None:
        self._backend.append_event(run_id, sequence, kind, data)

    def finish_run(self, run_id: str, stop_reason: str, steps_used: int) -> None:
        self._backend.finish_run(run_id, stop_reason, steps_used)

    def record_usage(self, run_id: str, usage: UsageRecord) -> None:
        self._backend.record_usage(run_id, usage)

    def usage(self, run_id: str) -> tuple[UsageRecord, ...]:
        return self._backend.usage(run_id)

    def continuation_session(
        self, run_id: str
    ) -> tuple[str, str | None, tuple[ChatMessage, ...]] | None:
        return self._backend.continuation_session(run_id)


class SQLiteToolLedgerRepository:
    """Tool lifecycle and recovery adapter backed by SQLite."""

    def __init__(self, backend: SQLiteRepository) -> None:
        self._backend = backend

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
        self._backend.record_tool(
            run_id, call_id, name, arguments, status, result, idempotent=idempotent
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
        self._backend.record_tool_lifecycle(
            run_id, call_id, name, arguments, result, idempotent=idempotent
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
        self._backend.record_tool_started(run_id, call_id, name, arguments, idempotent=idempotent)

    def unresolved_tools(self) -> tuple[UnresolvedTool, ...]:
        return self._backend.unresolved_tools()

    def retry_tool(
        self, run_id: str, call_id: str, continuation_run_id: str, session_id: str
    ) -> UnresolvedTool:
        return self._backend.retry_tool(run_id, call_id, continuation_run_id, session_id)

    def resolve_tool(
        self,
        run_id: str,
        call_id: str,
        result: str,
        continuation_run_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._backend.resolve_tool(run_id, call_id, result, continuation_run_id, session_id)


class InMemoryRepository(SQLiteRepository):
    """Test repository with the same public behavior and no filesystem state."""

    def __init__(self, *, redactor: Redactor | None = None) -> None:
        super().__init__(":memory:", redactor=redactor)
