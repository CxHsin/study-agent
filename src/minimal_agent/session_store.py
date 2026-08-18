import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from minimal_agent.agent import ChatMessage, ToolCall


class SessionNotFoundError(KeyError):
    pass


class SQLiteSessionStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                """
            )

    def create_session(self) -> str:
        session_id = str(uuid4())
        now = _timestamp()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now),
            )
        return session_id

    def latest_session_id(self) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC, session_id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row is not None else None

    def append_message(
        self,
        session_id: str,
        message: Mapping[str, object],
        *,
        run_id: str,
    ) -> None:
        payload = _encode_message(message)
        now = _timestamp()
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
                is None
            ):
                raise SessionNotFoundError(session_id)
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO messages(session_id, sequence, run_id, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, next_sequence, run_id, payload, now),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def load_messages(self, session_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM messages WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [cast(ChatMessage, json.loads(row[0])) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _encode_message(message: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            _normalize_value(dict(message)), ensure_ascii=False, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise TypeError("Message must contain only JSON-serializable values.") from error


def _normalize_value(value: object) -> object:
    if isinstance(value, ToolCall):
        return {"id": value.id, "name": value.name, "arguments": value.arguments}
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
