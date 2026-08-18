from pathlib import Path

from minimal_agent.agent import ToolCall
from minimal_agent.session_store import SQLiteSessionStore


def test_session_messages_survive_reopening_the_store(tmp_path: Path) -> None:
    database = tmp_path / "sessions.sqlite3"
    store = SQLiteSessionStore(database)

    session_id = store.create_session()
    store.append_message(
        session_id,
        {"role": "user", "content": "Read notes.txt."},
        run_id="run-1",
    )
    store.append_message(
        session_id,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"notes.txt"}',
                }
            ],
        },
        run_id="run-1",
    )

    reopened = SQLiteSessionStore(database)

    assert reopened.load_messages(session_id) == [
        {"role": "user", "content": "Read notes.txt."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"notes.txt"}',
                }
            ],
        },
    ]


def test_store_normalizes_runtime_tool_call_values(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    session_id = store.create_session()

    store.append_message(
        session_id,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": (ToolCall("call-1", "read_file", '{"path":"notes.txt"}'),),
        },
        run_id="run-1",
    )

    assert store.load_messages(session_id) == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path":"notes.txt"}',
                }
            ],
        }
    ]


def test_latest_session_is_reused_until_a_new_session_is_created(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")

    assert store.latest_session_id() is None
    first_session = store.create_session()
    assert store.latest_session_id() == first_session

    second_session = store.create_session()

    assert store.latest_session_id() == second_session


def test_recovered_history_can_expose_an_unfinished_tool_call(tmp_path: Path) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite3")
    session_id = store.create_session()
    store.append_message(
        session_id,
        {"role": "user", "content": "Read notes.txt."},
        run_id="run-1",
    )
    store.append_message(
        session_id,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1", "name": "read_file", "arguments": '{"path":"notes.txt"}'}
            ],
        },
        run_id="run-1",
    )

    recovered = store.load_messages(session_id)

    assert recovered[-1]["role"] == "assistant"
    assert "tool_calls" in recovered[-1]
    assert not any(message.get("role") == "tool" for message in recovered)
