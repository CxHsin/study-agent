import json
from pathlib import Path

import pytest

from minimal_agent.agent import TRACE_SCHEMA_VERSION, TraceEvent
from minimal_agent.trace import JsonlTraceStore


def test_jsonl_trace_store_persists_events_and_reads_last_run(tmp_path: Path) -> None:
    trace_file = tmp_path / "traces.jsonl"
    store = JsonlTraceStore(trace_file)
    first_run = TraceEvent(
        kind="task_end",
        data={"status": "completed"},
        run_id="run-1",
        sequence=1,
        timestamp="2026-08-17T10:00:00Z",
    )
    second_run_events = [
        TraceEvent(
            kind="agent_step",
            data={"step": 1},
            run_id="run-2",
            sequence=1,
            timestamp="2026-08-17T10:01:00Z",
        ),
        TraceEvent(
            kind="task_end",
            data={"status": "failed", "stop_reason": "model_error"},
            run_id="run-2",
            sequence=2,
            timestamp="2026-08-17T10:01:01Z",
        ),
    ]

    for event in [first_run, *second_run_events]:
        store(event)

    assert store.read_last_run() == second_run_events
    records = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    assert [record["run_id"] for record in records] == ["run-1", "run-2", "run-2"]
    assert all(record["schema_version"] == TRACE_SCHEMA_VERSION for record in records)


def test_read_last_run_stops_at_interleaved_run(tmp_path: Path) -> None:
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    events = [
        TraceEvent("agent_step", {"step": 1}, "run-a", 1, "2026-08-17T10:00:00Z"),
        TraceEvent("agent_step", {"step": 1}, "run-b", 1, "2026-08-17T10:00:01Z"),
        TraceEvent("task_end", {"status": "completed"}, "run-a", 2, "2026-08-17T10:00:02Z"),
        TraceEvent("task_end", {"status": "completed"}, "run-b", 2, "2026-08-17T10:00:03Z"),
    ]
    for event in events:
        store(event)

    assert store.read_last_run() == [events[-1]]


@pytest.mark.parametrize(
    "event",
    [
        TraceEvent("task_end", {}, "run-1", 1, "2026-08-17T10:00:00Z"),
    ],
)
def test_trace_event_rejects_invalid_schema_version(event: TraceEvent) -> None:
    with pytest.raises(ValueError, match="Unsupported Trace schema version"):
        TraceEvent(
            kind=event.kind,
            data=event.data,
            run_id=event.run_id,
            sequence=event.sequence,
            timestamp=event.timestamp,
            schema_version=2,
        )
