from minimal_agent.agent import StopReason, TaskResult, TaskStatus, TraceEvent
from minimal_agent.cli import run_console


class RecordingSession:
    def __init__(self) -> None:
        self.submissions: list[str] = []
        self.reset_count = 0

    def submit(self, user_input: str) -> TaskResult:
        self.submissions.append(user_input)
        return TaskResult(
            status=TaskStatus.COMPLETED,
            stop_reason=StopReason.FINAL_RESPONSE,
            final_response=f"Answer: {user_input}",
            steps_used=1,
        )

    def reset(self) -> None:
        self.reset_count += 1


class InterruptingSession(RecordingSession):
    def submit(self, user_input: str) -> TaskResult:
        raise KeyboardInterrupt


class RecordingTraceReader:
    def read_last_run(self) -> list[TraceEvent]:
        return [
            TraceEvent(
                kind="agent_step",
                data={"step": 1},
                run_id="run-1",
                sequence=1,
                timestamp="2026-08-17T10:00:00Z",
            ),
            TraceEvent(
                kind="model_call",
                data={"status": "ok", "tool_calls": 0},
                run_id="run-1",
                sequence=2,
                timestamp="2026-08-17T10:00:01Z",
                duration_ms=12.5,
            ),
        ]


def test_console_submits_tasks_resets_and_exits() -> None:
    inputs = iter(["hello", "/reset", "", "/exit"])
    outputs: list[str] = []
    session = RecordingSession()

    exit_code = run_console(
        session,
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert exit_code == 0
    assert session.submissions == ["hello"]
    assert session.reset_count == 1
    assert "Agent> Answer: hello" in outputs
    assert "Session reset." in outputs


def test_console_exits_cleanly_when_task_is_interrupted() -> None:
    exit_code = run_console(
        InterruptingSession(),
        input_fn=lambda prompt: "interrupt me",
        output_fn=lambda message: None,
    )

    assert exit_code == 0


def test_console_displays_the_last_trace_run() -> None:
    inputs = iter(["/trace last", "/exit"])
    outputs: list[str] = []

    exit_code = run_console(
        RecordingSession(),
        trace_reader=RecordingTraceReader(),
        input_fn=lambda prompt: next(inputs),
        output_fn=outputs.append,
    )

    assert exit_code == 0
    assert outputs == [
        "Trace> Last run:",
        "  1 agent_step step=1",
        "  2 model_call status=ok tool_calls=0 duration_ms=12.50",
    ]
