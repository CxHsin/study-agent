import json
import threading
from collections.abc import Sequence

from minimal_agent.eval import (
    EvalCase,
    EvalLimits,
    EvalRunner,
    TextExpectation,
    compare_text,
    load_cases,
)
from minimal_agent.protocol import ChatMessage, ModelResponse, ToolCall
from minimal_agent.tools import ToolDefinition, ToolRegistry


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        return next(self.responses)


class BlockingSteeringModel:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            assert self.release.wait(timeout=5)
            return ModelResponse(tool_calls=(ToolCall("id", "read", '{"path":"notes"}'),))
        assert any(
            (
                message.get("role") == "user" and message.get("content") == "add detail"
                if isinstance(message, dict)
                else message.role == "user" and message.content == "add detail"
            )
            for message in messages
        )
        return ModelResponse(content="ok")


def read_tools() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                "read",
                "Read",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
                lambda args: {"value": args["path"]},
                read_only=True,
            )
        ]
    )


def test_eval_runner_checks_final_text_and_strict_trajectory() -> None:
    case = EvalCase.from_mapping(
        {
            "case_id": "read-1",
            "prompt": "read notes",
            "expectation": {
                "stop_reason": "final",
                "allowed_tools": ["read"],
                "tool_trace": [{"name": "read", "arguments": {"path": "notes"}}],
                "trajectory_mode": "strict",
                "final_text": {"contains": ["notes"]},
            },
        }
    )
    runner = EvalRunner(
        lambda _: ScriptedModel(
            [
                ModelResponse(tool_calls=(ToolCall("id", "read", '{"path":"notes"}'),)),
                ModelResponse(content="notes"),
            ]
        ),
        lambda _: read_tools(),
        EvalLimits(max_model_calls=4),
    )

    report = runner.run([case])

    assert report.status == "passed"
    assert report.cases[0].status == "passed"
    assert report.cases[0].calls_used == 2


def test_eval_optional_event_and_usage_assertions_remain_deterministic() -> None:
    case = EvalCase.from_mapping(
        {
            "case_id": "stream-1",
            "prompt": "hello",
            "expectation": {
                "event_kinds": [
                    "run_started",
                    "model_call_started",
                    "model_usage_recorded",
                    "model_response",
                    "final_response",
                ],
                "usage_source": "estimated",
                "cache_hit_source": "unknown",
            },
        }
    )
    report = EvalRunner(
        lambda _: ScriptedModel([ModelResponse(content="ok")]),
        limits=EvalLimits(max_model_calls=2),
    ).run([case])

    assert report.status == "passed"
    assert report.cases[0].hard_rules[-2].name == "usage_source"


def test_successful_eval_artifact_serializes_trace_data(tmp_path) -> None:
    case = EvalCase.from_mapping({"case_id": "ok", "prompt": "hi", "expectation": {}})
    report = EvalRunner(lambda _: ScriptedModel([ModelResponse(content="ok")])).run([case])
    artifact = tmp_path / "artifact.jsonl"

    report.to_jsonl(artifact)

    assert '"case_id": "ok"' in artifact.read_text(encoding="utf-8")


def test_eval_case_can_drive_steering_messages() -> None:
    model = BlockingSteeringModel()
    case = EvalCase.from_mapping(
        {
            "case_id": "steer",
            "prompt": "hi",
            "steering_messages": ["add detail"],
            "expectation": {
                "steering_count": 1,
                "allowed_tools": ["read"],
                "tool_trace": [{"name": "read", "arguments": {"path": "notes"}}],
            },
        }
    )
    report_holder = []

    def run() -> None:
        report_holder.append(EvalRunner(lambda _: model, lambda _: read_tools()).run([case]))

    thread = threading.Thread(target=run)
    thread.start()
    assert model.started.wait(timeout=5)
    model.release.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    report = report_holder[0]

    assert report.status == "passed"
    assert report.cases[0].hard_rules[-1].name == "steering_count"


def test_eval_case_can_drive_follow_up_messages() -> None:
    case = EvalCase.from_mapping(
        {
            "case_id": "follow-up",
            "prompt": "first",
            "follow_up_messages": ["second"],
            "expectation": {"final_text": {"exact": "followed"}},
        }
    )
    report = EvalRunner(
        lambda _: ScriptedModel(
            [ModelResponse(content="initial"), ModelResponse(content="followed")]
        )
    ).run([case])

    assert report.status == "passed"
    assert report.cases[0].run is not None
    assert report.cases[0].run.final_response == "followed"


def test_eval_steering_is_not_replayed_during_follow_up() -> None:
    class SteeringThenFollowUp:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: Sequence[ChatMessage]) -> ModelResponse:
            self.calls += 1
            steering_count = sum(
                message.get("role") == "user" and message.get("content") == "add detail"
                for message in messages
            )
            assert steering_count == 1
            return ModelResponse(content="initial" if self.calls == 1 else "followed")

    case = EvalCase.from_mapping(
        {
            "case_id": "steer-follow-up",
            "prompt": "first",
            "steering_messages": ["add detail"],
            "follow_up_messages": ["second"],
            "expectation": {"final_text": {"exact": "followed"}},
        }
    )

    report = EvalRunner(lambda _: SteeringThenFollowUp()).run([case])

    assert report.status == "passed"


def test_provider_exception_is_inconclusive_and_artifact_is_redacted(tmp_path) -> None:
    case = EvalCase.from_mapping(
        {
            "case_id": "provider",
            "prompt": "hello",
            "expectation": {},
        }
    )
    report = EvalRunner(lambda _: (_ for _ in ()).throw(RuntimeError("api_key=secret"))).run([case])
    artifact = tmp_path / "result.jsonl"
    report.to_jsonl(artifact)

    assert report.status == "inconclusive"
    assert "provider_unavailable" in (report.cases[0].error or "")
    assert "secret" not in artifact.read_text(encoding="utf-8")


def test_case_file_and_text_rules(tmp_path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"cases": [{"case_id": "x", "prompt": "p", "expectation": {}}]}),
        encoding="utf-8",
    )
    assert load_cases(path)[0].case_id == "x"
    assert all(
        item.passed for item in compare_text("hello world", TextExpectation(contains=("world",)))
    )


def test_eval_limits_are_shared_across_cases() -> None:
    cases = [
        EvalCase.from_mapping({"case_id": str(index), "prompt": "p", "expectation": {}})
        for index in range(2)
    ]
    report = EvalRunner(
        lambda _: ScriptedModel([ModelResponse(content="ok")]),
        limits=EvalLimits(max_model_calls=1),
    ).run(cases)

    assert report.cases[0].status == "passed"
    assert report.cases[1].status == "inconclusive"
    assert "max_model_calls" in (report.cases[1].error or "")
