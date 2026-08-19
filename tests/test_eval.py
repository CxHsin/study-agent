import json
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
