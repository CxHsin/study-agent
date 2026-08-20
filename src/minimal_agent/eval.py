"""Bounded regression evaluation for complete Agent Core runs."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from minimal_agent.core import AgentCore, RunResult, StopReason
from minimal_agent.events import EventKind
from minimal_agent.protocol import (
    ModelAdapter,
    ModelRequest,
    ProviderCapabilities,
    message_to_dict,
)
from minimal_agent.provider_client import ProviderClient, provider_client_for
from minimal_agent.session import AgentSession
from minimal_agent.tools import ToolRegistry


class EvalSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class TextExpectation:
    exact: str | None = None
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    regex: str | None = None
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True)
class ToolExpectation:
    name: str
    arguments: Mapping[str, object] | None = None


@dataclass(frozen=True)
class EvalExpectation:
    stop_reason: str = StopReason.FINAL.value
    final_text: TextExpectation = field(default_factory=TextExpectation)
    allowed_tools: tuple[str, ...] = ()
    tool_trace: tuple[ToolExpectation, ...] = ()
    trajectory_mode: str = "strict"
    event_kinds: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    usage_source: str | None = None
    cache_hit_source: str | None = None
    steering_count: int | None = None
    recovery_status: str | None = None


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    prompt: str
    expectation: EvalExpectation
    system_prompt: str | None = None
    max_steps: int = 8
    read_only_only: bool = True
    schema_version: str = "1"
    steering_messages: tuple[str, ...] = ()
    follow_up_messages: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> EvalCase:
        try:
            case_id = str(data["case_id"])
            prompt = str(data["prompt"])
            raw_expectation = data["expectation"]
        except KeyError as error:
            raise EvalSchemaError(f"Missing case field: {error.args[0]}") from error
        if not isinstance(raw_expectation, Mapping):
            raise EvalSchemaError("expectation must be an object.")
        raw_text = raw_expectation.get("final_text", {})
        if not isinstance(raw_text, Mapping):
            raise EvalSchemaError("expectation.final_text must be an object.")
        text = TextExpectation(
            exact=_optional_str(raw_text.get("exact")),
            contains=tuple(str(item) for item in _sequence(raw_text.get("contains"))),
            excludes=tuple(str(item) for item in _sequence(raw_text.get("excludes"))),
            regex=_optional_str(raw_text.get("regex")),
            min_length=_optional_int(raw_text.get("min_length")),
            max_length=_optional_int(raw_text.get("max_length")),
        )
        raw_trace = raw_expectation.get("tool_trace", [])
        if not isinstance(raw_trace, Sequence) or isinstance(raw_trace, (str, bytes)):
            raise EvalSchemaError("expectation.tool_trace must be an array.")
        trace = tuple(
            ToolExpectation(
                str(item["name"]),
                item.get("arguments") if isinstance(item, Mapping) else None,
            )
            for item in raw_trace
            if isinstance(item, Mapping) and "name" in item
        )
        expectation = EvalExpectation(
            stop_reason=str(raw_expectation.get("stop_reason", StopReason.FINAL.value)),
            final_text=text,
            allowed_tools=tuple(
                str(item) for item in _sequence(raw_expectation.get("allowed_tools"))
            ),
            tool_trace=trace,
            trajectory_mode=str(raw_expectation.get("trajectory_mode", "strict")),
            event_kinds=tuple(str(item) for item in _sequence(raw_expectation.get("event_kinds"))),
            capabilities=tuple(
                str(item) for item in _sequence(raw_expectation.get("capabilities"))
            ),
            usage_source=_optional_str(raw_expectation.get("usage_source")),
            cache_hit_source=_optional_str(raw_expectation.get("cache_hit_source")),
            steering_count=_optional_int(raw_expectation.get("steering_count")),
            recovery_status=_optional_str(raw_expectation.get("recovery_status")),
        )
        case = cls(
            case_id,
            prompt,
            expectation,
            _optional_str(data.get("system_prompt")),
            int(data.get("max_steps", 8)),
            bool(data.get("read_only_only", True)),
            str(data.get("schema_version", "1")),
            tuple(str(item) for item in _sequence(data.get("steering_messages"))),
            tuple(str(item) for item in _sequence(data.get("follow_up_messages"))),
        )
        if not case.case_id or case.max_steps < 1:
            raise EvalSchemaError("case_id must be non-empty and max_steps must be positive.")
        if expectation.trajectory_mode not in {"strict", "structural"}:
            raise EvalSchemaError("trajectory_mode must be strict or structural.")
        return case


@dataclass(frozen=True)
class RuleResult:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    status: str
    hard_rules: tuple[RuleResult, ...]
    trajectory: RuleResult
    text: tuple[RuleResult, ...]
    run: RunResult | None = None
    error: str | None = None
    calls_used: int = 0
    elapsed_ms: float = 0
    estimated_tokens: int = 0


@dataclass(frozen=True)
class EvalReport:
    schema_version: str
    run_id: str
    status: str
    cases: tuple[CaseEvaluation, ...]

    def to_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for case in self.cases:
                payload = _redact(_artifact_value(case))
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


@dataclass(frozen=True)
class EvalLimits:
    max_model_calls: int = 20
    timeout_seconds: float = 60.0
    token_budget: int = 100_000

    def __post_init__(self) -> None:
        if self.max_model_calls < 1 or self.timeout_seconds <= 0 or self.token_budget < 1:
            raise ValueError("Eval limits must be positive.")


@dataclass
class _Budget:
    calls: int = 0
    tokens: int = 0


class _CountingClient(ProviderClient):
    def __init__(self, client: ProviderClient, limits: EvalLimits, budget: _Budget) -> None:
        self._client = client
        self.profile = client.profile
        self.limits = limits
        self.budget = budget
        self.calls = 0

    def stream(self, request: ModelRequest, *, on_attempt=None):
        if self.budget.calls >= self.limits.max_model_calls:
            raise EvalLimitError("max_model_calls exceeded")
        self.calls += 1
        self.budget.calls += 1
        self.budget.tokens += max(
            1,
            sum(
                len(str(value))
                for message in request.messages
                for value in message_to_dict(message, include_version=False).values()
            )
            // 4,
        )
        if self.budget.tokens > self.limits.token_budget:
            raise EvalLimitError("token_budget exceeded")
        yield from self._client.stream(request, on_attempt=on_attempt)


class EvalLimitError(RuntimeError):
    pass


ModelFactory = Callable[[EvalCase], ModelAdapter]
ToolsFactory = Callable[[EvalCase], ToolRegistry]


class EvalRunner:
    def __init__(
        self,
        model_factory: ModelFactory,
        tools_factory: ToolsFactory | None = None,
        limits: EvalLimits | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._tools_factory = tools_factory or (lambda case: ToolRegistry())
        self._limits = limits or EvalLimits()

    def run(self, cases: Iterable[EvalCase], *, run_id: str = "eval") -> EvalReport:
        budget = _Budget()
        evaluations = tuple(self._run_case(case, budget) for case in cases)
        statuses = {item.status for item in evaluations}
        status = (
            "passed"
            if evaluations and statuses == {"passed"}
            else ("inconclusive" if "inconclusive" in statuses else "failed")
        )
        return EvalReport("1", run_id, status, evaluations)

    def _run_case(self, case: EvalCase, budget: _Budget) -> CaseEvaluation:
        started = time.perf_counter()
        try:
            counter = _CountingClient(
                provider_client_for(self._model_factory(case)), self._limits, budget
            )
        except Exception as error:  # noqa: BLE001 - Provider setup failures are inconclusive.
            return _inconclusive(case, f"provider_unavailable: {error}", elapsed=started)
        try:
            tools = self._tools_factory(case)
            if case.read_only_only and not tools.all_read_only():
                return _inconclusive(case, "case includes non-read-only tool")
            core = AgentCore(
                counter,
                tools,
                AgentSession(system_prompt=case.system_prompt),
                max_steps=case.max_steps,
            )
            pending_steering = list(case.steering_messages)

            def drive_steering(event) -> None:
                if event.kind is EventKind.RUN_STARTED:
                    messages = tuple(pending_steering)
                    pending_steering.clear()
                    for message in messages:
                        core.steer(message)

            core.subscribe(drive_steering)
            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(core.prompt, case.prompt)
            try:
                result = future.result(timeout=self._limits.timeout_seconds)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            if result.error is not None and (
                "budget" in result.error.message or "max_model_calls" in result.error.message
            ):
                return _inconclusive(
                    case, result.error.message, calls=counter.calls, elapsed=started
                )
            if result.error is not None and result.error.error_type == "model_error":
                return _inconclusive(
                    case,
                    f"provider_unavailable: {result.error.message}",
                    calls=counter.calls,
                    elapsed=started,
                )
            for message in case.follow_up_messages:
                result = core.follow_up(message)
                if result.error is not None:
                    break
        except FuturesTimeoutError:
            return _inconclusive(case, "timeout", calls=counter.calls, elapsed=started)
        except EvalLimitError as error:
            return _inconclusive(case, str(error), calls=counter.calls, elapsed=started)
        except Exception as error:  # noqa: BLE001 - Provider and setup failures are inconclusive.
            return _inconclusive(
                case, f"provider_unavailable: {error}", calls=counter.calls, elapsed=started
            )
        hard = _hard_rules(result, case.expectation, counter.capabilities())
        trajectory = compare_trajectory(result, case.expectation, tools)
        text = compare_text(result.final_response, case.expectation.final_text)
        passed = (
            all(item.passed for item in hard)
            and trajectory.passed
            and all(item.passed for item in text)
        )
        return CaseEvaluation(
            case.case_id,
            "passed" if passed else "failed",
            tuple(hard),
            trajectory,
            tuple(text),
            result,
            calls_used=counter.calls,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            estimated_tokens=sum(
                int(item.get("estimated_tokens", 0)) for item in result.context_metadata
            ),
        )


def _hard_rules(
    result: RunResult, expected: EvalExpectation, capabilities: ProviderCapabilities
) -> list[RuleResult]:
    rules = [
        RuleResult(
            "stop_reason",
            result.stop_reason.value == expected.stop_reason,
            f"expected {expected.stop_reason}, got {result.stop_reason.value}",
        ),
        RuleResult(
            "final_response_present",
            bool(result.final_response) == (expected.stop_reason == StopReason.FINAL.value),
            "final response presence matches stop reason",
        ),
    ]
    if expected.event_kinds:
        observed = tuple(event.kind.value for event in result.events)
        rules.append(
            RuleResult(
                "event_kinds",
                observed == expected.event_kinds,
                f"expected {expected.event_kinds}, got {observed}",
            )
        )
    for capability in expected.capabilities:
        supported = bool(getattr(capabilities, capability, False))
        rules.append(
            RuleResult(f"capability:{capability}", supported, f"provider capability {capability}")
        )
    usage_events = [
        event
        for event in result.events
        if event.kind.value == "model_response" and event.data.get("usage_record")
    ]
    if expected.usage_source:
        actual = (
            getattr(usage_events[-1].data["usage_record"], "source", "unknown")
            if usage_events
            else "unknown"
        )
        rules.append(
            RuleResult(
                "usage_source",
                actual == expected.usage_source,
                f"expected {expected.usage_source}, got {actual}",
            )
        )
    if expected.cache_hit_source:
        actual = (
            getattr(usage_events[-1].data["usage_record"], "cache_hit_source", "unknown")
            if usage_events
            else "unknown"
        )
        rules.append(
            RuleResult(
                "cache_hit_source",
                actual == expected.cache_hit_source,
                f"expected {expected.cache_hit_source}, got {actual}",
            )
        )
    if expected.steering_count is not None:
        actual = sum(event.kind.value == "steering_message_accepted" for event in result.events)
        rules.append(
            RuleResult(
                "steering_count",
                actual == expected.steering_count,
                f"expected {expected.steering_count}, got {actual}",
            )
        )
    if expected.recovery_status is not None:
        actual = str(result.error.code if result.error else result.stop_reason.value)
        rules.append(
            RuleResult(
                "recovery_status",
                actual == expected.recovery_status,
                f"expected {expected.recovery_status}, got {actual}",
            )
        )
    return rules


def compare_trajectory(
    result: RunResult, expected: EvalExpectation, tools: ToolRegistry
) -> RuleResult:
    observed = [
        _tool_from_event(event.data)
        for event in result.events
        if event.kind.value == "tool_call_requested"
    ]
    names = [item.name for item in observed]
    if any(name not in expected.allowed_tools for name in names):
        return RuleResult(
            "allowed_tools",
            False,
            f"observed disallowed tool: {next(name for name in names if name not in expected.allowed_tools)}",
        )
    wanted = list(expected.tool_trace)
    if expected.trajectory_mode == "strict":
        passed = len(observed) == len(wanted) and all(
            _matches_tool(actual, want) for actual, want in zip(observed, wanted)
        )
    else:
        actual_names = [item.name for item in observed]
        wanted_names = [item.name for item in wanted]
        passed = all(name in actual_names for name in wanted_names) and all(
            tools.get(name) is not None and tools.get(name).read_only
            for name in actual_names
            if name not in wanted_names
        )
    return RuleResult(
        "trajectory", passed, f"expected {len(wanted)} calls, observed {len(observed)}"
    )


def compare_text(actual: str | None, expected: TextExpectation) -> list[RuleResult]:
    text = actual or ""
    checks: list[RuleResult] = []
    if expected.exact is not None:
        checks.append(RuleResult("text_exact", text == expected.exact, "exact text comparison"))
    checks.extend(
        RuleResult(f"text_contains:{item}", item in text, "required text present")
        for item in expected.contains
    )
    checks.extend(
        RuleResult(f"text_excludes:{item}", item not in text, "forbidden text absent")
        for item in expected.excludes
    )
    if expected.regex is not None:
        checks.append(
            RuleResult(
                "text_regex", re.search(expected.regex, text) is not None, "regex comparison"
            )
        )
    if expected.min_length is not None:
        checks.append(
            RuleResult("text_min_length", len(text) >= expected.min_length, "minimum length")
        )
    if expected.max_length is not None:
        checks.append(
            RuleResult("text_max_length", len(text) <= expected.max_length, "maximum length")
        )
    return checks or [RuleResult("text_non_empty", bool(text), "response must be non-empty")]


def load_cases(path: Path) -> tuple[EvalCase, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = data.get("cases", data) if isinstance(data, Mapping) else data
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise EvalSchemaError("Case file must contain an array or a cases array.")
    return tuple(EvalCase.from_mapping(item) for item in raw_cases if isinstance(item, Mapping))


def _tool_from_event(data: Mapping[str, object]) -> ToolExpectation:
    arguments = data.get("arguments", "{}")
    try:
        decoded = json.loads(str(arguments))
    except json.JSONDecodeError:
        decoded = str(arguments)
    return ToolExpectation(
        str(data.get("name", "")), decoded if isinstance(decoded, Mapping) else None
    )


def _matches_tool(actual: ToolExpectation, expected: ToolExpectation) -> bool:
    return actual.name == expected.name and (
        expected.arguments is None or actual.arguments == expected.arguments
    )


def _inconclusive(
    case: EvalCase, error: str, *, calls: int = 0, elapsed: float | None = None
) -> CaseEvaluation:
    return CaseEvaluation(
        case.case_id,
        "inconclusive",
        (),
        RuleResult("trajectory", False, error),
        (),
        error=error,
        calls_used=calls,
        elapsed_ms=((time.perf_counter() - elapsed) * 1000 if elapsed else 0),
    )


def _redact(value: Any) -> Any:
    sensitive = re.compile(
        r"(api[_-]?key|authorization|secret|password|\.env|absolute_path)", re.IGNORECASE
    )
    if isinstance(value, Mapping):
        return {
            key: "[REDACTED]" if sensitive.search(str(key)) else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(
            r"(?i)(api[_-]?key|authorization|secret|password)\s*[=:]\s*[^\s,;]+",
            lambda match: match.group(1) + "=[REDACTED]",
            value,
        )
    return value


def _artifact_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _artifact_value(getattr(value, field.name))
            for field in value.__dataclass_fields__.values()
        }
    if isinstance(value, MappingProxyType):
        return {key: _artifact_value(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {key: _artifact_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_artifact_value(item) for item in value]
    return value


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)
