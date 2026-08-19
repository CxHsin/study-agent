import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from uuid import uuid4

from minimal_agent.context import ContextBuilder, ContextError, ModelSummarizer
from minimal_agent.cost import PromptCacheStore, checkpoint_for, usage_record
from minimal_agent.events import AgentEvent, AgentEventListener, EventKind
from minimal_agent.protocol import (
    ChatMessage,
    ModelAdapter,
    ModelError,
    ModelResponse,
    ModelStreamChunk,
    ProviderCapabilities,
    ToolCall,
)
from minimal_agent.session import AgentSession
from minimal_agent.tools import ToolError, ToolRegistry, ToolResult


class StopReason(StrEnum):
    FINAL = "final"
    MAX_STEPS = "max_steps"
    REPEATED_TOOL_CALL = "repeated_tool_call"
    ABORTED = "aborted"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True)
class RunError:
    code: str
    message: str
    error_type: str
    step: int
    tool_call_id: str | None = None


@dataclass(frozen=True)
class RunResult:
    final_response: str | None
    stop_reason: StopReason
    steps_used: int
    run_id: str
    error: RunError | None = None
    events: tuple[AgentEvent, ...] = ()
    context_metadata: tuple[dict[str, object], ...] = ()


class RunControl:
    def __init__(self) -> None:
        self._stop_reason: StopReason | None = None

    def abort(self) -> None:
        self._stop_reason = StopReason.ABORTED

    def cancel(self) -> None:
        if self._stop_reason is None:
            self._stop_reason = StopReason.CANCELLED

    @property
    def stop_reason(self) -> StopReason | None:
        return self._stop_reason


class AgentCore:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry | None = None,
        session: AgentSession | None = None,
        max_steps: int = 8,
        context_builder: ContextBuilder | None = None,
        repository: object | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._session = session or AgentSession()
        self._max_steps = max_steps
        self._context_builder = context_builder or ContextBuilder(summarizer=ModelSummarizer(model))
        self._repository = repository
        self._cache_store = PromptCacheStore()
        self._listeners: list[AgentEventListener] = []
        self._sequence = 0
        self._active_lock = threading.Lock()
        self._active_control: RunControl | None = None
        self._steering_open = False
        self._steering: queue.Queue[str] = queue.Queue()

    def subscribe(self, listener: AgentEventListener) -> None:
        self._listeners.append(listener)

    @property
    def session(self) -> AgentSession:
        return self._session

    def prompt(self, user_input: str, control: RunControl | None = None) -> RunResult:
        return self._execute_prompt(user_input, control)

    def follow_up(self, user_input: str, control: RunControl | None = None) -> RunResult:
        """Start a new Run using the existing Conversation Session."""
        return self.prompt(user_input, control)

    def steer(self, message: str) -> bool:
        """Queue a user message for the next model-call boundary of the active Run."""
        with self._active_lock:
            if self._active_control is None or not self._steering_open:
                return False
            self._steering.put(message)
            return True

    def stream(self, user_input: str, control: RunControl | None = None) -> Iterator[AgentEvent]:
        """Yield the same ordered events produced by a synchronous prompt run."""
        control = control or RunControl()
        events: queue.Queue[AgentEvent | object] = queue.Queue()
        finished = object()
        completed = False

        def run() -> None:
            try:
                self._execute_prompt(user_input, control, event_sink=events.put)
            finally:
                events.put(finished)

        threading.Thread(target=run, name="minimal-agent-stream", daemon=True).start()
        try:
            while True:
                event = events.get()
                if event is finished:
                    completed = True
                    return
                yield event  # type narrowing is not available for the sentinel union
        finally:
            if not completed:
                control.cancel()

    def _execute_prompt(
        self,
        user_input: str,
        control: RunControl | None = None,
        *,
        event_sink: Callable[[AgentEvent], None] | None = None,
    ) -> RunResult:
        run_id = str(uuid4())
        control = control or RunControl()
        if not self._session.try_acquire_run():
            result = RunResult(
                None,
                StopReason.ERROR,
                0,
                run_id,
                RunError(
                    "SESSION_BUSY",
                    "Conversation Session already has an active Run.",
                    "control_error",
                    0,
                ),
            )
            if event_sink is not None:
                event_sink(
                    AgentEvent(
                        run_id,
                        EventKind.RUN_ERROR,
                        {
                            "stop_reason": StopReason.ERROR.value,
                            "steps_used": 0,
                            "error": result.error,
                        },
                        1,
                        datetime.now(UTC),
                        0,
                    )
                )
            return result
        with self._active_lock:
            self._active_control = control
            self._steering_open = True
        self._repository_call("start_run", run_id, self._session.session_id)
        self._sequence = 0
        trace: list[AgentEvent] = []
        started_at = time.perf_counter()
        disabled_listeners: set[int] = set()
        context_metadata: list[dict[str, object]] = []
        self._session.append({"role": "user", "content": user_input})
        self._repository_call(
            "save_session",
            self._session.session_id,
            self._session.system_prompt,
            self._session.messages,
        )
        emit = lambda kind, data: self._emit(
            run_id, kind, data, trace, started_at, disabled_listeners, event_sink
        )
        emit(
            EventKind.RUN_STARTED,
            {
                "query": user_input,
                "max_steps": self._max_steps,
                "message_count": len(self._session.messages),
            },
        )
        last_fingerprint: str | None = None
        pending_repeat: str | None = None
        try:
            for step in range(1, self._max_steps + 1):
                stop = _control_stop(control)
                if stop:
                    return self._result(run_id, stop, step - 1, trace, emit, context_metadata)
                self._apply_steering(run_id, step, emit)
                emit(
                    EventKind.MODEL_CALL_STARTED,
                    {"step": step, "message_count": len(self._session.messages)},
                )
                model_started = time.perf_counter()
                try:
                    context = self._context_builder.build(
                        self._session.messages, system_prompt=self._session.system_prompt
                    )
                    metadata = {
                        "step": step,
                        "estimated_tokens_before": context.estimated_tokens_before,
                        "estimated_tokens": context.estimated_tokens,
                        "input_budget": context.input_budget,
                        "compressed": context.compressed,
                        "summary_count": len(context.summaries),
                        "summary_cache_hits": context.cache_hits,
                        "compression_elapsed_ms": context.elapsed_ms,
                        "estimator": self._context_builder.estimator.name,
                    }
                    context_metadata.append(metadata)
                    response = _model_response(self._model, context.messages, emit, step)
                    capabilities = _provider_capabilities(self._model)
                    usage = usage_record(
                        response,
                        estimated_input=context.estimated_tokens,
                        latency_ms=(time.perf_counter() - model_started) * 1000,
                        capabilities=capabilities,
                    )
                    checkpoint = checkpoint_for(
                        context.messages,
                        session_id=self._session.session_id,
                        message_index=len(context.messages),
                        model=getattr(self._model, "model_name", type(self._model).__name__),
                        tool_schema=self._tools.definitions(),
                        system_prompt=self._session.system_prompt,
                        context_builder=self._context_builder.estimator.name,
                    )
                    local_cache_hit = self._cache_store.lookup(checkpoint)
                    self._cache_store.record(checkpoint)
                    usage = usage_record(
                        response,
                        estimated_input=context.estimated_tokens,
                        latency_ms=(time.perf_counter() - model_started) * 1000,
                        capabilities=capabilities,
                        local_cache_hit=local_cache_hit,
                    )
                except ContextError as error:
                    return self._error(
                        run_id,
                        error.code,
                        str(error),
                        "context_error",
                        step,
                        trace=trace,
                        emit=emit,
                        context_metadata=context_metadata,
                    )
                except ModelError as error:
                    return self._error(
                        run_id,
                        "MODEL_ERROR",
                        str(error) or "Model request failed.",
                        "model_error",
                        step,
                        trace=trace,
                        emit=emit,
                        context_metadata=context_metadata,
                    )
                emit(
                    EventKind.MODEL_RESPONSE,
                    {
                        "step": step,
                        "content": response.content,
                        "tool_calls": tuple(_tool_data(call) for call in response.tool_calls),
                        "usage": response.usage,
                        "provider_cache_hit": response.provider_cache_hit,
                        "usage_record": usage,
                        "prompt_cache_checkpoint": checkpoint,
                    },
                )
                stop = _control_stop(control)
                if stop:
                    return self._result(run_id, stop, step, trace, emit, context_metadata)
                if not response.tool_calls:
                    if not response.content:
                        return self._error(
                            run_id,
                            "INVALID_MODEL_RESPONSE",
                            "Model returned no final content.",
                            "model_error",
                            step,
                            trace=trace,
                            emit=emit,
                            context_metadata=context_metadata,
                        )
                    self._session.append({"role": "assistant", "content": response.content})
                    self._close_steering()
                    emit(EventKind.FINAL_RESPONSE, {"step": step, "content": response.content})
                    return self._finish(
                        RunResult(
                            response.content,
                            StopReason.FINAL,
                            step,
                            run_id,
                            context_metadata=tuple(context_metadata),
                        ),
                        trace,
                    )

                self._session.append(
                    {
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": response.tool_calls,
                    }
                )
                fingerprints = [_fingerprint(call) for call in response.tool_calls]
                if pending_repeat is not None and pending_repeat in fingerprints:
                    return self._error(
                        run_id,
                        "REPEATED_TOOL_CALL",
                        "Model repeated a blocked tool call.",
                        "control_error",
                        step,
                        StopReason.REPEATED_TOOL_CALL,
                        trace=trace,
                        emit=emit,
                        context_metadata=context_metadata,
                    )
                for tool_call in response.tool_calls:
                    stop = _control_stop(control)
                    if stop:
                        return self._result(run_id, stop, step, trace, emit, context_metadata)
                    fingerprint = _fingerprint(tool_call)
                    if fingerprint == last_fingerprint:
                        result = ToolResult(
                            tool_call.id,
                            tool_call.name,
                            False,
                            error=ToolError("REPEATED_TOOL_CALL", "Repeated tool call blocked."),
                        )
                        self._session.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result.to_json(),
                            }
                        )
                        emit(EventKind.TOOL_RESULT_PRODUCED, _tool_result_data(result))
                        pending_repeat = fingerprint
                        last_fingerprint = fingerprint
                        continue
                    pending_repeat = None
                    last_fingerprint = fingerprint
                    definition = self._tools.get(tool_call.name)
                    self._repository_call(
                        "record_tool",
                        run_id,
                        tool_call.id,
                        tool_call.name,
                        tool_call.arguments,
                        "requested",
                        None,
                        idempotent=definition.idempotent if definition else False,
                    )
                    emit(EventKind.TOOL_CALL_REQUESTED, _tool_data(tool_call))
                    self._repository_call(
                        "record_tool",
                        run_id,
                        tool_call.id,
                        tool_call.name,
                        tool_call.arguments,
                        "started",
                        None,
                        idempotent=definition.idempotent if definition else False,
                    )
                    result = self._tools.execute(tool_call, run_id=run_id, user_input=user_input)
                    self._repository_call(
                        "record_tool",
                        run_id,
                        tool_call.id,
                        tool_call.name,
                        tool_call.arguments,
                        "completed",
                        result.to_json(),
                        idempotent=definition.idempotent if definition else False,
                    )
                    self._session.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "content": result.to_json()}
                    )
                    emit(EventKind.TOOL_RESULT_PRODUCED, _tool_result_data(result))
                    stop = _control_stop(control)
                    if stop:
                        return self._result(run_id, stop, step, trace, emit, context_metadata)
            return self._result(
                run_id,
                StopReason.MAX_STEPS,
                self._max_steps,
                trace,
                emit,
                context_metadata,
            )
        except Exception as error:  # noqa: BLE001 - run boundary returns structured errors
            return self._error(
                run_id,
                "INTERNAL_ERROR",
                str(error) or "Internal error.",
                "internal_error",
                locals().get("step", 0),
                trace=trace,
                emit=emit,
                context_metadata=context_metadata,
            )
        finally:
            with self._active_lock:
                self._active_control = None
                self._steering_open = False
            self._apply_pending_steering_to_session()
            self._repository_call(
                "save_session",
                self._session.session_id,
                self._session.system_prompt,
                self._session.messages,
            )
            self._session.release_run()

    def _discard_pending_steering(self) -> None:
        while True:
            try:
                self._steering.get_nowait()
            except queue.Empty:
                return

    def _apply_pending_steering_to_session(self) -> None:
        while True:
            try:
                message = self._steering.get_nowait()
            except queue.Empty:
                return
            self._session.append({"role": "user", "content": message})

    def _repository_call(self, method: str, *args, **kwargs) -> None:
        if self._repository is None:
            return
        callback = getattr(self._repository, method, None)
        if callback is not None:
            callback(*args, **kwargs)

    def _apply_steering(self, run_id: str, step: int, emit) -> None:
        while True:
            try:
                message = self._steering.get_nowait()
            except queue.Empty:
                return
            self._session.append({"role": "user", "content": message})
            emit(
                EventKind.STEERING_MESSAGE_ACCEPTED,
                {"step": step, "content": message, "run_id": run_id},
            )

    def _finish(self, result: RunResult, trace: list[AgentEvent]) -> RunResult:
        self._close_steering()
        self._repository_call(
            "save_session",
            self._session.session_id,
            self._session.system_prompt,
            self._session.messages,
        )
        self._repository_call(
            "finish_run", result.run_id, result.stop_reason.value, result.steps_used
        )
        return RunResult(
            result.final_response,
            result.stop_reason,
            result.steps_used,
            result.run_id,
            result.error,
            tuple(trace),
            tuple(MappingProxyType(dict(item)) for item in result.context_metadata),
        )

    def _result(
        self,
        run_id: str,
        reason: StopReason,
        steps: int,
        trace: list[AgentEvent],
        emit,
        context_metadata: list[dict[str, object]] | None = None,
    ) -> RunResult:
        self._close_steering()
        emit(EventKind.RUN_STOPPED, {"stop_reason": reason.value, "steps_used": steps})
        return self._finish(
            RunResult(None, reason, steps, run_id, context_metadata=tuple(context_metadata or ())),
            trace,
        )

    def _error(
        self,
        run_id: str,
        code: str,
        message: str,
        error_type: str,
        step: int,
        reason: StopReason = StopReason.ERROR,
        *,
        trace: list[AgentEvent],
        emit,
        context_metadata: list[dict[str, object]] | None = None,
    ) -> RunResult:
        self._close_steering()
        error = RunError(code, message, error_type, step)
        emit(
            EventKind.RUN_ERROR,
            {"stop_reason": reason.value, "steps_used": step, "error": error},
        )
        return self._finish(
            RunResult(
                None, reason, step, run_id, error, context_metadata=tuple(context_metadata or ())
            ),
            trace,
        )

    def _close_steering(self) -> None:
        with self._active_lock:
            self._steering_open = False

    def _emit(
        self,
        run_id: str,
        kind: EventKind,
        data: dict[str, object],
        trace: list[AgentEvent],
        started_at: float,
        disabled_listeners: set[int],
        event_sink: Callable[[AgentEvent], None] | None = None,
    ) -> None:
        self._sequence += 1
        event = AgentEvent(
            run_id,
            kind,
            data,
            self._sequence,
            datetime.now(UTC),
            (time.perf_counter() - started_at) * 1000,
        )
        trace.append(event)
        self._repository_call("append_event", run_id, event.sequence, event.kind.value, event.data)
        if event_sink is not None:
            event_sink(event)
        for index, listener in enumerate(self._listeners):
            if index in disabled_listeners:
                continue
            try:
                listener(event)
            except Exception as error:  # noqa: BLE001 - observer failures are isolated
                disabled_listeners.add(index)
                self._emit(
                    run_id,
                    EventKind.LISTENER_ERROR,
                    {"listener": type(listener).__name__, "message": str(error)},
                    trace,
                    started_at,
                    disabled_listeners,
                    event_sink,
                )


def _tool_data(tool_call: ToolCall) -> dict[str, object]:
    return {"tool_call_id": tool_call.id, "name": tool_call.name, "arguments": tool_call.arguments}


def _tool_result_data(result: ToolResult) -> dict[str, object]:
    return {
        "tool_call_id": result.tool_call_id,
        "name": result.tool_name,
        "result": result.to_json(),
        "success": result.ok,
        "retryable": result.retryable,
    }


def _control_stop(control: RunControl) -> StopReason | None:
    return control.stop_reason


def _fingerprint(tool_call: ToolCall) -> str:
    try:
        arguments = json.dumps(
            json.loads(tool_call.arguments), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, json.JSONDecodeError):
        arguments = tool_call.arguments.strip()
    return f"{tool_call.name}\0{arguments}"


def _model_response(
    model: ModelAdapter,
    messages: tuple[ChatMessage, ...],
    emit: Callable[[EventKind, dict[str, object]], None],
    step: int,
) -> ModelResponse:
    capabilities = getattr(model, "capabilities", None)
    if callable(capabilities) and not capabilities().streaming:
        return model.complete(messages)
    stream = getattr(model, "stream", None)
    if not callable(stream):
        return model.complete(messages)

    content_parts: list[str] = []
    calls: dict[str, dict[str, str]] = {}
    order: list[str] = []
    usage = None
    provider_cache_hit = None
    completed = False
    for chunk in stream(messages):
        if not isinstance(chunk, ModelStreamChunk):
            raise ModelError("Provider returned an invalid stream chunk.")
        if chunk.usage is not None:
            usage = chunk.usage
        if chunk.provider_cache_hit is not None:
            provider_cache_hit = chunk.provider_cache_hit
        if chunk.done:
            completed = True
        if chunk.content_delta:
            content_parts.append(chunk.content_delta)
            emit(
                EventKind.MODEL_CONTENT_DELTA,
                {"step": step, "content_delta": chunk.content_delta},
            )
        if chunk.tool_call_id is not None:
            if chunk.tool_call_id not in calls:
                calls[chunk.tool_call_id] = {"name": chunk.tool_name or "", "arguments": ""}
                order.append(chunk.tool_call_id)
            call = calls[chunk.tool_call_id]
            if chunk.tool_name:
                call["name"] = chunk.tool_name
            if chunk.arguments_delta:
                call["arguments"] += chunk.arguments_delta
            emit(
                EventKind.TOOL_CALL_DELTA,
                {
                    "step": step,
                    "tool_call_id": chunk.tool_call_id,
                    "name": chunk.tool_name,
                    "arguments_delta": chunk.arguments_delta,
                },
            )
    if any(not calls[call_id]["name"] or not calls[call_id]["arguments"] for call_id in order):
        raise ModelError("Provider returned an incomplete Tool Call.")
    if not completed:
        raise ModelError("Provider stream ended before completion.")
    for call_id in order:
        try:
            json.loads(calls[call_id]["arguments"])
        except json.JSONDecodeError as error:
            raise ModelError("Provider returned malformed Tool Call arguments.") from error
    tool_calls = tuple(
        ToolCall(call_id, calls[call_id]["name"], calls[call_id]["arguments"]) for call_id in order
    )
    return ModelResponse(
        "".join(content_parts) or None,
        tool_calls,
        usage=usage,
        provider_cache_hit=provider_cache_hit,
    )


def _provider_capabilities(model: ModelAdapter) -> ProviderCapabilities:
    callback = getattr(model, "capabilities", None)
    if callable(callback):
        value = callback()
        if isinstance(value, ProviderCapabilities):
            return value
    return ProviderCapabilities(streaming=callable(getattr(model, "stream", None)))
