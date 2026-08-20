import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from minimal_agent.agent_loop import AgentLoop
from minimal_agent.context import ContextBuilder, ModelSummarizer
from minimal_agent.cost import PromptCacheStore
from minimal_agent.events import AgentEvent, AgentEventListener, EventKind
from minimal_agent.persistence import (
    Repository,
    RunRepository,
    SessionRepository,
    ToolExecutionRepository,
    repository_adapters,
)
from minimal_agent.protocol import (
    ChatMessage,
    ModelAdapter,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ProviderCapabilities,
    ProviderError,
    ProviderErrorKind,
    RequestOptions,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    UserMessage,
)
from minimal_agent.provider_client import ProviderClient, aggregate_stream
from minimal_agent.run import LoopOutcome, RunControl, RunError, RunResult, StopReason
from minimal_agent.session import AgentSession
from minimal_agent.tools import ToolRegistry


class AgentCore:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolRegistry | None = None,
        session: AgentSession | None = None,
        max_steps: int = 8,
        context_builder: ContextBuilder | None = None,
        repository: Repository | None = None,
        *,
        session_repository: SessionRepository | None = None,
        run_repository: RunRepository | None = None,
        tool_execution_repository: ToolExecutionRepository | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive.")
        self._model = model
        self._tools = tools or ToolRegistry()
        self._session = session or AgentSession()
        self._max_steps = max_steps
        self._context_builder = context_builder or ContextBuilder(summarizer=ModelSummarizer(model))
        profile = getattr(model, "profile", None)
        if isinstance(profile, ModelProfile):
            if self._context_builder.config.context_window_tokens > profile.context_window_tokens:
                raise ValueError("ContextConfig exceeds the Model Profile context window.")
            if self._context_builder.config.reserved_output_tokens > profile.max_output_tokens:
                raise ValueError("ContextConfig output reserve exceeds the Model Profile limit.")
        adapters = repository_adapters(repository)
        self._sessions = session_repository or adapters.sessions
        self._runs = run_repository or adapters.runs
        self._tool_ledger = tool_execution_repository or adapters.tools
        self._cache_store = PromptCacheStore()
        self._listeners: list[AgentEventListener] = []
        self._sequence = 0
        self._active_lock = threading.Lock()
        self._active_control: RunControl | None = None
        self._steering_open = False
        self._steering: queue.Queue[str] = queue.Queue()
        self._loop = AgentLoop(
            model=self._model,
            tools=self._tools,
            session=self._session,
            context_builder=self._context_builder,
            max_steps=self._max_steps,
            tool_ledger=self._tool_ledger,
            cache_store=self._cache_store,
            apply_steering=self._apply_steering,
            model_response=_model_response,
            provider_capabilities=_provider_capabilities,
        )

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

    def continue_run(
        self, continuation_run_id: str, user_input: str, control: RunControl | None = None
    ) -> RunResult:
        """Resume a persisted continuation using the current Session boundary."""
        if self._runs is None:
            raise RuntimeError("A Repository is required to continue a persisted Run.")
        lookup = getattr(self._runs, "continuation_session", None)
        continuation = lookup(continuation_run_id) if callable(lookup) else None
        if continuation is None:
            raise KeyError(f"Unknown continuation Run: {continuation_run_id}")
        session_id, system_prompt, messages = continuation
        if self._session.session_id != session_id:
            self._session = AgentSession(
                messages,
                system_prompt=system_prompt,
                session_id=session_id,
            )
            self._loop.use_session(self._session)
        else:
            self._session.restore(messages, system_prompt=system_prompt)
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
        events: queue.Queue[AgentEvent | Exception | object] = queue.Queue()
        finished = object()
        completed = False

        def run() -> None:
            try:
                self._execute_prompt(user_input, control, event_sink=events.put)
            except Exception as error:  # noqa: BLE001 - the iterator re-raises worker failures
                events.put(error)
            finally:
                events.put(finished)

        threading.Thread(target=run, name="minimal-agent-stream", daemon=True).start()
        try:
            while True:
                event = events.get()
                if event is finished:
                    completed = True
                    return
                if isinstance(event, Exception):
                    raise event
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
        try:
            return self._run_prompt(run_id, user_input, control, event_sink)
        finally:
            with self._active_lock:
                self._active_control = None
                self._steering_open = False
            self._apply_pending_steering_to_session()
            try:
                self._save_session()
            finally:
                self._session.release_run()

    def _run_prompt(
        self,
        run_id: str,
        user_input: str,
        control: RunControl,
        event_sink: Callable[[AgentEvent], None] | None,
    ) -> RunResult:
        with self._active_lock:
            self._active_control = control
            self._steering_open = True
        if self._runs is not None:
            self._runs.start_run(run_id, self._session.session_id)
        self._sequence = 0
        trace: list[AgentEvent] = []
        started_at = time.perf_counter()
        disabled_listeners: set[int] = set()
        self._session.append(UserMessage(user_input))
        self._save_session()
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
        outcome = self._loop.run(run_id, user_input, control, emit)
        return self._finalize(outcome, run_id, trace, emit)

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
            self._session.append(UserMessage(message))

    def _save_session(self) -> None:
        if self._sessions is not None:
            self._sessions.save_session(
                self._session.session_id,
                self._session.system_prompt,
                self._session.messages,
            )

    def _apply_steering(self, run_id: str, step: int, emit) -> None:
        while True:
            try:
                message = self._steering.get_nowait()
            except queue.Empty:
                return
            self._session.append(UserMessage(message))
            emit(
                EventKind.STEERING_MESSAGE_ACCEPTED,
                {"step": step, "content": message, "run_id": run_id},
            )

    def _finalize(
        self,
        outcome: LoopOutcome,
        run_id: str,
        trace: list[AgentEvent],
        emit,
    ) -> RunResult:
        self._close_steering()
        if outcome.stop_reason is StopReason.FINAL:
            emit(
                EventKind.FINAL_RESPONSE,
                {"step": outcome.steps_used, "content": outcome.final_response},
            )
        elif outcome.error is not None:
            emit(
                EventKind.RUN_ERROR,
                {
                    "stop_reason": outcome.stop_reason.value,
                    "steps_used": outcome.steps_used,
                    "error": outcome.error,
                },
            )
        elif outcome.stop_reason is not StopReason.FINAL:
            emit(
                EventKind.RUN_STOPPED,
                {
                    "stop_reason": outcome.stop_reason.value,
                    "steps_used": outcome.steps_used,
                },
            )
        self._save_session()
        if self._runs is not None:
            self._runs.finish_run(run_id, outcome.stop_reason.value, outcome.steps_used)
        return RunResult(
            outcome.final_response,
            outcome.stop_reason,
            outcome.steps_used,
            run_id,
            outcome.error,
            tuple(trace),
            tuple(MappingProxyType(dict(item)) for item in outcome.context_metadata),
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
        if self._runs is not None:
            self._runs.append_event(run_id, event.sequence, event.kind.value, event.data)
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


def _model_response(
    model: ModelAdapter,
    messages: tuple[ChatMessage, ...],
    tool_definitions,
    max_output_tokens: int,
    emit: Callable[[EventKind, dict[str, object]], None],
    step: int,
) -> ModelResponse:
    if isinstance(getattr(model, "profile", None), ModelProfile):
        events = []

        def attempt_event(event) -> None:
            error = event.error
            emit(
                EventKind(event.kind.value),
                {
                    "step": step,
                    "attempt": event.attempt,
                    "error_kind": error.kind.value if error else None,
                    "retryable": error.retryable if error else None,
                    "delay_seconds": event.delay_seconds,
                },
            )

        request = ModelRequest(
            messages,
            RequestOptions(
                tools=tuple(tool_definitions),
                max_output_tokens=max_output_tokens,
            ),
        )
        stream = (
            model.stream(request, on_attempt=attempt_event)
            if isinstance(model, ProviderClient)
            else model.stream(request)
        )
        for event in stream:
            events.append(event)
            if isinstance(event, TextDelta):
                emit(EventKind.MODEL_CONTENT_DELTA, {"step": step, "content_delta": event.text})
            elif isinstance(event, ToolCallDelta):
                emit(
                    EventKind.TOOL_CALL_DELTA,
                    {
                        "step": step,
                        "tool_call_id": event.call_id,
                        "name": event.name,
                        "arguments_delta": event.arguments_delta,
                    },
                )
        return aggregate_stream(events)
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
            raise ProviderError(
                ProviderErrorKind.INVALID_RESPONSE,
                "Provider returned an invalid stream chunk.",
            )
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
        raise ProviderError(
            ProviderErrorKind.INVALID_RESPONSE,
            "Provider returned an incomplete Tool Call.",
        )
    if not completed:
        raise ProviderError(
            ProviderErrorKind.INVALID_RESPONSE,
            "Provider stream ended before completion.",
        )
    for call_id in order:
        try:
            json.loads(calls[call_id]["arguments"])
        except json.JSONDecodeError as error:
            raise ProviderError(
                ProviderErrorKind.INVALID_RESPONSE,
                "Provider returned malformed Tool Call arguments.",
            ) from error
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
