from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import cast
from uuid import uuid4

from minimal_agent.cost import ModelCallPurpose, UsageRecord, UsageStatus, usage_record
from minimal_agent.protocol import (
    ModelAdapter,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelStreamChunk,
    ProviderCapabilities,
    ProviderError,
    ProviderErrorKind,
    ProviderStreamEvent,
    StreamEnd,
    StreamError,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    UsageUpdate,
)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.25
    max_delay: float = 4.0
    jitter: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive.")
        if min(self.base_delay, self.max_delay, self.jitter) < 0:
            raise ValueError("Retry delays cannot be negative.")


class ProviderAttemptKind(StrEnum):
    STARTED = "provider_attempt_started"
    FAILED = "provider_attempt_failed"
    RETRY_SCHEDULED = "provider_retry_scheduled"


@dataclass(frozen=True)
class ProviderAttemptEvent:
    kind: ProviderAttemptKind
    attempt: int
    error: ProviderError | None = None
    delay_seconds: float | None = None


AttemptListener = Callable[[ProviderAttemptEvent], None]
CallListener = Callable[[UsageRecord], None]


@dataclass(frozen=True)
class CallObservationScope:
    listener: CallListener
    step: int


class LegacyModelAdapter:
    """Translate the former in-process model shape into the Provider protocol."""

    def __init__(self, model: object) -> None:
        complete = getattr(model, "complete", None)
        stream = getattr(model, "stream", None)
        if not callable(complete) and not callable(stream):
            raise TypeError("Model must provide complete() or stream().")
        self._complete = complete if callable(complete) else None
        self._stream = stream if callable(stream) else None
        capabilities = _legacy_capabilities(model)
        self.profile = ModelProfile(
            "in_process",
            type(model).__name__,
            1_000_000,
            1_000_000,
            streaming=capabilities.streaming,
            tool_calls=capabilities.tool_calls,
            parallel_tool_calls=capabilities.parallel_tool_calls,
            usage=capabilities.usage,
            prompt_cache=capabilities.prompt_cache,
            cancellation=capabilities.cancellation,
        )

    def stream(self, request: ModelRequest) -> Iterator[ProviderStreamEvent]:
        try:
            if self.profile.streaming and self._stream is not None:
                yield from _legacy_chunk_events(self._stream(request.messages))
                return
            if self._complete is None:
                yield StreamError(
                    ProviderError(
                        ProviderErrorKind.INVALID_REQUEST, "Model cannot complete requests."
                    )
                )
                return
            yield from _legacy_response_events(self._complete(request.messages))
        except ProviderError as error:
            yield StreamError(error)
        except Exception as error:  # noqa: BLE001 - compatibility models expose their own failures
            yield StreamError(
                ProviderError(ProviderErrorKind.UNKNOWN, str(error) or type(error).__name__)
            )


class ProviderClient:
    def __init__(
        self,
        adapter: ModelAdapter,
        *,
        retry_policy: RetryPolicy | None = None,
        attempt_timeout: float = 30.0,
        total_timeout: float = 60.0,
        on_attempt: AttemptListener | None = None,
        report_attempts: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if attempt_timeout <= 0 or total_timeout <= 0:
            raise ValueError("Provider timeouts must be positive.")
        self.adapter = adapter
        self.profile = adapter.profile
        self.retry_policy = retry_policy or RetryPolicy()
        self.attempt_timeout = attempt_timeout
        self.total_timeout = total_timeout
        self._on_attempt = on_attempt
        self._report_attempts = report_attempts
        self._sleep = sleep
        self._monotonic = monotonic
        self._random = random_value
        self._call_scopes = threading.local()

    @property
    def model_name(self) -> str:
        return self.profile.model

    def capabilities(self) -> ProviderCapabilities:
        return self.profile.capabilities

    def complete(self, request: ModelRequest) -> ModelResponse:
        return self.invoke(request)

    def invoke(
        self,
        request: ModelRequest,
        *,
        on_event: Callable[[ProviderStreamEvent], None] | None = None,
        on_attempt: AttemptListener | None = None,
    ) -> ModelResponse:
        started = self._monotonic()
        call_id = str(uuid4())
        scope = self._active_call_scope()
        metadata = request.options.metadata
        step = scope.step if scope is not None else int(metadata.get("step", 0))
        purpose = ModelCallPurpose(metadata.get("call_purpose", ModelCallPurpose.AGENT))
        try:
            events = []
            for event in self.stream(request, on_attempt=on_attempt):
                events.append(event)
                if on_event is not None:
                    on_event(event)
            response = aggregate_stream(events)
        except ProviderError as error:
            self._notify_call(
                UsageRecord(
                    None,
                    None,
                    None,
                    (self._monotonic() - started) * 1000,
                    None,
                    "unknown",
                    "unknown",
                    step,
                    call_id,
                    purpose,
                    UsageStatus.FAILED,
                    error.kind.value,
                ),
                scope,
            )
            raise
        self._notify_call(
            usage_record(
                response,
                latency_ms=(self._monotonic() - started) * 1000,
                capabilities=self.capabilities(),
                step=step,
                call_id=call_id,
                purpose=purpose,
            ),
            scope,
        )
        return response

    @contextmanager
    def observe_calls(self, listener: CallListener, *, step: int) -> Iterator[None]:
        scopes = getattr(self._call_scopes, "value", [])
        scope = CallObservationScope(listener, step)
        self._call_scopes.value = [*scopes, scope]
        try:
            yield
        finally:
            self._call_scopes.value = scopes

    def _active_call_scope(self) -> CallObservationScope | None:
        scopes = getattr(self._call_scopes, "value", ())
        return scopes[-1] if scopes else None

    def _notify_call(self, usage: UsageRecord, scope: CallObservationScope | None) -> None:
        if scope is not None:
            scope.listener(usage)

    def stream(
        self, request: ModelRequest, *, on_attempt: AttemptListener | None = None
    ) -> Iterator[ProviderStreamEvent]:
        self._validate(request)
        if request.options.stream and not self.profile.streaming:
            request = replace(request, options=replace(request.options, stream=False))
        listener = (on_attempt or self._on_attempt) if self._report_attempts else None
        started = self._monotonic()
        for attempt in range(1, self.retry_policy.max_attempts + 1):
            remaining = self.total_timeout - (self._monotonic() - started)
            if remaining <= 0:
                yield StreamError(
                    ProviderError(
                        ProviderErrorKind.TIMEOUT,
                        "Provider call exceeded its total deadline.",
                        retryable=True,
                    )
                )
                return
            self._notify(ProviderAttemptEvent(ProviderAttemptKind.STARTED, attempt), listener)
            request_for_attempt = _with_timeout(request, min(self.attempt_timeout, remaining))
            exposed = False
            terminal_error: ProviderError | None = None
            try:
                for event in self.adapter.stream(request_for_attempt):
                    if isinstance(event, StreamError):
                        terminal_error = event.error
                        break
                    exposed = True
                    yield event
            except ProviderError as error:
                terminal_error = error
            except Exception as error:  # noqa: BLE001 - adapter boundary normalizes unknown failures
                terminal_error = ProviderError(
                    ProviderErrorKind.UNKNOWN,
                    "Provider adapter failed.",
                    retryable=False,
                )
                terminal_error.__cause__ = error
            if terminal_error is None:
                return
            self._notify(
                ProviderAttemptEvent(ProviderAttemptKind.FAILED, attempt, error=terminal_error),
                listener,
            )
            if exposed or not terminal_error.retryable or attempt >= self.retry_policy.max_attempts:
                yield StreamError(terminal_error)
                return
            delay = self._retry_delay(terminal_error, attempt)
            remaining = self.total_timeout - (self._monotonic() - started)
            if delay >= remaining:
                yield StreamError(
                    ProviderError(
                        ProviderErrorKind.TIMEOUT,
                        "Provider retry would exceed the total deadline.",
                        retryable=True,
                    )
                )
                return
            self._notify(
                ProviderAttemptEvent(
                    ProviderAttemptKind.RETRY_SCHEDULED,
                    attempt,
                    error=terminal_error,
                    delay_seconds=delay,
                ),
                listener,
            )
            self._sleep(delay)

    def _validate(self, request: ModelRequest) -> None:
        options = request.options
        if options.tools and not self.profile.tool_calls:
            raise ProviderError(
                ProviderErrorKind.UNSUPPORTED_CAPABILITY,
                f"Model {self.profile.model} does not support Tool Calls.",
            )
        output = options.max_output_tokens or self.profile.max_output_tokens
        if output > self.profile.max_output_tokens:
            raise ProviderError(
                ProviderErrorKind.INVALID_REQUEST,
                "Requested output tokens exceed the Model Profile limit.",
            )

    def _retry_delay(self, error: ProviderError, attempt: int) -> float:
        if error.retry_after is not None:
            return max(0.0, error.retry_after)
        base = min(self.retry_policy.max_delay, self.retry_policy.base_delay * 2 ** (attempt - 1))
        return base + base * self.retry_policy.jitter * self._random()

    def _notify(self, event: ProviderAttemptEvent, listener: AttemptListener | None) -> None:
        if listener is not None:
            listener(event)


def provider_client_for(model: object) -> ProviderClient:
    if isinstance(model, ProviderClient):
        return model
    if isinstance(getattr(model, "profile", None), ModelProfile) and callable(
        getattr(model, "stream", None)
    ):
        return ProviderClient(cast(ModelAdapter, model))
    return ProviderClient(LegacyModelAdapter(model), report_attempts=False)


def aggregate_stream(events) -> ModelResponse:
    content: list[str] = []
    calls: dict[str, dict[str, str]] = {}
    order: list[str] = []
    usage = None
    cache_hit = None
    ended = False
    for event in events:
        if isinstance(event, TextDelta):
            content.append(event.text)
        elif isinstance(event, ToolCallDelta):
            if event.call_id not in calls:
                calls[event.call_id] = {"name": event.name or "", "arguments": ""}
                order.append(event.call_id)
            call = calls[event.call_id]
            if event.name:
                call["name"] = event.name
            call["arguments"] += event.arguments_delta
        elif isinstance(event, UsageUpdate):
            usage = event.usage
            cache_hit = event.provider_cache_hit
        elif isinstance(event, StreamError):
            raise event.error
        elif isinstance(event, StreamEnd):
            ended = True
    if not ended:
        raise ProviderError(
            ProviderErrorKind.INVALID_RESPONSE,
            "Provider stream ended without a completion event.",
        )
    tool_calls: list[ToolCall] = []
    for call_id in order:
        call = calls[call_id]
        if not call["name"] or not call["arguments"]:
            raise ProviderError(
                ProviderErrorKind.INVALID_RESPONSE,
                "Provider returned an incomplete Tool Call.",
            )
        try:
            json.loads(call["arguments"])
        except json.JSONDecodeError as error:
            raise ProviderError(
                ProviderErrorKind.INVALID_RESPONSE,
                "Provider returned malformed Tool Call arguments.",
            ) from error
        tool_calls.append(ToolCall(call_id, call["name"], call["arguments"]))
    text = "".join(content) or None
    if text is None and not tool_calls:
        raise ProviderError(
            ProviderErrorKind.INVALID_RESPONSE,
            "Provider returned neither text nor Tool Calls.",
        )
    return ModelResponse(text, tuple(tool_calls), usage, cache_hit)


def _legacy_capabilities(model: object) -> ProviderCapabilities:
    callback = getattr(model, "capabilities", None)
    if callable(callback):
        capabilities = callback()
        if isinstance(capabilities, ProviderCapabilities):
            return capabilities
    return ProviderCapabilities(streaming=callable(getattr(model, "stream", None)))


def _legacy_response_events(response: object) -> Iterator[ProviderStreamEvent]:
    if not isinstance(response, ModelResponse):
        yield StreamError(
            ProviderError(ProviderErrorKind.INVALID_RESPONSE, "Model returned an invalid response.")
        )
        return
    if response.content:
        yield TextDelta(response.content)
    for call in response.tool_calls:
        try:
            json.loads(call.arguments)
            arguments = call.arguments
        except json.JSONDecodeError:
            arguments = json.dumps(call.arguments, ensure_ascii=False)
        yield ToolCallDelta(call.id, call.name, arguments)
    if response.usage is not None:
        yield UsageUpdate(response.usage, response.provider_cache_hit)
    yield StreamEnd()


def _legacy_chunk_events(
    chunks: Iterable[ModelStreamChunk],
) -> Iterator[ProviderStreamEvent]:
    ended = False
    for chunk in chunks:
        if not isinstance(chunk, ModelStreamChunk):
            yield StreamError(
                ProviderError(
                    ProviderErrorKind.INVALID_RESPONSE,
                    "Model returned an invalid stream chunk.",
                )
            )
            return
        if chunk.content_delta:
            yield TextDelta(chunk.content_delta)
        if chunk.tool_call_id is not None:
            yield ToolCallDelta(chunk.tool_call_id, chunk.tool_name, chunk.arguments_delta)
        if chunk.usage is not None:
            yield UsageUpdate(chunk.usage, chunk.provider_cache_hit)
        if chunk.done:
            ended = True
    if ended:
        yield StreamEnd()
    else:
        yield StreamError(
            ProviderError(
                ProviderErrorKind.INVALID_RESPONSE,
                "Model stream ended before completion.",
            )
        )


def _with_timeout(request: ModelRequest, timeout: float) -> ModelRequest:
    metadata = dict(request.options.metadata)
    metadata["request_timeout_seconds"] = timeout
    return replace(request, options=replace(request.options, metadata=metadata))
