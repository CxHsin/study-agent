"""Provider-independent model/tool execution loop."""

import json
import time
from collections.abc import Callable

from minimal_agent.context import ContextBuilder, ContextError
from minimal_agent.cost import PromptCacheStore, checkpoint_for, usage_record
from minimal_agent.events import AgentEvent, EventKind
from minimal_agent.persistence import ToolLedgerRepository
from minimal_agent.protocol import (
    AssistantMessage,
    ModelAdapter,
    ModelResponse,
    ProviderCapabilities,
    ProviderError,
    ToolCall,
    ToolResultMessage,
)
from minimal_agent.run import RunControl, RunResult, StopReason
from minimal_agent.session import AgentSession
from minimal_agent.tools import ToolError, ToolRegistry, ToolResult

EventSink = Callable[[AgentEvent], None]
Emit = Callable[[EventKind, dict[str, object]], None]
ResultFactory = Callable[
    [str, StopReason, int, list[AgentEvent], Emit, list[dict[str, object]]], RunResult
]
ErrorFactory = Callable[..., RunResult]
FinishFactory = Callable[[RunResult, list[AgentEvent]], RunResult]
ModelResponseFactory = Callable[..., ModelResponse]
CapabilitiesFactory = Callable[[ModelAdapter], ProviderCapabilities]


class AgentLoop:
    """Execute one bounded model -> tool -> model cycle."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        tools: ToolRegistry,
        session: AgentSession,
        context_builder: ContextBuilder,
        max_steps: int,
        tool_ledger: ToolLedgerRepository | None,
        cache_store: PromptCacheStore,
        apply_steering: Callable[[str, int, Emit], None],
        close_steering: Callable[[], None],
        result: ResultFactory,
        error: ErrorFactory,
        finish: FinishFactory,
        model_response: ModelResponseFactory,
        provider_capabilities: CapabilitiesFactory,
    ) -> None:
        self._model = model
        self._tools = tools
        self._session = session
        self._context_builder = context_builder
        self._max_steps = max_steps
        self._tool_ledger = tool_ledger
        self._cache_store = cache_store
        self._apply_steering = apply_steering
        self._close_steering = close_steering
        self._result = result
        self._error = error
        self._finish = finish
        self._model_response = model_response
        self._provider_capabilities = provider_capabilities

    def use_session(self, session: AgentSession) -> None:
        self._session = session

    def run(
        self,
        run_id: str,
        user_input: str,
        control: RunControl,
        emit: Emit,
        trace: list[AgentEvent],
    ) -> RunResult:
        context_metadata: list[dict[str, object]] = []
        last_fingerprint: str | None = None
        pending_repeat: str | None = None
        seen_tool_call_ids = {
            call.id
            for message in self._session.messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        }
        try:
            for step in range(1, self._max_steps + 1):
                stop = control.stop_reason
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
                    response = self._model_response(
                        self._model,
                        context.messages,
                        self._tools.definitions(),
                        self._context_builder.config.reserved_output_tokens,
                        emit,
                        step,
                    )
                    capabilities = self._provider_capabilities(self._model)
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
                except ProviderError as error:
                    return self._error(
                        run_id,
                        error.kind.value.upper(),
                        str(error) or "Model request failed.",
                        "provider_error",
                        step,
                        provider_error=error,
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
                stop = control.stop_reason
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
                    self._session.append(AssistantMessage(response.content))
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

                fingerprints = [_fingerprint(call) for call in response.tool_calls]
                if any(call.id in seen_tool_call_ids for call in response.tool_calls):
                    return self._error(
                        run_id,
                        "REPEATED_TOOL_CALL",
                        "Model repeated a Tool Call.",
                        "control_error",
                        step,
                        StopReason.REPEATED_TOOL_CALL,
                        trace=trace,
                        emit=emit,
                        context_metadata=context_metadata,
                    )
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
                self._session.append(AssistantMessage(response.content, response.tool_calls))
                seen_tool_call_ids.update(call.id for call in response.tool_calls)
                for tool_call in response.tool_calls:
                    stop = control.stop_reason
                    if stop:
                        return self._result(run_id, stop, step, trace, emit, context_metadata)
                    fingerprint = _fingerprint(tool_call)
                    if fingerprint == last_fingerprint:
                        tool_result = ToolResult(
                            tool_call.id,
                            tool_call.name,
                            False,
                            error=ToolError("REPEATED_TOOL_CALL", "Repeated tool call blocked."),
                        )
                        self._session.append(ToolResultMessage(tool_result))
                        emit(EventKind.TOOL_RESULT_PRODUCED, _tool_result_data(tool_result))
                        pending_repeat = fingerprint
                        last_fingerprint = fingerprint
                        continue
                    pending_repeat = None
                    last_fingerprint = fingerprint
                    definition = self._tools.get(tool_call.name)
                    emit(EventKind.TOOL_CALL_REQUESTED, _tool_data(tool_call))
                    if self._tool_ledger is not None:
                        self._tool_ledger.record_tool_started(
                            run_id,
                            tool_call.id,
                            tool_call.name,
                            tool_call.arguments,
                            idempotent=definition.idempotent if definition else False,
                        )
                    tool_result = self._tools.execute(
                        tool_call, run_id=run_id, user_input=user_input, control=control
                    )
                    if self._tool_ledger is not None:
                        self._tool_ledger.record_tool(
                            run_id,
                            tool_call.id,
                            tool_call.name,
                            tool_call.arguments,
                            "completed",
                            tool_result.to_json(),
                            idempotent=definition.idempotent if definition else False,
                        )
                    self._session.append(ToolResultMessage(tool_result))
                    emit(EventKind.TOOL_RESULT_PRODUCED, _tool_result_data(tool_result))
                    stop = control.stop_reason
                    if stop:
                        return self._result(run_id, stop, step, trace, emit, context_metadata)
            return self._result(
                run_id, StopReason.MAX_STEPS, self._max_steps, trace, emit, context_metadata
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


def _fingerprint(tool_call: ToolCall) -> str:
    try:
        arguments = json.dumps(
            json.loads(tool_call.arguments), sort_keys=True, separators=(",", ":")
        )
    except (TypeError, json.JSONDecodeError):
        arguments = tool_call.arguments.strip()
    return f"{tool_call.name}\0{arguments}"
