"""Prompt-prefix checkpoint and model usage accounting."""

import hashlib
import json
from dataclasses import dataclass

from minimal_agent.protocol import ChatMessage, ModelResponse, ModelUsage, ProviderCapabilities


@dataclass(frozen=True)
class PromptCacheCheckpoint:
    key: str
    session_id: str
    message_index: int
    model: str
    tool_schema_hash: str
    system_prompt_hash: str
    context_builder: str


@dataclass(frozen=True)
class UsageRecord:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    latency_ms: float | None
    cost: float | None
    source: str
    cache_hit_source: str


def checkpoint_for(
    messages: list[ChatMessage] | tuple[ChatMessage, ...],
    *,
    session_id: str,
    message_index: int,
    model: str,
    tool_schema: object = (),
    system_prompt: str | None = None,
    context_builder: str = "default",
) -> PromptCacheCheckpoint:
    prefix = messages[:message_index]
    normalized = json.dumps(
        prefix, ensure_ascii=False, sort_keys=True, default=repr, separators=(",", ":")
    )
    schema = json.dumps(
        tool_schema, ensure_ascii=False, sort_keys=True, default=repr, separators=(",", ":")
    )
    prefix_hash = hashlib.sha256(normalized.encode()).hexdigest()
    schema_hash = hashlib.sha256(schema.encode()).hexdigest()
    system_hash = hashlib.sha256((system_prompt or "").encode()).hexdigest()
    key_material = (
        f"{session_id}|{prefix_hash}|{model}|{schema_hash}|{system_hash}|{context_builder}"
    )
    return PromptCacheCheckpoint(
        hashlib.sha256(key_material.encode()).hexdigest(),
        session_id,
        message_index,
        model,
        schema_hash,
        system_hash,
        context_builder,
    )


def usage_record(
    response: ModelResponse,
    *,
    estimated_input: int | None = None,
    estimated_output: int | None = None,
    latency_ms: float | None = None,
    cost: float | None = None,
    capabilities: ProviderCapabilities | None = None,
    local_cache_hit: bool = False,
) -> UsageRecord:
    usage: ModelUsage | None = response.usage
    reported = usage is not None and not usage.estimated
    return UsageRecord(
        usage.input_tokens if usage else estimated_input,
        usage.output_tokens if usage else estimated_output,
        usage.cached_tokens if usage else None,
        usage.latency_ms if usage and usage.latency_ms is not None else latency_ms,
        usage.cost if usage and usage.cost is not None else cost,
        "provider"
        if reported
        else "estimated"
        if usage or estimated_input is not None
        else "unknown",
        "both"
        if local_cache_hit and response.provider_cache_hit
        else "local"
        if local_cache_hit
        else "provider"
        if response.provider_cache_hit is True
        and (capabilities is None or capabilities.prompt_cache)
        else "unknown",
    )


class PromptCacheStore:
    def __init__(self) -> None:
        self._checkpoints: dict[str, PromptCacheCheckpoint] = {}

    def lookup(self, checkpoint: PromptCacheCheckpoint) -> bool:
        return checkpoint.key in self._checkpoints

    def record(self, checkpoint: PromptCacheCheckpoint) -> None:
        self._checkpoints[checkpoint.key] = checkpoint
