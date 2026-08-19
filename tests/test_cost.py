from minimal_agent.cost import checkpoint_for, usage_record
from minimal_agent.protocol import ModelResponse, ModelUsage, ProviderCapabilities


def test_checkpoint_is_stable_and_sensitive_to_contract_inputs() -> None:
    messages = [{"role": "user", "content": "hello"}]
    first = checkpoint_for(messages, session_id="s", message_index=1, model="m")
    same = checkpoint_for(messages, session_id="s", message_index=1, model="m")
    changed = checkpoint_for(messages, session_id="s", message_index=1, model="other")

    assert first == same
    assert first.key != changed.key


def test_usage_distinguishes_provider_and_estimated_values() -> None:
    reported = usage_record(
        ModelResponse(
            content="ok",
            usage=ModelUsage(input_tokens=4, output_tokens=2, cached_tokens=1),
            provider_cache_hit=True,
        ),
        capabilities=ProviderCapabilities(usage=True, prompt_cache=True),
    )
    estimated = usage_record(ModelResponse(content="ok"), estimated_input=10, estimated_output=3)

    assert reported.source == "provider"
    assert reported.cache_hit_source == "provider"
    assert estimated.source == "estimated"
    assert estimated.input_tokens == 10
