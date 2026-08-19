from minimal_agent.protocol import (
    ModelProfile,
    ModelRequest,
    ProviderError,
    ProviderErrorKind,
    RequestOptions,
    StreamEnd,
    StreamError,
    TextDelta,
    ToolDefinition,
    UserMessage,
)
from minimal_agent.provider_client import ProviderAttemptKind, ProviderClient, RetryPolicy


class ScriptedAdapter:
    def __init__(self, attempts, *, streaming: bool = True) -> None:
        self.profile = ModelProfile("test", "test-model", 4096, 512, streaming=streaming)
        self.attempts = iter(attempts)
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        yield from next(self.attempts)


def test_client_retries_retryable_failure_before_exposing_output() -> None:
    failure = ProviderError(ProviderErrorKind.RATE_LIMIT, "limited", retryable=True)
    adapter = ScriptedAdapter([[StreamError(failure)], [TextDelta("done"), StreamEnd()]])
    attempts = []
    delays = []
    client = ProviderClient(
        adapter,
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.5, jitter=0),
        on_attempt=attempts.append,
        sleep=delays.append,
    )

    response = client.complete(ModelRequest((UserMessage("hello"),)))

    assert response.content == "done"
    assert len(adapter.requests) == 2
    assert delays == [0.5]
    assert [event.kind for event in attempts] == [
        ProviderAttemptKind.STARTED,
        ProviderAttemptKind.FAILED,
        ProviderAttemptKind.RETRY_SCHEDULED,
        ProviderAttemptKind.STARTED,
    ]


def test_client_does_not_retry_after_a_delta_is_exposed() -> None:
    failure = ProviderError(ProviderErrorKind.NETWORK, "disconnected", retryable=True)
    adapter = ScriptedAdapter([[TextDelta("partial"), StreamError(failure)]])
    client = ProviderClient(adapter, sleep=lambda _delay: None)

    try:
        client.complete(ModelRequest((UserMessage("hello"),)))
    except ProviderError as error:
        assert error is failure
    else:
        raise AssertionError("Expected the interrupted stream to fail")

    assert len(adapter.requests) == 1


def test_retry_after_takes_precedence_over_local_backoff() -> None:
    failure = ProviderError(
        ProviderErrorKind.RATE_LIMIT,
        "limited",
        retryable=True,
        retry_after=1.25,
    )
    adapter = ScriptedAdapter([[StreamError(failure)], [TextDelta("ok"), StreamEnd()]])
    delays = []
    client = ProviderClient(
        adapter,
        retry_policy=RetryPolicy(max_attempts=2),
        sleep=delays.append,
    )

    client.complete(ModelRequest((UserMessage("hello"),)))

    assert delays == [1.25]


def test_streaming_capability_falls_back_to_one_shot_request() -> None:
    adapter = ScriptedAdapter([[TextDelta("ok"), StreamEnd()]], streaming=False)
    client = ProviderClient(adapter)

    assert client.complete(ModelRequest((UserMessage("hello"),))).content == "ok"
    assert adapter.requests[0].options.stream is False


def test_tool_calls_fail_when_profile_does_not_support_them() -> None:
    adapter = ScriptedAdapter([])
    adapter.profile = ModelProfile("test", "no-tools", 4096, 512, tool_calls=False)
    tool = ToolDefinition("read", "Read", {"type": "object"})
    request = ModelRequest((UserMessage("read"),), RequestOptions(tools=(tool,)))

    try:
        ProviderClient(adapter).complete(request)
    except ProviderError as error:
        assert error.kind is ProviderErrorKind.UNSUPPORTED_CAPABILITY
    else:
        raise AssertionError("Expected unsupported Tool Calls to fail")
