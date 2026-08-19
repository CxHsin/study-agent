from minimal_agent.protocol import (
    MESSAGE_SCHEMA_VERSION,
    AssistantMessage,
    ContextSummaryMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    message_from_dict,
    message_to_dict,
    normalize_messages,
)


def test_v1_messages_normalize_to_immutable_records_and_write_v2() -> None:
    messages = normalize_messages(
        [
            {"role": "user", "content": "read"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "read", "arguments": "{}"}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": '{"ok":true}'},
        ]
    )

    assert messages[0] == UserMessage("read")
    assert isinstance(messages[1], AssistantMessage)
    assert isinstance(messages[2], ToolResultMessage)
    assert messages[2].result.tool_name == "read"
    assert message_to_dict(messages[0])["schema_version"] == MESSAGE_SCHEMA_VERSION


def test_unknown_message_schema_version_is_rejected() -> None:
    try:
        message_from_dict({"schema_version": 99, "role": "user", "content": "hello"})
    except ValueError as error:
        assert "Unsupported message schema version" in str(error)
    else:
        raise AssertionError("Expected an unknown schema version to fail")


def test_tool_results_must_follow_a_unique_preceding_call() -> None:
    invalid = [
        AssistantMessage(None, (ToolCall("c1", "read", "{}"),)),
        UserMessage("skip the result"),
    ]

    try:
        normalize_messages(invalid)
    except ValueError as error:
        assert "pending Tool Results" in str(error)
    else:
        raise AssertionError("Expected invalid Tool Result ordering to fail")


def test_context_summary_retains_its_identity_across_serialization() -> None:
    message = ContextSummaryMessage("facts", "summary-1")

    assert message_from_dict(message_to_dict(message)) == message
