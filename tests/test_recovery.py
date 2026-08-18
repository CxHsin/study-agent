import pytest

from minimal_agent.agent import ToolExecutionStatus
from minimal_agent.recovery import RecoveryAction, ToolRetryPolicy, policy_for_tool
from minimal_agent.workspace_tools import WorkspaceTools


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ToolExecutionStatus.PENDING, RecoveryAction.RETRY),
        (ToolExecutionStatus.STARTED, RecoveryAction.RETRY),
    ],
)
def test_idempotent_tool_can_recover_incomplete_execution(
    status: ToolExecutionStatus, expected: RecoveryAction
) -> None:
    policy = ToolRetryPolicy(is_idempotent=True, retry_allowed=True)

    assert policy.decide(status) is expected


@pytest.mark.parametrize("status", list(ToolExecutionStatus))
def test_non_idempotent_tool_requires_resolution(status: ToolExecutionStatus) -> None:
    policy = ToolRetryPolicy(is_idempotent=False, retry_allowed=True)

    expected = (
        RecoveryAction.STOP
        if status
        in {
            ToolExecutionStatus.SUCCEEDED,
            ToolExecutionStatus.FAILED,
        }
        else RecoveryAction.RESOLVE
    )
    assert policy.decide(status) is expected


def test_retry_can_be_disabled_even_for_idempotent_tool() -> None:
    policy = ToolRetryPolicy(is_idempotent=True, retry_allowed=False)

    assert policy.decide(ToolExecutionStatus.STARTED) is RecoveryAction.RESOLVE


def test_tool_metadata_controls_policy(tmp_path) -> None:
    policy = policy_for_tool(WorkspaceTools(tmp_path), "read_file")

    assert policy.decide(ToolExecutionStatus.STARTED) is RecoveryAction.RETRY


def test_unknown_tool_metadata_fails_closed() -> None:
    policy = policy_for_tool(object(), "unknown")

    assert policy.decide(ToolExecutionStatus.STARTED) is RecoveryAction.RESOLVE
