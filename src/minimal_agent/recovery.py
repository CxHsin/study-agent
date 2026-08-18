from dataclasses import dataclass
from enum import StrEnum

from minimal_agent.agent import ToolExecutionStatus


class RecoveryAction(StrEnum):
    RETRY = "retry"
    RESOLVE = "resolve"
    STOP = "stop"


@dataclass(frozen=True)
class ToolRetryPolicy:
    is_idempotent: bool
    retry_allowed: bool

    def decide(self, status: ToolExecutionStatus) -> RecoveryAction:
        if status in {ToolExecutionStatus.SUCCEEDED, ToolExecutionStatus.FAILED}:
            return RecoveryAction.STOP
        if not self.is_idempotent or not self.retry_allowed:
            return RecoveryAction.RESOLVE
        if status in {ToolExecutionStatus.PENDING, ToolExecutionStatus.STARTED}:
            return RecoveryAction.RETRY
        return RecoveryAction.RESOLVE


def policy_for_tool(tool_executor: object, tool_name: str) -> ToolRetryPolicy:
    provider = getattr(tool_executor, "retry_policy", None)
    if callable(provider):
        policy = provider(tool_name)
        if isinstance(policy, ToolRetryPolicy):
            return policy
    return ToolRetryPolicy(is_idempotent=False, retry_allowed=False)
