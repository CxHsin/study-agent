from minimal_agent.persistence import UnresolvedTool
from minimal_agent.recovery import ContinuationRun, RecoveryCoordinator


class RecoveryStoreFake:
    def __init__(self, tools: tuple[UnresolvedTool, ...]) -> None:
        self.tools = tools
        self.marked = False
        self.retries = []
        self.resolutions = []

    def mark_interrupted_runs(self) -> None:
        self.marked = True

    def unresolved_tools(self) -> tuple[UnresolvedTool, ...]:
        return self.tools

    def schedule_tool_retry(self, tool, continuation) -> None:
        self.retries.append((tool, continuation))

    def store_tool_resolution(self, tool, result, continuation) -> None:
        self.resolutions.append((tool, result, continuation))


def test_recovery_coordinator_builds_plan_and_enforces_retry_policy() -> None:
    unsafe = UnresolvedTool("run", "unsafe", "write", "{}", False)
    safe = UnresolvedTool("run", "safe", "read", "{}", True)
    store = RecoveryStoreFake((unsafe, safe))
    recovery = RecoveryCoordinator(store)

    plan = recovery.recover()
    try:
        recovery.retry_tool("run", "unsafe", ContinuationRun("child-unsafe", "session"))
    except RuntimeError as error:
        assert "idempotent" in str(error)
    else:
        raise AssertionError("Expected unsafe retry to be rejected")
    continuation = ContinuationRun("child-safe", "session")
    recovery.retry_tool("run", "safe", continuation)

    assert store.marked is True
    assert plan.requires_resolution is True
    assert store.retries == [(safe, continuation)]


def test_recovery_resolution_uses_a_typed_continuation_identity() -> None:
    tool = UnresolvedTool("run", "call", "write", "{}", False)
    store = RecoveryStoreFake((tool,))
    recovery = RecoveryCoordinator(store)

    continuation = ContinuationRun("child", "session")
    recovery.resolve_tool("run", "call", '{"ok":true}', continuation)

    assert store.resolutions == [(tool, '{"ok":true}', continuation)]
