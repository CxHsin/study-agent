"""Recovery policy for interrupted Tool Executions and Continuation Runs."""

from dataclasses import dataclass
from typing import Protocol

from minimal_agent.persistence import UnresolvedTool


@dataclass(frozen=True)
class ContinuationRun:
    run_id: str
    session_id: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.session_id:
            raise ValueError("Continuation Run identity must not be empty.")


class RecoveryStore(Protocol):
    def mark_interrupted_runs(self) -> None: ...

    def unresolved_tools(self) -> tuple[UnresolvedTool, ...]: ...

    def schedule_tool_retry(
        self,
        tool: UnresolvedTool,
        continuation: ContinuationRun,
    ) -> None: ...

    def store_tool_resolution(
        self,
        tool: UnresolvedTool,
        result: str,
        continuation: ContinuationRun | None,
    ) -> None: ...


@dataclass(frozen=True)
class RecoveryPlan:
    unresolved: tuple[UnresolvedTool, ...]

    @property
    def requires_resolution(self) -> bool:
        return bool(self.unresolved)


class RecoveryCoordinator:
    def __init__(self, store: RecoveryStore) -> None:
        self._store = store

    def recover(self) -> RecoveryPlan:
        self._store.mark_interrupted_runs()
        return RecoveryPlan(self._store.unresolved_tools())

    def retry_tool(
        self,
        run_id: str,
        call_id: str,
        continuation: ContinuationRun,
    ) -> UnresolvedTool:
        tool = self._find(run_id, call_id)
        if not tool.idempotent:
            raise RuntimeError("Only idempotent Tool Executions may be retried automatically.")
        self._store.schedule_tool_retry(tool, continuation)
        return tool

    def resolve_tool(
        self,
        run_id: str,
        call_id: str,
        result: str,
        continuation: ContinuationRun | None = None,
    ) -> None:
        tool = self._find(run_id, call_id)
        self._store.store_tool_resolution(tool, result, continuation)

    def _find(self, run_id: str, call_id: str) -> UnresolvedTool:
        tool = next(
            (
                item
                for item in self._store.unresolved_tools()
                if item.run_id == run_id and item.call_id == call_id
            ),
            None,
        )
        if tool is None:
            raise KeyError(f"Unknown unresolved Tool Execution: {run_id}/{call_id}")
        return tool
