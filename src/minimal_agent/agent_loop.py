"""Execution seam for one Agent Loop run.

The first iteration keeps the mature loop implementation in ``AgentCore``
behind this small interface.  This lets callers and tests depend on the loop
contract while the implementation is migrated incrementally.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from minimal_agent.events import AgentEvent

if TYPE_CHECKING:
    from minimal_agent.core import RunControl, RunResult


EventSink = Callable[[AgentEvent], None]
LoopImplementation = Callable[[str, str, "RunControl", EventSink | None], "RunResult"]


class AgentLoop:
    """Small interface for executing a single bounded model/tool loop."""

    def __init__(self, implementation: LoopImplementation) -> None:
        self._implementation = implementation

    def run(
        self,
        run_id: str,
        user_input: str,
        control: "RunControl",
        event_sink: EventSink | None = None,
    ) -> "RunResult":
        return self._implementation(run_id, user_input, control, event_sink)
