# 01 — Streaming Event 基础与兼容同步 API

**What to build:** The Agent Core exposes one provider-independent event stream that can be consumed incrementally while preserving the existing synchronous prompt result and listener behavior.

**Blocked by:** None — can start immediately.

**Status:** done

- [ ] A stream consumer can observe ordered lifecycle and incremental model events for a Run.
- [ ] `prompt()` aggregates the same stream into the existing RunResult contract.
- [ ] Listener failures remain isolated and terminal events are emitted consistently.
- [ ] Partial Tool Call data is buffered and only complete, valid calls reach Tool execution.
- [ ] Existing Core, event, and Tool tests remain green, with deterministic streaming coverage added.
