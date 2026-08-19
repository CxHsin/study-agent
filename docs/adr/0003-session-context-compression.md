---
status: accepted
---

# Keep Complete Session History and Derive Compressed Model Context

Conversation Sessions retain their complete Message History and fixed system prompt. A
provider-independent Context Builder derives the messages sent to each model call. When
the configured utilization threshold is reached, it may replace an old, closed history
prefix with a versioned Context Summary while preserving the original Session history.
Summary failures and irreducible budget overflow are structured Context Errors.

**Considered Options**

- Mutate Session history in place when trimming or summarizing.
- Let each Provider Adapter own context limits and compression.
- Keep complete history and derive an observable, provider-independent compressed context (chosen).

**Consequences**

- Context decisions must be recorded in Trace/RunResult metadata.
- Summarizers are separate capabilities and may fail independently of the main model.
- Providers must translate the internal context-summary message representation.
