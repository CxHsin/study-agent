# 04 — SQLite Repository 与恢复安全边界

**What to build:** Session, Run, event, and Tool lifecycle records survive restart in a versioned SQLite Repository, with redaction, safe recovery, idempotent retry policy, Tool Resolution, and linked Continuation Runs.

**Blocked by:** 02 — Steering、Follow-up 与 Session 串行控制; 03 — Provider Capability 与适配器契约

**Status:** ready-for-agent

- [ ] Repository writes append-only events and current projections with explicit schema/format versions.
- [ ] Requested, started, and completed Tool boundaries are persisted transactionally with Run state.
- [ ] Restart identifies uncertain Tool executions and avoids silent replay.
- [ ] Idempotent retry and explicit Tool Resolution create linked Continuation Runs.
- [ ] Default redaction protects persisted messages, arguments, results, usage, and errors.
- [ ] SQLite and in-memory Repository behavior are covered without network dependencies.
