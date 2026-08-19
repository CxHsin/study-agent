# 02 — Steering、Follow-up 与 Session 串行控制

**What to build:** An active Run accepts ordered Steering at model boundaries, completed Runs accept Follow-up as a new Run, and a Conversation Session rejects concurrent active Runs with explicit disconnect cancellation behavior.

**Blocked by:** 01 — Streaming Event 基础与兼容同步 API

**Status:** done

- [ ] Steering is ordered, persisted in Session history, and applied at the next model boundary.
- [ ] Steering is queued while a Tool runs and is rejected after termination.
- [ ] Follow-up creates a new Run in the same Session with independent run identity.
- [ ] A Session permits only one active Run and reports a stable busy outcome.
- [ ] Abandoning a stream requests cooperative cancellation by default.
