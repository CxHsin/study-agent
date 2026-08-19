# 05 — Prompt Cache Checkpoint 与 Usage Record

**What to build:** Each model call can report token, latency, cost, and cache-hit-source data while stable normalized message-prefix checkpoints are recorded and invalidated when model-facing contracts change.

**Blocked by:** 03 — Provider Capability 与适配器契约; 04 — SQLite Repository 与恢复安全边界

**Status:** done

- [ ] Usage records distinguish Provider-reported, estimated, and unknown values.
- [ ] Local checkpoints use normalized prefix identity and record relevant invalidation inputs.
- [ ] Local and Provider cache hits are reported separately.
- [ ] Checkpoints invalidate on system prompt, model, tool/schema, context-builder, or prefix changes.
- [ ] Cost and latency metrics are deterministic in tests and do not claim unsupported computation reuse.
