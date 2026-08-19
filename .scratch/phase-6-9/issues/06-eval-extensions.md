# 06 — Eval 扩展与阶段 6–9 验收

**What to build:** Existing Eval Cases and Artifacts can optionally assert streaming, interaction, Provider capability, recovery, usage, and cache behavior while all existing cases remain valid.

**Blocked by:** 01 — Streaming Event 基础与兼容同步 API; 02 — Steering、Follow-up 与 Session 串行控制; 03 — Provider Capability 与适配器契约; 04 — SQLite Repository 与恢复安全边界; 05 — Prompt Cache Checkpoint 与 Usage Record

**Status:** ready-for-agent

- [ ] Existing Eval Case loading, execution, trajectory checks, and artifact redaction remain compatible.
- [ ] Optional assertions cover event order, Steering/Follow-up, capabilities, recovery, usage, and cache sources.
- [ ] Provider unavailable/unsupported evidence remains inconclusive or estimated as appropriate.
- [ ] Deterministic tests cover the complete phase 6–9 acceptance surface.
