# 03 — Provider Capability 与适配器契约

**What to build:** Provider Adapters declare optional capabilities and normalize streaming, cancellation, usage, cache metadata, and errors behind a common contract with configurable per-capability fallback behavior.

**Blocked by:** 01 — Streaming Event 基础与兼容同步 API

**Status:** ready-for-agent

- [ ] Providers expose explicit capability metadata without changing the existing completion contract.
- [ ] Streaming and non-streaming responses normalize to the Core event protocol.
- [ ] Optional cancellation, usage, and cache metadata are represented when supported.
- [ ] Missing capabilities follow explicit fallback policies and are never silently fabricated.
- [ ] DeepSeek is covered as the reference adapter and contract tests define the OpenAI/Anthropic extension boundary.
