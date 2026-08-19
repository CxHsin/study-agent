---
status: accepted
---

# Type Provider Messages and Centralize Model Calls

Provider integrations use immutable, provider-independent message and tool records, with
versioned serialization at persistence boundaries. Provider-specific Message Codecs own
wire translation and emit one typed event stream; a shared Provider Client aggregates
that stream for synchronous callers and owns capability validation, deadlines, fallback,
and observable retries. Model limits belong to the configured Model Profile, while
adapters classify external failures without deciding retry policy. This prevents one
provider's payload shape or SDK behavior from becoming the Runtime protocol and keeps
streaming and non-streaming response parsing on one path.

**Considered Options**

- Keep dictionary messages and duplicate synchronous/streaming parsing in every adapter.
- Treat an OpenAI-compatible payload as the common internal representation.
- Use typed internal records, Provider-specific codecs, and a shared call policy (chosen).

**Consequences**

- Old session records are normalized and validated when loaded, then written in the new format.
- Provider SDK retries are disabled so attempts, deadlines, and retry delays remain observable.
- A response that has exposed any model delta cannot be retried implicitly.
- Provider parallel Tool Call generation does not imply parallel Local Tool execution.
