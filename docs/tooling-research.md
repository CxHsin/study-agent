# Official Tool-Use Research

This note records first-party guidance reviewed for Phase 3. It is research input, not an implementation specification.

## OpenAI

Source: [Function calling](https://developers.openai.com/api/docs/guides/function-calling)

- Tool use is an application-controlled loop: send tools, receive one or more calls, execute them in application code, return tool outputs, and ask the model to continue.
- Function tools use JSON Schema. OpenAI recommends `strict: true`; strict schemas require `additionalProperties: false` and every property to be required. Optional values can be represented with a nullable type.
- The application must handle zero, one, or multiple calls. `parallel_tool_calls: false` can constrain a turn to at most one call.
- Tool definitions should have clear names, detailed descriptions, parameter explanations, and edge cases. Keep the initial tool set small; OpenAI suggests fewer than 20 as a soft guideline and offers deferred loading/tool search for larger surfaces.
- Tool choice can be `auto`, `required`, `none`, a forced function, or an allowed subset. This is a model-side selection constraint, not an authorization system.
- Tool outputs are application-generated strings (or structured media/file outputs), so the application owns output formatting and error representation.

Source: [Safety in building agents](https://platform.openai.com/docs/guides/agent-builder-safety)

- Prompt injection is untrusted text attempting to redirect model behavior toward data leakage or unintended actions.
- Do not put untrusted variables into developer messages. Use structured outputs to constrain data flow and isolate untrusted data.
- Keep tool approvals enabled for sensitive operations; human approval is a control boundary, not a prompt convention.
- Guardrails and trace/eval systems reduce risk but do not make agents perfect. Risk increases when arbitrary text can influence tool calls.

## Anthropic

Source: [Define tools](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use)

- Client tools have a name, detailed description, JSON Schema `input_schema`, and optional schema-valid `input_examples`.
- Descriptions should explain what the tool does, when to use it, when not to use it, parameter semantics, and caveats. Anthropic recommends consolidating related operations and returning only high-signal stable data.
- Tool names must match `^[a-zA-Z0-9_-]{1,64}$`.
- `tool_choice` supports `auto`, `any`, a forced tool, and `none`; strict tool use can combine with forced selection to guarantee schema-conforming input.

Source: [Handle tool calls](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/handle-tool-calls)

- A client tool use contains an ID, name, and structured input. The application replies with a `tool_result` carrying the matching `tool_use_id`.
- Tool results must immediately follow the corresponding assistant tool-use blocks. Multiple results belong first in the user message content before any text.
- Execution failures are represented with `is_error: true`, allowing the model to recover instead of treating every tool failure as a transport failure.
- Tool results may contain external text and must be treated as untrusted; keep that content inside tool-result boundaries rather than promoting it to system/developer instructions.

## Implications For This Project

- Adopt strict, provider-independent input validation at the `ToolRegistry` boundary, while retaining domain checks inside tools.
- Keep model selection controls (`tool_choice`-like policy) separate from local authorization and confirmation.
- Preserve one or many calls per model step, but make parallel execution an explicit later decision because this Core is synchronous and in-memory.
- Keep structured, stable error results visible to the model; do not turn ordinary tool failures into Core errors.
- Treat all tool output as untrusted data and preserve its boundary in the Internal Message Protocol.
- Add human confirmation and workspace authorization as application-side policy hooks, not prompt text.
- Do not copy provider-specific message shapes into the provider-independent protocol until a concrete adapter requires them.
