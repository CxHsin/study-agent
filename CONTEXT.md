# Minimal Agent Core

This context describes a small, provider-independent Agent Core built to learn how model decisions, local tools, messages, and events form a controlled loop. The implementation is intentionally synchronous and in-memory before persistence, streaming, and recovery are added as separate learning stages.

## Language

**Agent Core**:
The focused execution component that runs one model/tool decision cycle and publishes its progress as structured Agent Events.
_Avoid_: Agent framework, Session Store, Runtime Recovery

**Agent Loop**:
The repeated decision cycle in which the model either requests a tool or produces a final response.
_Avoid_: Workflow, chain

**Internal Message Protocol**:
The provider-independent message vocabulary used by the Core to represent user messages, model responses, Tool Calls, and Tool Results.
_Avoid_: Provider payload, API response

**Tool Registry**:
The collection of Local Tool definitions available to an Agent Core, including each tool's name, description, input contract, and executor.
_Avoid_: Tool Executor, Tool Catalog

**Local Tool**:
A capability executed on the user's machine and exposed to the model through a defined input and output contract.
_Avoid_: Function, plugin

**Tool Call**:
A model request to execute one named Local Tool with a stable call ID and serialized arguments.
_Avoid_: Function call, API request

**Tool Result**:
A structured message reporting either the successful output or the failure of a requested Local Tool execution back to the model.
_Avoid_: Exception, log line

**Agent Event**:
One structured observation published by an Agent Core while processing a prompt, such as a model response, Tool Call, Tool Result, or Final Response.
_Avoid_: Message History, transcript

**Conversation Session**:
The in-memory sequence of model-facing messages shared by consecutive prompts until it is cleared.
_Avoid_: Run State, process state

**Message History**:
The ordered model-facing messages belonging to a Conversation Session, including user messages, assistant responses, Tool Calls, and Tool Results.
_Avoid_: Trace, event stream

**Final Response**:
The model response that completes the current prompt because it contains no request to execute a Local Tool.
_Avoid_: Tool Result, intermediate response

**Agent Workspace**:
The fixed filesystem boundary within which Local Tools are authorized to discover and read files.
_Avoid_: Working directory, repository root

**Provider Adapter**:
A boundary that translates an external model API into the Internal Message Protocol used by Agent Core.
_Avoid_: Model implementation, Tool Registry

**Provider Client**:
The provider-independent model-call boundary that applies request validation, capability fallback, response aggregation, retry policy, and time limits around a Provider Adapter.
_Avoid_: Provider Adapter, SDK client

**Model Profile**:
The declared request limits and supported protocol features of one configured model, including its context and output token ceilings.
_Avoid_: Provider capability, model name

**Model Request**:
A single provider-independent request containing typed model-facing messages, Local Tool definitions, and call-specific generation choices.
_Avoid_: Provider payload, Session

**Provider Stream Event**:
A typed, provider-independent increment produced during one model call, such as text, Tool Call data, usage, completion, or failure.
_Avoid_: Agent Event, SDK chunk

**Message Codec**:
The Provider-specific translator between the Internal Message Protocol and one external model API's request, response, and streaming representations.
_Avoid_: Context Builder, Provider Client

**RunResult**:
The explicit outcome of one Agent Core prompt, including its final response, stop reason, steps used, run ID, and an optional structured error.
_Avoid_: AgentResult, status-only result

**Stop Reason**:
The canonical reason a single Agent Core run ended: final, max_steps, repeated_tool_call, aborted, cancelled, or error.
_Avoid_: status, exit code

**Run Control**:
A cooperative control object that lets the caller abort or cancel a synchronous run at model and tool execution boundaries.
_Avoid_: thread manager, scheduler

**Trace**:
The ordered, immutable snapshot of Agent Events produced by one Agent Core run.
_Avoid_: Message History, log file

**Agent Event**:
A timestamped, sequenced observation with a run ID describing one lifecycle transition or execution result within a Trace.
_Avoid_: Conversation message, debug print

**Event Listener**:
An external callback that receives Agent Events during a run for real-time observation without owning the Trace.
_Avoid_: Event Store, Trace

**Tool Result**:
The provider-independent, structured outcome of one Tool Call, containing success data or a stable error classification.
_Avoid_: JSON error string, exception

**Tool Error**:
A typed failure raised by a Local Tool to describe a domain-level execution problem and whether it is retryable.
_Avoid_: Core Error, model error

**Tool Authorizer**:
The application policy boundary that decides whether a validated Tool Call is allowed to execute.
_Avoid_: Tool Choice, Prompt Instruction

**Confirmation Policy**:
The application callback that decides whether a Tool Call requiring human approval may execute.
_Avoid_: Model consent, authorization

**Context Window**:
The provider-defined maximum model-facing token capacity for one completion, including the selected input messages and the reserved output budget.
_Avoid_: Message limit, character limit

**Context Builder**:
The component that derives the model-facing context for a Run from the Session's complete Message History without changing that history by default.
_Avoid_: Provider Adapter, Message History

**Context Summary**:
A model-facing representation of an explicitly covered range of older, closed Message History, carrying enough metadata to identify its scope and version.
_Avoid_: Final Response, hidden prompt

**Summarizer**:
The independent model capability that produces a Context Summary from selected Message History; it is separate from the main Agent Loop decision.
_Avoid_: Agent Loop, automatic retry

**Compression Trigger**:
The configured utilization ratio at which the Context Builder requests Context Summary generation before the next main model call.
_Avoid_: Hard provider limit, truncation point

**Eval Case**:
A versioned description of one Agent Core Run input, controlled model/provider setup, and structured expectations used for regression evaluation.
_Avoid_: Production request, unit test only

**Eval Artifact**:
The redacted, versioned record of an Eval Case execution, including automatic results and independent human or judge assessments.
_Avoid_: Raw provider transcript, secret-bearing log

**Hard Rule**:
A deterministic evaluation assertion over RunResult or Trace data, such as Stop Reason, allowed tools, normalized arguments, order, count, authorization, or text constraints.
_Avoid_: LLM Judge, subjective score

**Trajectory Mode**:
The configured strict or structural comparison policy for Tool Call sequences; it ignores unstable identifiers and timestamps according to the selected policy.
_Avoid_: Trace equality, final-text score

**Inconclusive Evaluation**:
An Eval Case outcome where the evidence is insufficient to judge the task, such as an unavailable real Provider; it is distinct from a business failure.
_Avoid_: Passed, failed

**Eval Run**:
One bounded execution of a selected Eval Case collection with explicit provider, call, timeout, and budget limits.
_Avoid_: Agent Run, benchmark score

**Trajectory Comparison**:
The deterministic comparison of observed Tool Calls against a Case expectation under a selected Trajectory Mode.
_Avoid_: Text similarity, Judge Result

**Streaming Event**:
An incremental Agent Event emitted while a Run is progressing, including partial model output or a completed lifecycle transition.
_Avoid_: Log record, final response

**Steering Message**:
A user message accepted during an active Run and applied at the next model-call boundary to influence the remaining decision cycle.
_Avoid_: Follow-up message, tool cancellation

**Follow-up Message**:
A user message that starts a new Run while continuing the same Conversation Session after a prior Run has ended.
_Avoid_: Steering Message, retry command

**Provider Capability**:
An explicit feature of a configured Model Profile, such as streaming, Tool Calls, parallel Tool Call generation, usage reporting, or prompt-cache metadata.
_Avoid_: Provider-specific workaround, feature flag

**Provider Error**:
A provider-independent failure classification carrying retry and diagnostic facts from one external model call without exposing an SDK exception as the Runtime contract.
_Avoid_: Tool Error, raw SDK exception

**Retry Policy**:
The bounded rule used by a Provider Client to repeat a model attempt only before any model output has been exposed to its caller.
_Avoid_: SDK retry, Tool retry

**Recovery Checkpoint**:
A persisted safe boundary containing enough Session and Run records to resume after a process restart without assuming an unfinished external Tool Execution can be replayed.
_Avoid_: Snapshot-only backup, arbitrary event offset

**Prompt Cache Checkpoint**:
A stable message-prefix boundary used to measure or reuse prompt work, separately from any cache-hit information reported by a Provider.
_Avoid_: Context Summary, token estimate

**Run State**:
The persisted lifecycle position of a Run, including active waiting boundaries and terminal outcomes, used to decide whether it can continue after restart.
_Avoid_: Stop Reason, process status

**Tool Resolution**:
An explicit decision that supplies or confirms the outcome of a Tool Call whose execution was interrupted or remains unknown after recovery.
_Avoid_: Tool retry, exception handling

**Usage Record**:
Per-model-call accounting data for input, output, cached tokens, latency, and cost status, with estimated values distinguished from Provider-reported values.
_Avoid_: Token estimator, billing invoice

**Cache Hit Source**:
The origin of a prompt-cache hit classification: local checkpoint, Provider metadata, both, or estimated/unknown.
_Avoid_: Cache key, Context Summary

**Continuation Run**:
A new Run created to continue or compensate for an earlier Run after retry or Tool Resolution, linked by a parent Run identifier.
_Avoid_: Replayed Run, duplicate Run

**Capability Fallback**:
The configured behavior when a Provider lacks an optional capability, such as degrading streaming, marking usage unknown, or relying on local cache metrics.
_Avoid_: Silent emulation, Provider error

**Checkpoint Invalidation**:
The rule that prevents reuse of a local Prompt Cache Checkpoint when model-facing inputs or the context-building contract have changed.
_Avoid_: Session clear, cache eviction

**Read File Tool**:
A fixed Local Tool that returns UTF-8 text from an authorized Agent Workspace path, optionally bounded by inclusive one-based line numbers.
_Avoid_: File browser, arbitrary file access

**Bash Tool**:
A fixed Local Tool that runs one shell command in the Agent Workspace and reports structured process output; every call requires explicit confirmation.
_Avoid_: Shell access, unrestricted command runner

**Tool Confirmation**:
An explicit approval decision required before a governed Tool Call executes; absence of a confirmation policy is a denial.
_Avoid_: Model consent, authorization

**Execution Boundary**:
The set of workspace, path, environment, timeout, output, and cancellation rules governing a Local Tool execution.
_Avoid_: Full operating-system sandbox
