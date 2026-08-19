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
