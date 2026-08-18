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
