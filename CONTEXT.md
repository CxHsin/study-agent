# Minimal Agent Harness

This context describes the smallest agent runtime used to learn how model decisions, local tool execution, and task completion form a controlled loop.

## Language

**Agent Runtime**:
The complete execution environment that governs agent state, model interaction, tool use, context, permissions, observability, and run control. It contains the Agent Harness and the Agent Loop rather than naming either one alone.
_Avoid_: Agent framework, Agent Harness, Agent Loop

**Agent Harness**:
The runtime boundary that coordinates model decisions, tool execution, and completion for one agent interaction.
_Avoid_: Agent framework, chatbot

**Agent Loop**:
The repeated decision cycle in which the model either requests a tool or produces a final response.
_Avoid_: Workflow, chain

**Run State**:
The current lifecycle state of one submitted task, including whether its model decision, tool work, or completion is pending, active, recoverable, or terminal.
_Avoid_: Session state, process state

**Internal Message Protocol**:
The provider-independent message vocabulary and ordering rules used by the Agent Runtime to represent model decisions, Tool Calls, Tool Results, and Final Responses.
_Avoid_: Provider message format, API payload

**Tool Execution Status**:
The recorded outcome of a Tool Call execution, including whether it is pending, completed, failed, or requires resolution before it may be retried.
_Avoid_: Tool result, function status

**Recoverable Run**:
A Run that stopped before terminal completion but retains enough validated state to continue without guessing what work already happened.
_Avoid_: Retryable task, resumable session

**Idempotent Tool**:
A Local Tool whose repeated execution with the same call identity and arguments has no additional externally visible effect.
_Avoid_: Safe tool, read-only tool

**Retry Policy**:
The explicit rule that determines whether a failed or interrupted Tool Call may be attempted again and under what conditions.
_Avoid_: Error handling, fallback

**Call Identity**:
The stable identity of one requested Tool Call across persistence, execution, recovery, and Trace inspection.
_Avoid_: Request ID, parameter fingerprint

**State Transition Table**:
The explicit mapping from a Run State and an observed event to the next Run State and its permitted actions.
_Avoid_: Control-flow diagram, workflow definition

**Local Tool**:
A capability executed on the user's machine and exposed to the model through a defined input and output contract.
_Avoid_: Function, plugin

**Interactive Console Agent**:
An agent interface that accepts tasks and displays results through a continuing command-line session.
_Avoid_: CUI, TUI, chatbot

**Agent Workspace**:
The fixed filesystem boundary within which local tools are authorized to discover and read files.
_Avoid_: Working directory, repository

**Tool Result**:
A structured message reporting either the successful output or the failure of a requested local tool execution back to the model.
_Avoid_: Observation, exception

**Final Response**:
The model response that completes the current task because it contains no request to execute a local tool.
_Avoid_: Chat message, completion

**Conversation Session**:
The continuing interaction context identified by a Session ID and shared by consecutive tasks until the user starts a new session. The first console startup creates the default Session; `/reset` starts another one while retaining the previous Session.
_Avoid_: Chat, thread, Run

**Message History**:
The ordered model-facing messages belonging to a Conversation Session, including user messages, assistant responses, Tool Calls, and Tool Results needed to continue that session.
_Avoid_: Transcript, Trace

**Session ID**:
The stable identity used to distinguish one Conversation Session's Message History from another.
_Avoid_: Run ID, request ID

**Session Store**:
Durable storage that owns Message History for Conversation Sessions and can restore a session after the interactive process exits.
_Avoid_: Trace Store, cache

**Agent Step**:
One model decision within the Agent Loop, containing either an ordered set of local tool requests or a Final Response.
_Avoid_: Iteration, turn, API call

**Trace**:
Structured runtime evidence for one task, recording Agent Steps, model calls, Tool Calls, Tool Results, durations, and the terminal stop reason without recording chain-of-thought or full file contents.
_Avoid_: Log, transcript

**Agent Event**:
One structured Trace record emitted while an Agent Harness processes a task.
_Avoid_: Message, log line

**Tool Event**:
An Agent Event describing one Local Tool request or its Tool Result, including safe arguments, status, duration, and internal correlation.
_Avoid_: Function event, observation

**Run**:
One execution of a submitted user task from its first Agent Event through its terminal Agent Event. Every Run has an identity independent of the continuing Conversation Session.
_Avoid_: Conversation, process

**Trace Store**:
Durable storage of Agent Events grouped and ordered by Run so completed behavior can be inspected after execution.
_Avoid_: Message History, Session Store
