# Minimal Agent Harness

A small interactive console Agent built to make the model/tool loop explicit. It uses DeepSeek for model decisions and exposes only read-only tools inside the local `workspace/` directory.

## Setup

```powershell
uv sync --dev
Copy-Item .env.example .env
```

Set your local key in `.env`:

```dotenv
DEEPSEEK_API_KEY=your-key-here
```

The real `.env` file is ignored by Git.

## Run

Put UTF-8 text files under `workspace/`, then start the console:

```powershell
uv run minimal-agent
```

The console reuses the most recent persisted Conversation Session after restart. Use `/reset`
to create a new Session without deleting the old one, `/trace last` to inspect the most recent
persisted Run, and `/exit` to quit.

Trace events are appended to `.minimal-agent/traces.jsonl`; Message History is stored in
`.minimal-agent/sessions.sqlite3`. The directory is ignored by Git. Trace records exclude
chain-of-thought, full messages, and file contents, while Message History currently stores the
model-facing messages needed to restore a Session.

## Current Flow

```text
user task
-> DeepSeek model decision
-> ordered local Tool Calls, when requested
-> structured Tool Results
-> another model decision
-> Final Response or terminal guard
```

The Adapter uses `deepseek-v4-flash` through the OpenAI-compatible Chat Completions interface and explicitly disables thinking mode.

## Verify

```powershell
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## References

- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)
- [DeepSeek Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode/)
- [OpenAI Function Calling](https://developers.openai.com/api/docs/guides/function-calling/)
- [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/)
