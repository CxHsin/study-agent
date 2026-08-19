# Minimal Agent Harness

A small interactive console Agent built to make the model/tool loop explicit. The console
uses DeepSeek by default, while the runtime also includes OpenAI and optional Anthropic
Provider Adapters behind one typed model protocol.

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

Install the optional Anthropic SDK when using that adapter:

```powershell
uv sync --extra anthropic --dev
```

## Run

Put UTF-8 text files under `workspace/`, then start the console:

```powershell
uv run minimal-agent
```

The console keeps the current Conversation Session in memory. Use `/reset` to clear it,
`/trace last` to inspect the current run's Agent Events, and `/exit` to quit.

## Current Flow

```text
user task
-> DeepSeek model decision
-> ordered local Tool Calls, when requested
-> structured Tool Results
-> another model decision
-> Final Response or terminal guard
```

The default Adapter uses `deepseek-v4-flash` through the OpenAI-compatible Chat
Completions interface and explicitly disables thinking mode. Provider-specific Message
Codecs translate immutable internal messages, Tool Calls, Tool Results, summaries, and
stream fragments. A shared Provider Client validates the configured Model Profile,
aggregates synchronous responses, applies bounded retries before output is exposed, and
classifies authentication, rate-limit, timeout, network, request, server, and response
failures. Sessions and recovery checkpoints are persisted locally in SQLite.

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
