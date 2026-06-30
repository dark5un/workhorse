# Workhorse

Workhorse is a Rust-based interactive CLI harness that routes prompts to different LLMs based on computed complexity. It supports any OpenAI-compatible provider — local runtimes like LM Studio and Ollama, or cloud APIs like OpenAI, Anthropic, and OpenRouter — and persists session state in SQLite.

Short prompts go to fast, cheap models. Complex prompts go to larger, more capable ones. You always have the final say with `/model`.

## Why use Workhorse

- **Cost-aware routing** — trivial prompts don't waste expensive model calls
- **Local-first** — works with LM Studio and Ollama out of the box, no API keys needed
- **Config-driven** — every threshold, model ID, and fallback chain lives in YAML, not code
- **Persistent sessions** — conversation history survives restarts via SQLite
- **MCP tools** — tool execution through the Model Context Protocol (Phase 4)
- **Cost tracking** — per-session spend accumulator with configurable limits

---

## Quick Start: Local LLMs (LM Studio)

This is the fastest way to get running. No API keys, no cloud, no bills.

### 1. Install LM Studio and load a model

Download [LM Studio](https://lmstudio.ai/), download a model (e.g. `zai-org/glm-4.7-flash`, `qwen/qwen3-coder-next`), and start the local server:

- Open LM Studio
- Go to the **Developer** tab
- Click **Start Server** (default port: 1234)
- Make sure at least one model is loaded

### 2. Build Workhorse

```sh
git clone <repo-url> workhorse
cd workhorse
cargo build --release
```

### 3. Run it

```sh
./target/release/workhorse
```

That's it. Workhorse automatically detects models from your LM Studio server and routes prompts based on complexity. Short prompts like "hello" go to fast models; longer, complex prompts go to larger models.

### How routing works

Workhorse analyzes each prompt using a heuristic engine (token count, keywords, structure) and assigns a complexity tier:

| Tier | Token range | Example models |
|------|-------------|----------------|
| simple | 0–50 | glm-4.7-flash, gpt-4o-mini |
| medium | 51–200 | gemma-4-26b, gpt-4o |
| complex | 201–4096 | qwen3.6-35b, claude-opus |
| expert | 4097+ | qwen3-coder-next, llama-3.3-70b |

The first model in each tier's list is the primary; the rest form a fallback chain. All of this is configured in `config/routing.yaml` — change the lists, change the routing.

You can always override routing with `/model`:

```
> /model lm-studio/qwen/qwen3-coder-next
Model override set to: lm-studio/qwen/qwen3-coder-next
```

### Verify it works

```sh
curl http://localhost:1234/v1/models
```

This should return a list of loaded models. Workhorse uses the same endpoint to discover what's available.

---

## Quick Start: Ollama

[Ollama](https://ollama.ai/) is another popular local runtime. It exposes an OpenAI-compatible API at `http://localhost:11434/v1`.

1. Start Ollama and pull a model:

   ```sh
   ollama serve
   ollama pull llama3.3
   ```

2. Uncomment the `ollama` provider in `config/providers.yaml`:

   ```yaml
   ollama:
     base_url: "http://localhost:11434/v1"
     pricing: {}
   ```

3. Add model entries to `config/routing.yaml` using `ollama/` as the provider prefix:

   ```yaml
   simple:
     models:
       - "ollama/llama3.3"
   ```

4. Run Workhorse:

   ```sh
   ./target/release/workhorse
   ```

---

## Quick Start: Cloud Providers

Cloud providers require API keys set as environment variables.

### OpenAI

```sh
export OPENAI_API_KEY="sk-..."
./target/release/workhorse
```

### Anthropic

```sh
export ANTHROPIC_API_KEY="sk-ant-..."
./target/release/workhorse
```

### OpenRouter (access many models through one API)

```sh
export OPENROUTER_API_KEY="sk-or-..."
./target/release/workhorse
```

### Mixing local and cloud

Workhorse supports all providers simultaneously. Put local models first in each tier for free inference, with cloud models as fallbacks. Example from the default `config/routing.yaml`:

```yaml
simple:
  models:
    - "lm-studio/zai-org/glm-4.7-flash"   # local, free
    - "openai/gpt-4o-mini"                 # cloud, cheap
    - "anthropic/claude-haiku"             # cloud, cheap
```

If LM Studio is running, the prompt goes there. If not, Workhorse falls through to the next model in the chain.

---

## Model ID Format

Workhorse uses `provider/model` as the canonical model identifier. The provider prefix must match a key in `config/providers.yaml`.

For model IDs that themselves contain slashes (common with LM Studio and OpenRouter), the first `/` is the provider boundary and everything after is the model name:

| Model ID string | Provider | Model name (sent to API) |
|-----------------|----------|--------------------------|
| `lm-studio/zai-org/glm-4.7-flash` | `lm-studio` | `zai-org/glm-4.7-flash` |
| `openrouter/anthropic/claude-3.5-sonnet` | `openrouter` | `anthropic/claude-3.5-sonnet` |
| `openai/gpt-4o` | `openai` | `gpt-4o` |

---

## Configuration

All config lives in the `config/` directory. Workhorse loads it at startup using figment (layered: YAML files < environment variables).

### Files

| File | Purpose |
|------|---------|
| `config/providers.yaml` | Provider endpoints, API key env vars, pricing tables |
| `config/routing.yaml` | Complexity tiers, keywords, model lists, fallback chains |
| `config/session.yaml` | Session storage, context window, cost tracking, ranking |
| `config/tools.yaml` | MCP server definitions and sandbox settings |
| `config/system_prompt.md` | System prompt loaded at session start |

### Environment variable overrides

Any config field can be overridden with `HARNESS_` prefixed env vars, using `__` as the nesting separator:

```sh
# Change session storage path
HARNESS_SESSION__PATH=/tmp/my-sessions.db ./target/release/workhorse

# Disable cost tracking
HARNESS_SESSION__COST_TRACKING__ENABLED=false ./target/release/workhorse
```

---

## REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/model <provider/model>` | Override routing — force a specific model |
| `/model` | Show current model (or "auto") |
| `/clear` | Clear session history and reset cost |
| `/cost` | Show session spend and budget |
| `/budget <tokens>` | Set context token budget |
| `/tools` | List registered MCP tools |
| `/rate [1-5]` | Rate the last response (for ranking) |
| `/rate <model> [1-5]` | Rate a specific model |
| `/ratings [tier]` | Show model rankings for a tier |
| `/reset-ratings [global]` | Clear ratings (current scope or global) |
| `/ranking on\|off\|status` | Enable/disable/query the ranking engine |
| `/quit` | Exit |

---

## How It Works

### Per-prompt flow

1. **Heuristic analysis** — tokenizes the prompt (tiktoken `cl100k_base`), checks length against tier thresholds, matches keywords, detects structure (code blocks, JSON, multi-step instructions). Produces a `ComplexityResult` with tier, confidence, and signals.
2. **Optional classifier** — if enabled in config, a cheap LLM call classifies the prompt with structured output. Can override the heuristic. Falls back to heuristic on failure.
3. **Routing** — looks up the tier's model list from config. First model is primary; the rest are the fallback chain.
4. **Adapter call** — the OpenAI-compatible adapter sends the request to the selected provider. If the provider is unreachable, the session loop falls through the chain.
5. **Response** — text chunks and tool calls are surfaced as session events. Token usage and cost are recorded.
6. **Context management** — sliding window eviction keeps the context within the token budget. The system prompt is never evicted.

### Cost tracking

Pricing is defined per-model in `config/providers.yaml` (input/output per 1M tokens, in USD cents). After each call, Workhorse computes the cost and accumulates it. Local providers have empty pricing tables, so cost is always $0.00.

```sh
> /cost
Session cost: $0.03 (3 cents)
Budget limit: $20.00
```

---

## Development

### Requirements

- Rust stable toolchain (edition 2024)
- Cargo
- SQLite (bundled via `rusqlite` — no system install needed)

### Build and run

```sh
cargo build --release
./target/release/workhorse
```

Or use the cargo alias:

```sh
cargo dev    # runs: cargo run --release --bin workhorse
```

### Tests

```sh
cargo test --all
```

Tests use a mock adapter (no network calls). Session tests route to `mock/test` to stay offline.

### Lint and format

```sh
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

### Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `providers` | enabled | HTTP adapter for real LLM providers (reqwest + rustls) |
| `repl` | disabled | Full line-editor stack (clap + reedline) |
| `otel` | disabled | OpenTelemetry export hooks |
| `wasmtime-sandbox` | disabled | Wasmtime sandbox for MCP tools |
| `docker-sandbox` | disabled | Docker sandbox for MCP tools |

Build without providers (offline/mock-only):

```sh
cargo build --no-default-features
```

### Project layout

```
src/
  cli/            REPL loop, input parsing, output
  core/           Analyzer, Router, Session, Ranking
  adapters/       LLMAdapter trait, OpenAI-compatible adapter, mock
  tools/          MCP client, tool registry, consent sandbox
  config/         Figment-based config loading and validation
  observability/  tracing setup
config/
  providers.yaml  Provider endpoints and pricing
  routing.yaml    Complexity tiers and model lists
  session.yaml    Session storage, context window, cost tracking
  tools.yaml      MCP server definitions
  system_prompt.md
```

### Packaging

```sh
cargo install --path .
# or
docker build -t workhorse .
```

---

## License

See [LICENSE](LICENSE).
