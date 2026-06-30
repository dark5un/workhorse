# AGENTS.md

## 1. Project Overview
Build a **Rust-based interactive CLI harness** that:
- Routes prompts to different LLMs dynamically based on computed complexity
- Supports an interactive, persistent session (like Claude Code)
- Executes tools via the **Model Context Protocol (MCP)** -- the de facto standard tool protocol, not a bespoke one
- Operates entirely from external configuration (zero hardcoded config values)
- Is structured for safe, incremental agent-driven development

All logic must be trait-first, config-driven, and explicitly decoupled. Tools manage their own outputs. The harness only orchestrates routing, session state, and execution flow.

---

## 2. Architecture & Module Layout

**Single crate** with module separation (not a Cargo workspace). Adapters and tools live as modules within the crate; the workspace pattern adds boilerplate that is not justified at this scope. If adapters/tools are later published independently, they can be extracted into separate crates with minimal refactoring.

```
src/
├── lib.rs          # Public re-exports, crate root
├── cli/            # REPL loop, input parsing, streaming output
├── core/           # Router, Analyzer, Session, State management
├── adapters/       # LLM trait + provider implementations
├── tools/          # MCP client, tool registry, sandbox, consent
├── config/         # Schema, loaders (figment), validators, env overrides
└── observability/  # tracing-based structured logging, metrics, OTel hooks
```

**Module Boundaries:**
- No cross-module circular dependencies
- Traits defined in parent modules, implementations in children
- Config parsed once at startup, passed immutably downstream
- Tools are isolated modules; harness never reads/writes tool-generated files

---

## 3. Core Interfaces (Rust Traits)

### 3.0 Async Trait Compatibility Note

This crate targets Rust edition 2024. `async fn` in traits is stable, but traits using `async fn` are **not automatically dyn-compatible** (cannot be made into `Box<dyn Trait>`). Since the tool registry and adapter registry use dynamic dispatch, all async traits in this design use the `#[async_trait]` macro from the `async-trait` crate. This is a deliberate, documented choice; if dyn-compatibility lands natively in a future edition, the macro can be removed with a mechanical refactor.

### 3.1 Prompt Complexity Analyzer

The analyzer has **two stages**, configurable per deployment:

1. **Heuristic stage** (always available, no LLM call): length + keyword + structural rules from `config/routing.yaml`. Fast, free, approximate.
2. **Classifier stage** (optional, enabled via config): routes the prompt through a cheap model (e.g. `gpt-4o-mini`, `claude-haiku`) that returns a structured `ComplexityTier` + confidence. More accurate; costs one cheap call per prompt.

If the classifier stage is enabled, it runs after the heuristic stage and can override the heuristic result. If the classifier fails (network, parse error, timeout), the harness falls back to the heuristic result and logs the degradation. The user can always override with `/model <id>`.

```rust
#[async_trait]
pub trait PromptAnalyzer: Send + Sync {
    async fn analyze(&self, prompt: &str) -> Result<ComplexityResult, AnalysisError>;
}

pub struct ComplexityResult {
    pub tier: ComplexityTier,
    pub confidence: f32,
    pub signals: Vec<String>,
    pub source: AnalysisSource,  // Heuristic | Classifier | Override
}

pub enum ComplexityTier {
    Simple,
    Medium,
    Complex,
    Expert,
}

pub enum AnalysisSource {
    Heuristic,
    Classifier { model: String },
    FallbackHeuristic { reason: String },
}
```

*Rule engines must read thresholds, keywords, and structural rules from `config/routing.yaml`. No compiled-in config values. The matching ALGORITHM (case-insensitive, prefix matching, etc.) is compiled logic, not config -- see §9.*

### 3.2 Router Engine
```rust
#[async_trait]
pub trait Router: Send + Sync {
    async fn route(
        &self,
        complexity: &ComplexityResult,
        user_override: Option<&ModelId>,
    ) -> Result<ModelSpec, RoutingError>;
}

pub struct ModelSpec {
    pub model_id: ModelId,
    pub fallback_chain: Vec<ModelId>,
    pub budget_limit: Option<Cost>,  // in USD cents, not bare u64
}

/// Canonical model identifier: provider + model name.
/// Parsed and validated at config load time. No bare strings in routing.
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

/// Monetary cost in USD cents. Newtype prevents mixing with raw token counts.
pub struct Cost(u64);
```
*Fallback chains and tier mappings are loaded from config. Router does NOT pre-validate provider availability (that adds latency per routing decision). Instead, it returns the preferred model + fallback chain; the adapter attempts the primary, and on failure falls through the chain. See §7 for retry/backoff.*

### 3.3 LLM Adapter Abstraction
```rust
#[async_trait]
pub trait LLMAdapter: Send + Sync {
    /// Stream a completion. Uses `&self` -- adapters manage mutable state
    /// (HTTP client pools, etc.) via interior mutability (Arc<Mutex<...>>).
    async fn send(
        &self,
        messages: Vec<Message>,
        config: ModelConfig,
    ) -> Result<AdapterStream, LLMError>;

    fn capabilities(&self) -> ModelCapabilities;
}

/// Wrapper around Pin<Box<dyn Stream + Send>> for dyn-compatibility.
/// The stream yields ResponseEvent items (text deltas, tool calls, completion).
pub struct AdapterStream(/* Pin<Box<dyn Stream<Item = Result<ResponseEvent, LLMError>> + Send>> */);

pub struct ModelConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
    pub tools: Option<Vec<ToolSpec>>,
    pub response_format: Option<ResponseFormat>,  // JSON mode / structured output
}

pub struct ModelCapabilities {
    pub streaming: bool,
    pub tool_calling: bool,
    pub structured_output: bool,
    pub vision: bool,
    pub max_context_tokens: usize,
}
```

**Tool-call round-trip (critical, was underspecified):**

Providers use different tool-call schemas (OpenAI: function calling; Anthropic: `tool_use` blocks). The adapter trait must normalize these:

```rust
/// Normalized provider response. Adapters translate provider-specific
/// tool-call formats into this common shape.
pub enum ResponseEvent {
    Chunk(String),                    // text delta
    ToolCall(ToolInvocation),         // normalized tool call request
    Done(Usage),                      // completion with token usage
}

pub struct ToolInvocation {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost: Cost,  // computed from config pricing table
}
```

The session loop feeds `ToolResult` back as a `Message` of role `Tool`, which the adapter serializes in the provider's expected format.

*Providers implement this trait. The harness never calls provider-specific SDKs directly.*

### 3.4 Tool System (MCP-based)

The harness implements an **MCP client**. Tools are MCP servers (local subprocesses or remote). The internal `Tool` trait is a thin adapter over the MCP protocol -- it exists so the registry can manage lifecycles, but the wire format and schema are MCP standard.

This means:
- Any existing MCP server (filesystem, git, shell, etc.) works out of the box.
- Custom tools written for this harness are portable to any MCP-compatible client.
- Tool schemas use the MCP `tools/call` and `tools/list` JSON-RPC methods.

```rust
/// Internal adapter trait wrapping an MCP server connection.
/// The harness owns the MCP client transport; this trait exposes
/// a uniform interface to the registry and session loop.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> serde_json::Value;  // MCP tool schema

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError>;

    /// Cleanup is called when the tool's session ends.
    /// Tools should ALSO implement Drop for panic-safe cleanup.
    /// The harness calls this for graceful shutdown; Drop is the safety net.
    async fn cleanup(&self) -> Result<(), ToolError>;
}

pub struct ToolResult {
    pub content: Vec<ToolContent>,  // text, image, embedded resource (MCP types)
    pub is_error: bool,
}

pub enum ToolContent {
    Text(String),
    Image { mime_type: String, data: Vec<u8> },
    Resource { uri: String, mime_type: String },
}

pub struct ToolRegistry {
    // registers MCP servers, validates tool schemas, resolves tool deps
    // wraps each tool in a RAII guard that ensures cleanup on drop
}
```

**Tool lifecycle:**
1. Registry starts the MCP server (subprocess or connects to endpoint).
2. Registry calls `tools/list` to discover available tools and their schemas.
3. On invocation, session loop calls `tools/call` via the adapter trait.
4. Each tool execution runs in a **per-session temp directory** (`/tmp/harness-<session-id>/`) to isolate concurrent file access.
5. On session end or tool unregister: registry calls `cleanup()`, then the RAII guard's `Drop` kills the subprocess / closes the connection.

*Tools are self-contained. They create, modify, or delete their own files within their sandbox. The harness only receives execution requests and routes them.*

### 3.5 Session & REPL Controller
```rust
#[async_trait]
pub trait SessionController: Send + Sync {
    async fn process(&mut self, input: &str) -> Result<SessionStream, SessionError>;
    async fn reset(&mut self);
    fn status(&self) -> SessionState;
}
```

---

## 4. Configuration Schema
All defaults, thresholds, fallbacks, and permissions are externalized.

**Config loading uses `figment`** (layered config: YAML files < env vars < CLI flags). This replaces hand-rolled `env::var()` fallbacks and gives strongly-typed serde structs with compile-time field checking. Validation is done via serde deserialization + custom validators, not runtime `serde_json::Value` schema walking.

### `config/routing.yaml`
```yaml
analyzer:
  heuristic:
    enabled: true
    tiers:
      simple:
        thresholds: { min_tokens: 0, max_tokens: 50 }
        keywords: ["hello", "translate", "format"]
        models: ["openai/gpt-4o-mini", "anthropic/claude-haiku"]
      medium:
        thresholds: { min_tokens: 51, max_tokens: 200 }
        keywords: ["analyze", "summarize", "refactor"]
        models: ["openai/gpt-4o", "anthropic/claude-sonnet"]
      complex:
        thresholds: { min_tokens: 201, max_tokens: 4096 }
        keywords: ["debug", "architect", "optimize"]
        models: ["anthropic/claude-opus", "openai/gpt-4-turbo"]
      expert:
        thresholds: { min_tokens: 4097, max_tokens: null }
        keywords: ["reason", "plan", "simulate"]
        models: ["custom/70b", "openrouter/mixtral"]
    fallback_policy: "sequential"
    timeout_seconds: 30
  classifier:
    enabled: true
    model: "openai/gpt-4o-mini"
    fallback_on_error: true  # fall back to heuristic if classifier fails
    timeout_seconds: 10
```

### `config/tools.yaml`
```yaml
tools:
  mcp_servers:
    - name: "filesystem"
      transport: "subprocess"
      command: "mcp-server-filesystem"
      args: ["--allow", "/home/user/projects"]
      sandbox: "consent"  # consent | wasmtime | docker | none
    - name: "shell"
      transport: "subprocess"
      command: "mcp-server-shell"
      sandbox: "consent"
  defaults:
    sandbox: "consent"
    session_temp_dir: "/tmp/harness-${SESSION_ID}"
```

### `config/providers.yaml`
```yaml
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    pricing:  # per 1M tokens, in USD cents
      gpt-4o: { input: 250, output: 500 }
      gpt-4o-mini: { input: 15, output: 30 }
  anthropic:
    base_url: "https://api.anthropic.com"
    api_key_env: "ANTHROPIC_API_KEY"
    pricing:
      claude-sonnet: { input: 300, output: 1500 }
      claude-haiku: { input: 25, output: 125 }
```

### `config/session.yaml`
```yaml
session:
  storage: "sqlite"  # sqlite | json (sqlite recommended)
  path: "~/.harness/sessions.db"
  context_window:
    strategy: "sliding_window"  # sliding_window | summarize | sticky
    max_tokens: 128000
    sticky_system_prompt: true
    summarize:
      model: "openai/gpt-4o-mini"
      trigger_at_pct: 80  # summarize when context hits 80% of max
  system_prompt_file: "config/system_prompt.md"
  cost_tracking:
    enabled: true
    warn_at_usd: 5.00
    hard_limit_usd: 20.00
```

### Validation Rules
- Config loaded via `figment::Figment` with strongly-typed serde structs.
- Missing required fields → `figment` error with the exact path (e.g. `routing.tiers.simple.models`).
- Environment variables override config fields via figment's env provider (`HARNESS_ROUTING__TIMEOUT_SECONDS=60`).
- CLI flags (`--model`, `--budget`) override everything else.

---

## 5. Routing & Complexity Analysis Logic

### Heuristic Stage
1. Parse prompt → tokenize using **tiktoken-rs** (`cl100k_base` encoding as a reference tokenizer). This runs before the destination model is known, so a reference tokenizer is used. The result is approximate; documented as such.
2. Apply rule engine from `routing.yaml`:
   - Length check against tier thresholds (token count, not byte/word count)
   - Keyword matching (case-insensitive, exact or prefix -- algorithm is compiled, keywords are config)
   - Structural detection (JSON/YAML, code blocks, multi-step instructions)
3. Compute `confidence` (0.0–1.0) based on signal overlap.

### Classifier Stage (if enabled)
4. Send prompt to the configured classifier model with a structured-output request asking it to return `{ "tier": "...", "confidence": 0.0-1.0, "reasoning": "..." }`.
5. If classifier succeeds, use its result. If it fails, fall back to heuristic result and log the degradation.

### Routing
6. Router selects first model from the tier's model chain (from config).
7. If user passes `--model` or `/model <id>`, bypass routing entirely.
8. Log decision: `{tier, confidence, source, selected_model, fallback_count}`.

*All thresholds, keywords, and model IDs come from config. The analyzer code contains no literal config values. The matching algorithm (how keywords are matched) is compiled logic, not config -- see §9.*

---

## 6. Interactive REPL & Session Management
- Async event loop (`tokio::spawn` + `select!`)
- **Line editor**: `reedline` (Rust-native, Rustyline alternative with multiline, syntax highlighting, history).
- **CLI framework**: `clap` with derive macros (NOT `typer` -- that is a Python library).
- Persistent context window with token budget tracking
- **Context window management** (configurable via `session.yaml`):
  - `sliding_window`: drop oldest messages when over budget, keep system prompt + last N turns.
  - `summarize`: when context hits threshold (default 80%), call a cheap model to compress old turns into a summary message.
  - `sticky`: keep system prompt + user-selected sticky messages, drop everything else.
- Streaming output via async stream (`AdapterStream`)
- Slash commands: `/help`, `/tools`, `/model`, `/clear`, `/budget <tokens>`, `/cost`, `/quit`
- Error recovery: network retries (exponential backoff + jitter via `backoff` crate), config validation, graceful fallback
- **Session state**: stored in SQLite (`rusqlite` crate) via `~/.harness/sessions.db`. Append-only message log, resumable sessions, crash-safe. JSON storage available as a fallback option but not recommended for multi-session use.
- **System prompt**: loaded from `config/system_prompt.md` at session start, treated as a sticky message (never dropped by context management).
- **Cost tracking**: per-session accumulator using the pricing table from `config/providers.yaml`. `/cost` command shows current session spend. Configurable warn + hard limits.

---

## 7. Tool/Plugin System (MCP)

### Sandbox Models

The harness supports multiple sandbox levels, configured per MCP server in `tools.yaml`:

| Sandbox | Description | Platform | Overhead |
|---------|-------------|----------|----------|
| `consent` | Default. Asks the user before destructive operations (file write, shell exec). Like Claude Code's permission model. | All (macOS, Linux) | None |
| `wasmtime` | Runs the MCP server inside a Wasm runtime. True isolation: filesystem and network are virtualized. Tools must be Wasm-compatible. | All | Medium |
| `docker` | Runs the MCP server inside a Docker/Podman container with volume mounts. | Linux, macOS (Docker Desktop) | High |
| `none` | No sandbox. For trusted local tools only. Explicit opt-in required. | All | None |

`consent` is the default and works on macOS today. `wasmtime` and `docker` are optional isolation layers for untrusted tools.

### Execution Loop
1. Parse tool call from LLM response (normalized by adapter -- see §3.3).
2. Validate tool name against registry (discovered via MCP `tools/list`).
3. If sandbox is `consent` and the tool is destructive, prompt the user for approval.
4. Execute via MCP `tools/call` in the tool's per-session temp directory.
5. Return structured `ToolResult` to session loop.
6. Tools manage their own file I/O within their sandboxed working directory.

### Retry / Backoff
- Network retries use the `backoff` crate with exponential backoff + jitter.
- Retry policy is per-provider configurable (different providers have different rate-limit headers and semantics).
- Max retries and base delay come from `config/providers.yaml`.

*Harness never reads tool-generated files directly. Tools expose status via MCP resource endpoints or tool result content.*

---

## 8. Progression Phases (0 → Completion)

### Phase 0: Scaffolding & Toolchain
- `.cargo/config.toml` with aliases (`dev = "run --release"`)
- `.gitignore` (target/, *.db, ~/.harness/)
- `clippy.toml`, `rustfmt.toml`
- CI stub (`.github/workflows/ci.yml`)
- **Acceptance:** `cargo build`, `cargo clippy --all-targets`, `cargo fmt --all` all pass cleanly on the hello-world binary.

### Phase 1: Skeleton & Config
- Module layout (single crate: `cli/`, `core/`, `adapters/`, `tools/`, `config/`, `observability/`)
- `figment`-based config loader with strongly-typed structs
- Config schema validation (serde deserialization + custom validators)
- `PromptAnalyzer` trait + heuristic rule engine (tiktoken-rs tokenization)
- **Acceptance:** `config/routing.yaml` parsed, analyzer returns `ComplexityResult` for test prompts. No hardcoded values in the analyzer.

### Phase 2: Router & Adapter Interface
- `Router` trait + tier→model mapping
- `LLMAdapter` trait + mock provider
- `ResponseEvent` / `ToolInvocation` / `Usage` types
- Streaming output contract (`AdapterStream`)
- Cost tracking infrastructure (pricing table → `Usage.cost`)
- **Acceptance:** Router selects correct model from config; adapter streams mock chunks; cost is computed from mock usage.

### Phase 3: Interactive REPL
- Async CLI loop with `clap` + `reedline`
- Session state management (SQLite backend)
- Slash command parser (`/help`, `/tools`, `/model`, `/clear`, `/budget`, `/cost`, `/quit`)
- Streaming REPL output
- Context window management (sliding window strategy first)
- System prompt loading from file
- **Acceptance:** Interactive mode runs, accepts input, streams mock responses, handles `/clear`, `/help`, `/cost`. Session persists across restart.

### Phase 4: Tool System (MCP)
- MCP client (subprocess transport)
- `Tool` trait adapter + `ToolRegistry`
- Consent-based sandbox (user approval for destructive ops)
- Per-session temp directory isolation
- Tool lifecycle (RAII guard + `Drop` safety net)
- Integration with session loop (tool-call round-trip)
- **Acceptance:** MCP server registers, executes, returns results, cleans up. Harness routes calls without touching tool files.

### Phase 5: Classifier Routing
- Classifier stage for `PromptAnalyzer` (structured-output LLM call)
- Fallback to heuristic on classifier failure
- Config-driven enable/disable
- **Acceptance:** With classifier enabled, routing accuracy improves on test prompts. With classifier disabled or failing, heuristic routing still works.

### Phase 6: Advanced Sandbox & Hardening
- `wasmtime` sandbox implementation (optional, behind a feature flag)
- `docker` sandbox implementation (optional, behind a feature flag)
- Structured logging via `tracing` + `tracing-subscriber` (JSON output)
- OpenTelemetry export hooks (`tracing-opentelemetry`)
- Config override tests (env vars, CLI flags)
- Error recovery + fallback chains (backoff crate)
- Packaging (`cargo install`, Dockerfile)
- **Acceptance:** Full end-to-end test suite passes. CLI installs cleanly. All config-driven. `wasmtime`/`docker` sandboxes work for at least one tool. `tracing` output is structured JSON.

---

## 9. Constraints & Rules
- **Zero hardcoded config values:** All thresholds, keywords, model IDs, fallbacks, timeouts, budgets, and defaults must come from config or env vars. The matching ALGORITHM (case-insensitivity, prefix logic, structural detection rules) is compiled logic -- it is NOT config. The distinction: config holds WHAT to match; code holds HOW to match.
- **Decoupling first:** Define traits before implementations. No monolithic modules.
- **Tool ownership:** Tools create, modify, and clean up their own files within their sandbox. The harness only receives execution requests and routes them.
- **No provider coupling:** Adapters abstract SDKs. Harness never imports `openai`, `anthropic`, or `ollama` crates directly. HTTP calls go through a shared `reqwest`-based client.
- **Async throughout:** I/O, streaming, and REPL must use `async/await`. Dynamic-dispatch traits use `#[async_trait]`.
- **Strict typing:** Use `serde`, `thiserror` (for trait/library errors), `anyhow` (for application/CLI code only -- never in trait return types), `tokio`, `futures`, `async-trait`. No `unwrap()` or `expect()` in production paths (tests are fine).
- **Error types:** Each module defines a `thiserror` error enum (`AnalysisError`, `RoutingError`, `LLMError`, `ToolError`, `ConfigError`, `SessionError`). The application layer has a top-level `anyhow::Error` that wraps them. Trait methods return concrete errors, not `anyhow`.
- **Config validation:** Fail fast on invalid schema. `figment` provides the exact field path in errors.
- **Canonical model IDs:** All model references use the `ModelId` type (`provider/model`), parsed and validated at config load. No bare strings in routing logic.

---

## 10. Testing & Validation
- **Unit tests:** Analyzer (heuristic + classifier mock), Router, Config loader, Tool registry, Cost tracker
- **Mock adapters:** Return deterministic `ResponseEvent` chunks for reproducible REPL tests
- **Integration tests:** Full session loop with mock MCP servers and mock adapters
- **Config tests:** Schema validation, env override precedence, CLI flag override, missing field handling
- **Tool tests:** Consent sandbox (approval flow), temp dir isolation, cleanup on drop, error propagation
- **Token counting tests:** Verify tiktoken-rs output against known strings
- **Context management tests:** Sliding window eviction, summarization trigger, sticky system prompt
- **Coverage target:** ≥80% line coverage, 100% trait contract coverage

---

## 11. Build & Deployment

### Cargo Aliases (`.cargo/config.toml`)
```toml
[alias]
dev = "run --release"
```

### Commands
```bash
# Development
cargo dev            # runs: cargo run --release
cargo test --all
cargo clippy --all-targets -- -D warnings
cargo fmt --all

# Packaging
cargo install --path .
# or
docker build -t ai-harness .
```

### CI Pipeline (`.github/workflows/ci.yml`)
- `cargo check`, `clippy -- -D warnings`, `fmt --check`
- Unit + integration tests
- Config schema validation test
- Release binary + checksum

### Release Checklist
- [ ] All thresholds/keywords/models externalized to config
- [ ] Trait interfaces finalized and documented
- [ ] MCP tool system works with at least one external MCP server
- [ ] Consent sandbox prompts correctly for destructive ops
- [ ] REPL streams correctly, handles errors gracefully
- [ ] Session state persists in SQLite across restarts
- [ ] Context window management prevents overflow
- [ ] Cost tracking accumulates and warns at thresholds
- [ ] Config validation blocks invalid schemas with actionable errors
- [ ] Tests pass at ≥80% coverage
- [ ] `AGENTS.md` matches implementation

---

## Agent Execution Protocol
1. Read this file completely before generating code.
2. Follow phases sequentially (0 → 6). Do not skip acceptance criteria.
3. Output one phase per response. Include file tree, trait signatures, config examples, and test stubs.
4. Enforce constraints strictly. Reject hardcoded config values with explicit config alternatives.
5. When complete, output a final validation checklist matching Section 11.

Proceed phase-by-phase. Await confirmation before advancing.
