# Workhorse

Workhorse is a Rust-based interactive CLI harness for routing prompts across multiple LLMs, executing tools via MCP (Model Context Protocol), and persisting conversational session state.

The system is config-driven and trait-first:
- Prompt complexity is analyzed first, then routed to the best model for the task.
- Routing and fallbacks are defined in config files, not hardcoded values.
- Tool execution is handled through MCP servers.
- Session history and metadata are persisted for resumable workflows.

## Current Naming

The project is named Workhorse, and the executable binary name is lowercase: workhorse.

Current identifiers:
- Cargo package: workhorse
- Library crate: workhorse
- CLI binary: workhorse

## Key Features

- Two-stage prompt analysis flow:
  - Heuristic stage (token length, keywords, structure)
  - Optional classifier stage (can override heuristic result)
- Classifier-safe fallback behavior:
  - If classifier fails (parse/network/timeout), heuristic result is used
  - Decision source is preserved in analysis metadata
- Config-driven router with model fallback chains per complexity tier
- Interactive REPL session with slash commands
- Persistent SQLite-backed session storage
- Cost tracking using provider pricing config
- MCP tool-call round trip support and tool event surfacing
- Structured observability via tracing

## Project Layout

Single crate, modular architecture:

- src/cli: REPL loop and user input/output
- src/core: analyzer, router, session, ranking
- src/adapters: adapter traits and provider implementations
- src/tools: MCP integration and sandbox controls
- src/config: schema + figment loading/validation
- src/observability: tracing setup and hooks

## Requirements

- Rust stable toolchain (edition 2024)
- Cargo
- SQLite (bundled via rusqlite feature in this project)

## Quick Start

1. Build

   cargo build

2. Run interactive CLI

   cargo run

3. Run tests

   cargo test --all

4. Lint and format checks

   cargo clippy --all-targets -- -D warnings
   cargo fmt --all --check

## Optional Feature Flags

- providers: enables HTTP provider integration
- repl: enables full line-editor stack (clap + reedline)
- otel: enables OpenTelemetry hooks
- wasmtime-sandbox: optional Wasmtime sandbox backend
- docker-sandbox: optional Docker sandbox backend

Examples:

- cargo run --features providers
- cargo test --all --features "providers repl"

## Configuration

Workhorse loads layered configuration from the config directory:
- config/routing.yaml
- config/tools.yaml
- config/providers.yaml
- config/session.yaml

Precedence:
1. YAML files
2. Environment variables (prefix HARNESS_, split by __)
3. CLI-level overrides (where implemented)

Example env override:

HARNESS_SESSION__PATH=~/.workhorse/sessions.db

## Runtime Behavior

For each user prompt:
1. Heuristic analyzer computes a complexity result.
2. Optional classifier can override the heuristic result.
3. On classifier failure, routing falls back to heuristic output.
4. Router selects primary model plus fallback chain from config.
5. Session executes the turn, streams events, records usage/cost.

The user can force a model with the model override command.

## REPL Commands

Implemented commands include:
- /help
- /tools
- /model <provider/model>
- /clear
- /budget <tokens>
- /cost
- /rate [1-5]
- /rate <provider/model> [1-5]
- /ratings [tier]
- /reset-ratings [global]
- /ranking on|off|status
- /quit

## MCP Tooling Notes

- MCP servers are configured externally in config/tools.yaml.
- Consent sandbox is the default mode.
- Tool invocations are surfaced as session events.
- Tool output is returned through structured tool result payloads.

## Persistence and Cost

- Session messages are persisted in SQLite at the configured session path.
- System prompt is loaded from config/system_prompt.md.
- Cost accounting is computed from provider pricing tables.
- Hard/warn budget thresholds are configured in session config.

## Development Workflow

Common commands:

- cargo build
- cargo run
- cargo test --all
- cargo clippy --all-targets -- -D warnings
- cargo fmt --all

Packaging:

- cargo install --path .
- docker build -t workhorse .

## Roadmap Context

The project follows phased implementation in AGENTS.md (Phase 0 through Phase 6), covering:
- core scaffold and config
- router + adapter contracts
- interactive REPL + persistence
- MCP tool system
- classifier routing
- hardening and advanced sandbox options

## License

See LICENSE.
