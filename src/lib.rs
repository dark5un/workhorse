//! workhorse -- Rust-based interactive CLI harness for LLM routing,
//! MCP tool execution, and persistent sessions.
//!
//! Module layout (single crate, not a workspace):
//! - `core`: Router, Analyzer, Session, shared types
//! - `adapters`: LLM trait + provider implementations
//! - `tools`: MCP client, tool registry, sandbox
//! - `config`: figment-based config loading and schema
//! - `cli`: REPL loop, input parsing, streaming output
//! - `observability`: tracing-based structured logging

pub mod adapters;
pub mod cli;
pub mod config;
pub mod core;
pub mod observability;
pub mod tools;
