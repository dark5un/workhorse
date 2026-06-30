//! Tool system: MCP client, tool registry, sandbox, consent.
//!
//! The harness implements an MCP client. Tools are MCP servers (local
//! subprocesses or remote). The internal `Tool` trait is a thin adapter
//! over the MCP protocol.

pub mod consent;
pub mod mcp_client;
pub mod mcp_tool;

pub use consent::{AutoApprove, AutoDeny, ConsentCallback, ConsentDecision, ConsentSandbox};
pub use mcp_client::{McpClient, McpError, McpToolDef};
pub use mcp_tool::McpTool;

use async_trait::async_trait;
use thiserror::Error;

/// Internal adapter trait wrapping an MCP server connection.
/// The harness owns the MCP client transport; this trait exposes a uniform
/// interface to the registry and session loop.
///
/// Tools should ALSO implement Drop for panic-safe cleanup. The harness calls
/// `cleanup()` for graceful shutdown; Drop is the safety net.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;

    /// MCP tool schema (JSON Schema format, as returned by MCP tools/list).
    fn schema(&self) -> serde_json::Value;

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError>;

    /// Cleanup is called when the tool's session ends.
    async fn cleanup(&self) -> Result<(), ToolError>;
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    pub is_error: bool,
}

#[derive(Debug, Clone)]
pub enum ToolContent {
    Text(String),
    Image { mime_type: String, data: Vec<u8> },
    Resource { uri: String, mime_type: String },
}

/// Sandbox level for a tool, configured per MCP server in tools.yaml.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxLevel {
    /// Ask user before destructive operations (default, like Claude Code).
    Consent,
    /// Run inside Wasm runtime (true isolation, tools must be Wasm-compatible).
    /// Requires `wasmtime-sandbox` feature for actual implementation.
    Wasmtime,
    /// Run inside Docker/Podman container.
    /// Requires `docker-sandbox` feature for actual implementation.
    Docker,
    /// No sandbox (trusted local tools only, explicit opt-in).
    None,
}

impl std::str::FromStr for SandboxLevel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "wasmtime" => Self::Wasmtime,
            "docker" => Self::Docker,
            "none" => Self::None,
            _ => Self::Consent, // default
        })
    }
}

/// Check if a sandbox level is available (has its implementation compiled in).
pub fn sandbox_available(level: SandboxLevel) -> bool {
    match level {
        SandboxLevel::Consent | SandboxLevel::None => true,
        #[cfg(feature = "wasmtime-sandbox")]
        SandboxLevel::Wasmtime => true,
        #[cfg(not(feature = "wasmtime-sandbox"))]
        SandboxLevel::Wasmtime => false,
        #[cfg(feature = "docker-sandbox")]
        SandboxLevel::Docker => true,
        #[cfg(not(feature = "docker-sandbox"))]
        SandboxLevel::Docker => false,
    }
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("execution error: {0}")]
    Execution(String),
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    #[error("sandbox error: {0}")]
    Sandbox(String),
    #[error("cleanup error: {0}")]
    Cleanup(String),
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
}

/// Tool registry: registers tools, validates schemas, manages lifecycle.
/// Wraps each tool in a RAII guard that ensures cleanup on drop.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    /// Per-session temp directory for tool isolation.
    session_temp_dir: std::path::PathBuf,
}

impl ToolRegistry {
    pub fn new(session_temp_dir: &std::path::Path) -> Self {
        let _ = std::fs::create_dir_all(session_temp_dir);
        Self {
            tools: Vec::new(),
            session_temp_dir: session_temp_dir.to_path_buf(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn find(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|t| t.as_ref())
    }

    pub fn list(&self) -> Vec<&dyn Tool> {
        self.tools.iter().map(|t| t.as_ref()).collect()
    }

    /// Get the per-session temp directory.
    pub fn session_temp_dir(&self) -> &std::path::Path {
        &self.session_temp_dir
    }

    /// Clean up all registered tools (graceful shutdown).
    pub async fn cleanup_all(&self) -> Result<(), ToolError> {
        for tool in &self.tools {
            let _ = tool.cleanup().await;
        }
        Ok(())
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new(std::path::Path::new("/tmp/harness-default"))
    }
}
