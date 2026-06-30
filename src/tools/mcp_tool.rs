//! MCP tool adapter: wraps an MCP client connection and implements the Tool trait.
//!
//! Each McpTool corresponds to a single tool discovered via MCP tools/list.
//! The MCP client (subprocess) is shared across tools from the same server.

use async_trait::async_trait;
use std::sync::Arc;

use super::consent::ConsentSandbox;
use super::mcp_client::{McpClient, McpToolDef};
use super::{Tool, ToolError, ToolResult};

/// Tool adapter that wraps an MCP client for a specific discovered tool.
///
/// The MCP client (subprocess) is held in an Arc for shared ownership.
/// The consent sandbox is optional -- when present, destructive operations
/// are intercepted before being sent to the MCP server.
pub struct McpTool {
    def: McpToolDef,
    client: Arc<McpClient>,
    sandbox: Option<Arc<ConsentSandbox>>,
    /// Whether this tool is considered destructive (triggers consent).
    destructive: bool,
    cleaned_up: std::sync::atomic::AtomicBool,
}

impl McpTool {
    pub fn new(
        def: McpToolDef,
        client: Arc<McpClient>,
        sandbox: Option<Arc<ConsentSandbox>>,
        destructive: bool,
    ) -> Self {
        Self {
            def,
            client,
            sandbox,
            destructive,
            cleaned_up: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Mark this tool as destructive (triggers consent sandbox).
    pub fn set_destructive(&mut self, destructive: bool) {
        self.destructive = destructive;
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn description(&self) -> &str {
        &self.def.description
    }

    fn schema(&self) -> serde_json::Value {
        self.def.schema.clone()
    }

    async fn execute(&self, args: serde_json::Value) -> Result<ToolResult, ToolError> {
        // Check consent sandbox for destructive operations
        if self.destructive {
            if let Some(sandbox) = &self.sandbox {
                sandbox.check(&self.def.name, "execute", &args)?;
            }
        }

        self.client
            .call_tool(&self.def.name, args)
            .await
            .map_err(ToolError::from)
    }

    async fn cleanup(&self) -> Result<(), ToolError> {
        // Cleanup is called once; the actual subprocess shutdown is handled
        // by the MCP client's Drop impl.
        self.cleaned_up
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

impl Drop for McpTool {
    fn drop(&mut self) {
        // Safety net: if cleanup() wasn't called, mark it as cleaned up.
        // The actual subprocess kill is handled by McpClient's Drop.
        self.cleaned_up
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}
