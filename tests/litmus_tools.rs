//! Litmus tests: tool system contracts (AGENTS.md 3.4, 7).
//!
//! Some tests pass now (ToolRegistry register/find/list). Others are
//! #[ignore] until Phase 4 (MCP tool system) is implemented.

use async_trait::async_trait;
use myharness::tools::{Tool, ToolContent, ToolError, ToolRegistry, ToolResult};

// ============================================================
// ToolRegistry contracts (pass now)
// ============================================================

#[test]
fn registry_registers_and_finds_tool_by_name() {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(DummyTool {
        name: "echo".to_string(),
    }));
    assert!(registry.find("echo").is_some());
    assert!(registry.find("nonexistent").is_none());
}

#[test]
fn registry_lists_all_registered_tools() {
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(DummyTool {
        name: "tool_a".to_string(),
    }));
    registry.register(Box::new(DummyTool {
        name: "tool_b".to_string(),
    }));
    let tools = registry.list();
    assert_eq!(tools.len(), 2);
}

#[test]
fn registry_starts_empty() {
    let registry = ToolRegistry::new();
    assert_eq!(registry.list().len(), 0);
}

// ============================================================
// MCP contracts (Phase 4)
// ============================================================

#[ignore = "Phase 4: MCP client not yet implemented"]
#[tokio::test]
async fn mcp_server_discovered_via_tools_list() {
    // After starting an MCP server, the registry should discover its tools
    // via the MCP tools/list method.
    let registry = create_mcp_registry();
    let tools = registry.list();
    assert!(
        !tools.is_empty(),
        "MCP server must expose at least one tool"
    );
}

#[ignore = "Phase 4: MCP client not yet implemented"]
#[tokio::test]
async fn mcp_tool_execution_returns_result() {
    let registry = create_mcp_registry();
    let tool = registry.find("echo").expect("echo tool must be registered");
    let result = tool
        .execute(serde_json::json!({"text": "hello"}))
        .await
        .unwrap();
    assert!(!result.content.is_empty());
    // MCP tool results contain content blocks
    let has_text = result
        .content
        .iter()
        .any(|c| matches!(c, ToolContent::Text(_)));
    assert!(has_text, "tool result should contain text content");
}

#[ignore = "Phase 4: MCP client not yet implemented"]
#[tokio::test]
async fn mcp_tool_result_can_be_error() {
    let registry = create_mcp_registry();
    let tool = registry.find("echo").unwrap();
    // Invalid arguments should produce is_error=true
    let result = tool
        .execute(serde_json::json!({"wrong_arg": true}))
        .await
        .unwrap();
    assert!(result.is_error, "invalid args should produce error result");
}

// ============================================================
// Sandbox contracts (Phase 4 / 6)
// ============================================================

#[ignore = "Phase 4: consent sandbox not yet implemented"]
#[tokio::test]
async fn consent_sandbox_prompts_for_destructive_ops() {
    // When sandbox=consent, destructive operations (file write, shell exec)
    // must prompt the user for approval before executing.
    // This test verifies the consent flow is triggered.
}

#[ignore = "Phase 4: per-session temp dir not yet implemented"]
#[tokio::test]
async fn tool_executes_in_per_session_temp_dir() {
    // Each tool execution must run in an isolated temp directory
    // (/tmp/harness-<session-id>/) to prevent concurrent file access races.
}

#[ignore = "Phase 4: tool cleanup on drop not yet implemented"]
#[tokio::test]
async fn tool_cleanup_called_on_session_end() {
    // When the session ends, the registry calls cleanup() on each tool.
    // The RAII guard's Drop is the safety net for panic-safe cleanup.
}

// ============================================================
// Mock implementations
// ============================================================

struct DummyTool {
    name: String,
}

#[async_trait]
impl Tool for DummyTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        "A dummy tool for testing"
    }

    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" }
            }
        })
    }

    async fn execute(&self, _args: serde_json::Value) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            content: vec![ToolContent::Text("ok".to_string())],
            is_error: false,
        })
    }

    async fn cleanup(&self) -> Result<(), ToolError> {
        Ok(())
    }
}

fn create_mcp_registry() -> ToolRegistry {
    unimplemented!("Phase 4")
}
