//! Litmus tests: tool system contracts (AGENTS.md 3.4, 7).
//!
//! Phase 4 tests (MCP client, consent sandbox, tool lifecycle) are enabled.

use async_trait::async_trait;
use myharness::tools::{
    ConsentSandbox, McpClient, McpTool, Tool, ToolContent, ToolError, ToolRegistry, ToolResult,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

static DIR_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Generate a unique temp directory for each test invocation.
fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = DIR_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("harness-test-{prefix}-{id}"));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

// ============================================================
// ToolRegistry contracts (pass since Phase 0)
// ============================================================

#[test]
fn registry_registers_and_finds_tool_by_name() {
    let dir = unique_temp_dir("registry");
    let mut registry = ToolRegistry::new(&dir);
    registry.register(Box::new(DummyTool {
        name: "echo".to_string(),
    }));
    assert!(registry.find("echo").is_some());
    assert!(registry.find("nonexistent").is_none());
}

#[test]
fn registry_lists_all_registered_tools() {
    let dir = unique_temp_dir("registry-list");
    let mut registry = ToolRegistry::new(&dir);
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
    let dir = unique_temp_dir("registry-empty");
    let registry = ToolRegistry::new(&dir);
    assert_eq!(registry.list().len(), 0);
}

// ============================================================
// MCP contracts (Phase 4) -- ENABLED
// ============================================================

#[tokio::test]
async fn mcp_server_discovered_via_tools_list() {
    let registry = create_mcp_registry().await;
    let tools = registry.list();
    assert!(
        !tools.is_empty(),
        "MCP server must expose at least one tool"
    );
    assert!(tools.iter().any(|t| t.name() == "echo"));
}

#[tokio::test]
async fn mcp_tool_execution_returns_result() {
    let registry = create_mcp_registry().await;
    let tool = registry.find("echo").expect("echo tool must be registered");
    let result = tool
        .execute(serde_json::json!({"text": "hello"}))
        .await
        .unwrap();
    assert!(!result.content.is_empty());
    let has_text = result
        .content
        .iter()
        .any(|c| matches!(c, ToolContent::Text(_)));
    assert!(has_text, "tool result should contain text content");
    if let Some(ToolContent::Text(text)) = result.content.first() {
        assert_eq!(text, "hello");
    }
}

#[tokio::test]
async fn mcp_tool_result_can_be_error() {
    let registry = create_mcp_registry().await;
    let tool = registry.find("echo").unwrap();
    let result = tool
        .execute(serde_json::json!({"wrong_arg": true}))
        .await
        .unwrap();
    assert!(result.is_error, "invalid args should produce error result");
}

// ============================================================
// Sandbox contracts (Phase 4) -- ENABLED
// ============================================================

#[tokio::test]
async fn consent_sandbox_prompts_for_destructive_ops() {
    let consent = Arc::new(ConsentSandbox::new(Box::new(TrackingConsent {
        count: AtomicU32::new(0),
    })));

    let working_dir = unique_temp_dir("consent");
    let client = Arc::new(
        McpClient::spawn(&get_mock_server_path(), &[], &working_dir)
            .await
            .unwrap(),
    );
    let tools = client.list_tools().await.unwrap();
    let def = tools.into_iter().find(|t| t.name == "echo").unwrap();

    let tool = McpTool::new(def, client, Some(consent.clone()), true);

    let result = tool
        .execute(serde_json::json!({"text": "test"}))
        .await
        .unwrap();
    assert!(!result.is_error);

    assert_eq!(consent.consent_request_count(), 1);
    assert!(consent.last_operation_allowed());
}

#[tokio::test]
async fn consent_sandbox_denies_when_callback_denies() {
    let consent = Arc::new(ConsentSandbox::new(Box::new(myharness::tools::AutoDeny)));

    let working_dir = unique_temp_dir("deny");
    let client = Arc::new(
        McpClient::spawn(&get_mock_server_path(), &[], &working_dir)
            .await
            .unwrap(),
    );
    let tools = client.list_tools().await.unwrap();
    let def = tools.into_iter().find(|t| t.name == "echo").unwrap();

    let tool = McpTool::new(def, client, Some(consent), true);

    let result = tool.execute(serde_json::json!({"text": "test"})).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        ToolError::PermissionDenied(_) => {}
        e => panic!("expected PermissionDenied, got: {e}"),
    }
}

// ============================================================
// Per-session temp dir contract (Phase 4) -- ENABLED
// ============================================================

#[tokio::test]
async fn tool_executes_in_per_session_temp_dir() {
    let session_dir = unique_temp_dir("session-dir");
    let client = McpClient::spawn(&get_mock_server_path(), &[], &session_dir)
        .await
        .unwrap();

    assert!(client.working_dir().exists());
    assert_eq!(client.working_dir(), session_dir.as_path());

    let _ = client.shutdown().await;
}

// ============================================================
// Tool cleanup on drop contract (Phase 4) -- ENABLED
// ============================================================

#[tokio::test]
async fn tool_cleanup_called_on_session_end() {
    let working_dir = unique_temp_dir("cleanup");
    let client = Arc::new(
        McpClient::spawn(&get_mock_server_path(), &[], &working_dir)
            .await
            .unwrap(),
    );

    assert!(client.is_alive().await);

    client.shutdown().await.unwrap();

    assert!(!client.is_alive().await);
}

// ============================================================
// Helper implementations
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

struct TrackingConsent {
    count: AtomicU32,
}

impl myharness::tools::ConsentCallback for TrackingConsent {
    fn request_consent(
        &self,
        _tool_name: &str,
        _operation: &str,
        _args: &serde_json::Value,
    ) -> myharness::tools::ConsentDecision {
        self.count.fetch_add(1, Ordering::SeqCst);
        myharness::tools::ConsentDecision::Allow
    }
}

fn get_mock_server_path() -> String {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let exe_name = if cfg!(windows) {
        "mock_mcp_server.exe"
    } else {
        "mock_mcp_server"
    };
    format!("{manifest_dir}/target/{profile}/{exe_name}")
}

async fn create_mcp_registry() -> ToolRegistry {
    let working_dir = unique_temp_dir("mcp-registry");
    let client = Arc::new(
        McpClient::spawn(&get_mock_server_path(), &[], &working_dir)
            .await
            .unwrap(),
    );

    let tools = client.list_tools().await.unwrap();
    let mut registry = ToolRegistry::new(&working_dir);

    for def in tools {
        let tool = McpTool::new(def, client.clone(), None, false);
        registry.register(Box::new(tool));
    }

    registry
}
