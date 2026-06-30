//! MCP client: JSON-RPC 2.0 over subprocess stdio.
//!
//! The client spawns an MCP server as a subprocess, communicates via
//! stdin/stdout using JSON-RPC 2.0, and discovers tools via tools/list.
//!
//! Protocol reference: https://spec.modelcontextprotocol.io/

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::Mutex as AsyncMutex;

use super::{ToolContent, ToolError, ToolResult};

/// MCP tool definition as returned by tools/list.
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

/// Errors specific to MCP client operations.
#[derive(Debug, Error)]
pub enum McpError {
    #[error("failed to spawn MCP server: {0}")]
    Spawn(String),
    #[error("JSON-RPC error: {0}")]
    JsonRpc(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("server returned error: {0}")]
    ServerError(String),
    #[error("IO error: {0}")]
    Io(String),
}

impl From<McpError> for ToolError {
    fn from(e: McpError) -> Self {
        ToolError::Execution(e.to_string())
    }
}

/// MCP client that communicates with a server subprocess via JSON-RPC 2.0.
pub struct McpClient {
    /// Server subprocess handle.
    child: AsyncMutex<Option<Child>>,
    /// Stdin for writing JSON-RPC requests.
    stdin: AsyncMutex<Option<ChildStdin>>,
    /// Stdout reader (buffered) for reading JSON-RPC responses.
    stdout: AsyncMutex<Option<BufReader<ChildStdout>>>,
    /// Request ID counter.
    request_id: AtomicU64,
    /// Tool definitions discovered via tools/list.
    tools: Mutex<HashMap<String, McpToolDef>>,
    /// Working directory for tool execution (per-session temp dir).
    working_dir: std::path::PathBuf,
}

impl McpClient {
    /// Spawn an MCP server subprocess and initialize the connection.
    pub async fn spawn(
        command: &str,
        args: &[String],
        working_dir: &std::path::Path,
    ) -> Result<Self, McpError> {
        // Ensure working dir exists
        std::fs::create_dir_all(working_dir)
            .map_err(|e| McpError::Spawn(format!("failed to create working dir: {e}")))?;

        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(working_dir)
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| McpError::Spawn(format!("failed to spawn '{command}': {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Spawn("failed to capture stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Spawn("failed to capture stdout".to_string()))?;

        let client = Self {
            child: AsyncMutex::new(Some(child)),
            stdin: AsyncMutex::new(Some(stdin)),
            stdout: AsyncMutex::new(Some(BufReader::new(stdout))),
            request_id: AtomicU64::new(1),
            tools: Mutex::new(HashMap::new()),
            working_dir: working_dir.to_path_buf(),
        };

        // Initialize the MCP connection
        client.initialize().await?;

        Ok(client)
    }

    /// Send the MCP initialize request.
    async fn initialize(&self) -> Result<(), McpError> {
        let _response = self
            .send_request(
                "initialize",
                serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "myharness",
                        "version": "0.1.0"
                    }
                }),
            )
            .await?;

        // Send initialized notification (no response expected)
        self.send_notification("notifications/initialized", serde_json::json!({}))
            .await?;

        Ok(())
    }

    /// Discover tools via MCP tools/list.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDef>, McpError> {
        let response = self
            .send_request("tools/list", serde_json::json!({}))
            .await?;

        let tools_json = response
            .get("tools")
            .ok_or_else(|| McpError::Protocol("missing 'tools' in response".to_string()))?;

        let mut tools = Vec::new();
        if let Some(arr) = tools_json.as_array() {
            for tool_json in arr {
                let name = tool_json
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| McpError::Protocol("tool missing 'name'".to_string()))?
                    .to_string();
                let description = tool_json
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let schema = tool_json
                    .get("inputSchema")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                let def = McpToolDef {
                    name: name.clone(),
                    description,
                    schema,
                };
                tools.push(def.clone());
                self.tools.lock().unwrap().insert(name, def);
            }
        }

        Ok(tools)
    }

    /// Call a tool via MCP tools/call.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult, McpError> {
        let response = self
            .send_request(
                "tools/call",
                serde_json::json!({
                    "name": name,
                    "arguments": arguments
                }),
            )
            .await?;

        // Check for error
        if let Some(error) = response.get("isError").and_then(|v| v.as_bool()) {
            if error {
                return Ok(ToolResult {
                    content: self.parse_content(&response)?,
                    is_error: true,
                });
            }
        }

        Ok(ToolResult {
            content: self.parse_content(&response)?,
            is_error: false,
        })
    }

    /// Parse MCP content blocks from a tools/call response.
    fn parse_content(&self, response: &serde_json::Value) -> Result<Vec<ToolContent>, McpError> {
        let content_arr = response
            .get("content")
            .and_then(|v| v.as_array())
            .ok_or_else(|| McpError::Protocol("missing 'content' in response".to_string()))?;

        let mut result = Vec::new();
        for item in content_arr {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("text");
            match item_type {
                "text" => {
                    let text = item
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    result.push(ToolContent::Text(text));
                }
                "image" => {
                    if let (Some(mime), Some(data)) = (
                        item.get("mimeType").and_then(|v| v.as_str()),
                        item.get("data").and_then(|v| v.as_str()),
                    ) {
                        if let Ok(decoded) = base64_decode(data) {
                            result.push(ToolContent::Image {
                                mime_type: mime.to_string(),
                                data: decoded,
                            });
                        }
                    }
                }
                "resource" => {
                    if let (Some(uri), Some(mime)) = (
                        item.get("uri").and_then(|v| v.as_str()),
                        item.get("mimeType").and_then(|v| v.as_str()),
                    ) {
                        result.push(ToolContent::Resource {
                            uri: uri.to_string(),
                            mime_type: mime.to_string(),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(result)
    }

    /// Get a discovered tool definition by name.
    pub fn get_tool_def(&self, name: &str) -> Option<McpToolDef> {
        self.tools.lock().unwrap().get(name).cloned()
    }

    /// Send a JSON-RPC 2.0 request and wait for the response.
    async fn send_request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, McpError> {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        let request_str =
            serde_json::to_string(&request).map_err(|e| McpError::JsonRpc(e.to_string()))?;

        // Write request to stdin
        {
            let mut stdin_guard = self.stdin.lock().await;
            let stdin = stdin_guard
                .as_mut()
                .ok_or_else(|| McpError::Io("stdin not available".to_string()))?;
            stdin
                .write_all(request_str.as_bytes())
                .await
                .map_err(|e| McpError::Io(e.to_string()))?;
            stdin
                .write_all(b"\n")
                .await
                .map_err(|e| McpError::Io(e.to_string()))?;
            stdin
                .flush()
                .await
                .map_err(|e| McpError::Io(e.to_string()))?;
        }

        // Read response from stdout
        let response_str = {
            let mut stdout_guard = self.stdout.lock().await;
            let stdout = stdout_guard
                .as_mut()
                .ok_or_else(|| McpError::Io("stdout not available".to_string()))?;
            let mut line = String::new();
            stdout
                .read_line(&mut line)
                .await
                .map_err(|e| McpError::Io(e.to_string()))?;
            line
        };

        let response: serde_json::Value = serde_json::from_str(response_str.trim())
            .map_err(|e| McpError::JsonRpc(format!("failed to parse response: {e}")))?;

        // Check for JSON-RPC error
        if let Some(error) = response.get("error") {
            let message = error
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown error");
            return Err(McpError::ServerError(message.to_string()));
        }

        // Return the result field
        Ok(response
            .get("result")
            .cloned()
            .unwrap_or(serde_json::json!({})))
    }

    /// Send a JSON-RPC 2.0 notification (no response expected).
    async fn send_notification(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<(), McpError> {
        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        let notif_str =
            serde_json::to_string(&notification).map_err(|e| McpError::JsonRpc(e.to_string()))?;

        let mut stdin_guard = self.stdin.lock().await;
        let stdin = stdin_guard
            .as_mut()
            .ok_or_else(|| McpError::Io("stdin not available".to_string()))?;
        stdin
            .write_all(notif_str.as_bytes())
            .await
            .map_err(|e| McpError::Io(e.to_string()))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::Io(e.to_string()))?;
        stdin
            .flush()
            .await
            .map_err(|e| McpError::Io(e.to_string()))?;

        Ok(())
    }

    /// Get the working directory for this client.
    pub fn working_dir(&self) -> &std::path::Path {
        &self.working_dir
    }

    /// Shut down the MCP server subprocess.
    pub async fn shutdown(&self) -> Result<(), McpError> {
        // Try graceful shutdown first
        let _ = self
            .send_notification("notifications/cancelled", serde_json::json!({}))
            .await;

        // Close stdin to signal EOF
        {
            let mut stdin_guard = self.stdin.lock().await;
            stdin_guard.take(); // Drop the stdin handle
        }

        // Kill the process if still running
        let mut child_guard = self.child.lock().await;
        if let Some(child) = child_guard.as_mut() {
            let _ = child.kill().await;
            let _ = child.wait().await;
        }
        *child_guard = None;

        Ok(())
    }

    /// Check if the subprocess is still running.
    pub async fn is_alive(&self) -> bool {
        let child_guard = self.child.lock().await;
        child_guard.is_some()
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Safety net: kill the subprocess if still running.
        // We can't await in Drop, so we try to take the child and kill it.
        // try_lock() avoids blocking if the lock is held (e.g. during shutdown).
        if let Ok(mut guard) = self.child.try_lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
            *guard = None;
        }
        // If try_lock fails, the async shutdown path is already handling it.
    }
}

/// Simple base64 decoder (avoids adding a base64 dependency).
fn base64_decode(input: &str) -> Result<Vec<u8>, ()> {
    let input = input.trim();
    let mut output = Vec::new();
    let lookup = |c: u8| -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    };

    let bytes: Vec<u8> = input.bytes().filter(|&b| b != b'=' && b != b'\n').collect();
    for chunk in bytes.chunks(4) {
        let mut vals: [u8; 4] = [0; 4];
        for (i, &b) in chunk.iter().enumerate() {
            vals[i] = lookup(b).ok_or(())?;
        }
        output.push((vals[0] << 2) | (vals[1] >> 4));
        if chunk.len() > 2 {
            output.push((vals[1] << 4) | (vals[2] >> 2));
        }
        if chunk.len() > 3 {
            output.push((vals[2] << 6) | vals[3]);
        }
    }

    Ok(output)
}
