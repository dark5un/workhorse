//! Mock MCP server for testing.
//!
//! Implements a minimal MCP server over stdio (JSON-RPC 2.0) with an "echo" tool.
//! Used by litmus tests to verify the MCP client without external dependencies.
//!
//! Usage: mock_mcp_server
//! Reads JSON-RPC from stdin, writes responses to stdout.

use std::io::{self, BufRead, Write};

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let method = request.get("method").and_then(|v| v.as_str()).unwrap_or("");

        // Notifications don't have an id and don't get a response
        let id = request.get("id").cloned();
        if id.is_none() {
            continue;
        }

        let response = match method {
            "initialize" => serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "mock-mcp-server",
                        "version": "0.1.0"
                    }
                }
            }),
            "tools/list" => serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echoes back the input text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "Text to echo back"
                                    }
                                },
                                "required": ["text"]
                            }
                        }
                    ]
                }
            }),
            "tools/call" => {
                let name = request
                    .pointer("/params/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let arguments = request
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                if name == "echo" {
                    // Check if "text" argument is present
                    if let Some(text) = arguments.get("text").and_then(|v| v.as_str()) {
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": text
                                    }
                                ],
                                "isError": false
                            }
                        })
                    } else {
                        // Missing required "text" argument
                        serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Error: missing required argument 'text'"
                                    }
                                ],
                                "isError": true
                            }
                        })
                    }
                } else {
                    serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": format!("Unknown tool: {name}")
                        }
                    })
                }
            }
            _ => serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("Unknown method: {method}")
                }
            }),
        };

        let response_str = serde_json::to_string(&response).unwrap();
        let _ = writeln!(stdout, "{response_str}");
        let _ = stdout.flush();
    }
}
