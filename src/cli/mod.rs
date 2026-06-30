//! CLI: REPL loop, input parsing, streaming output.
//!
//! Basic REPL using stdin/stdout. The `repl` feature flag enables
//! reedline (line editor with history, syntax highlighting).

use anyhow::Result;
use std::io::{BufRead, Write};

use crate::core::{Session, SessionController, SessionEvent};

/// Entry point for the CLI. Runs the interactive REPL.
pub async fn run() -> Result<()> {
    crate::observability::init();

    let config = crate::config::load_config("config").map_err(|e| {
        tracing::error!(error = %e, "config load failed");
        anyhow::anyhow!(e)
    })?;
    tracing::info!(storage = %config.session.storage, "config loaded");

    // Expand ~ in DB path
    let db_path = expand_tilde(&config.session.path);

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut session = Session::new(config, &db_path, "default")?;

    println!("workhorse: interactive LLM harness");
    println!("Type /help for commands, /quit to exit.\n");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    loop {
        write!(stdout, "> ")?;
        stdout.flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" {
            println!("Goodbye.");
            break;
        }

        match session.process(input).await {
            Ok(output) => {
                tracing::info!(
                    events = output.events.len(),
                    has_usage = output.usage.is_some(),
                    "turn completed"
                );
                for event in &output.events {
                    match event {
                        SessionEvent::Text(text) => print!("{text}"),
                        SessionEvent::ToolCall(inv) => {
                            println!("\n[tool call: {} ({})]", inv.tool_name, inv.call_id);
                        }
                        SessionEvent::ToolResult(result) => {
                            println!("\n[tool result: error={}]", result.is_error);
                        }
                        SessionEvent::Error(err) => {
                            eprintln!("\n[error: {err}]");
                        }
                    }
                }
                println!();

                // Show which model was used, after the response
                if let Some(model) = &output.model_used {
                    let cost_str = output
                        .usage
                        .as_ref()
                        .map(|u| format!(" | cost: ${:.2}", u.cost.as_usd()))
                        .unwrap_or_default();
                    let token_str = output
                        .usage
                        .as_ref()
                        .map(|u| format!(" | tokens: {}/{}", u.input_tokens, u.output_tokens))
                        .unwrap_or_default();
                    println!("[model: {}{token_str}{cost_str}]", model.as_str());
                }
            }
            Err(e) => {
                eprintln!("Error: {e}");
            }
        }
    }

    crate::observability::shutdown();
    Ok(())
}
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}
