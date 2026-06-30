//! Session controller: REPL loop, context window management, state persistence.

use async_trait::async_trait;
use thiserror::Error;

/// Controls the interactive session: processing input, managing state, reset.
#[async_trait]
pub trait SessionController: Send + Sync {
    async fn process(&mut self, input: &str) -> Result<SessionOutput, SessionError>;
    async fn reset(&mut self);
    fn status(&self) -> SessionState;
}

/// Output from processing a single input turn.
#[derive(Debug, Clone)]
pub struct SessionOutput {
    pub events: Vec<SessionEvent>,
    pub usage: Option<super::super::adapters::Usage>,
}

#[derive(Debug, Clone)]
pub enum SessionEvent {
    Text(String),
    ToolCall(super::super::adapters::ToolInvocation),
    ToolResult(super::super::tools::ToolResult),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub message_count: usize,
    pub total_cost_cents: u64,
    pub current_model: Option<String>,
    pub context_tokens_used: usize,
    pub context_token_limit: usize,
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("context window exhausted")]
    ContextExhausted,
    #[error("budget exceeded: ${0:.2} spent, limit ${1:.2}")]
    BudgetExceeded(f64, f64),
}
