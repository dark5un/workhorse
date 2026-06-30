//! LLM adapter abstraction: trait + types for provider-agnostic LLM calls.
//!
//! Providers implement `LLMAdapter`. The harness never calls provider-specific
//! SDKs directly. Adapters normalize provider-specific tool-call schemas into
//! the common `ResponseEvent` type.

pub mod mock;
pub mod retry;

pub use mock::MockAdapter;
pub use retry::{RetryError, RetryPolicy};

use async_trait::async_trait;
use thiserror::Error;

use crate::core::{Cost, Message};

/// Abstract LLM adapter. Providers implement this trait.
///
/// Uses `&self` -- adapters manage mutable state (HTTP client pools, etc.)
/// via interior mutability (Arc<Mutex<...>>).
#[async_trait]
pub trait LLMAdapter: Send + Sync {
    /// Send a completion request, returning a stream of response events.
    async fn send(
        &self,
        messages: Vec<Message>,
        config: ModelConfig,
    ) -> Result<Vec<ResponseEvent>, LLMError>;

    fn capabilities(&self) -> ModelCapabilities;
}

/// Normalized provider response. Adapters translate provider-specific
/// tool-call formats into this common shape.
#[derive(Debug, Clone)]
pub enum ResponseEvent {
    /// Text delta (streaming).
    Chunk(String),
    /// Normalized tool call request.
    ToolCall(ToolInvocation),
    /// Completion with token usage and cost.
    Done(Usage),
}

#[derive(Debug, Clone)]
pub struct ToolInvocation {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost: Cost,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
    pub tools: Option<Vec<ToolSpec>>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    Text,
    Json,
}

#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub streaming: bool,
    pub tool_calling: bool,
    pub structured_output: bool,
    pub vision: bool,
    pub max_context_tokens: usize,
}

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("network error: {0}")]
    Network(String),
    #[error("authentication error: {0}")]
    Auth(String),
    #[error("rate limited: retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("model error: {0}")]
    Model(String),
    #[error("response parse error: {0}")]
    Parse(String),
    #[error("budget exceeded")]
    BudgetExceeded,
}
