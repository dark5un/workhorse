//! Mock LLM adapter for testing and development.
//!
//! Produces deterministic ResponseEvents based on input messages.
//! Computes cost from a pricing table. Does not make network calls.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::adapters::{
    LLMAdapter, LLMError, ModelCapabilities, ModelConfig, ResponseEvent, ToolInvocation, Usage,
};
use crate::core::{Cost, Message, MessageContent, Role};

/// Mock adapter that produces deterministic responses.
///
/// - If the user message contains "read the file", emits a ToolCall.
/// - Otherwise, streams text chunks followed by a Done event with usage.
/// - Cost is computed from the pricing table (input/output per 1M tokens).
pub struct MockAdapter {
    /// Pricing table: model_name -> (input_per_1m_cents, output_per_1m_cents)
    pricing: HashMap<String, (u64, u64)>,
    /// Capabilities to report
    capabilities: ModelCapabilities,
    /// Counter for generating unique call IDs
    call_counter: AtomicU32,
}

impl MockAdapter {
    /// Create a mock adapter with the given pricing table.
    pub fn new(pricing: HashMap<String, (u64, u64)>) -> Self {
        Self {
            pricing,
            capabilities: ModelCapabilities {
                streaming: true,
                tool_calling: true,
                structured_output: true,
                vision: false,
                max_context_tokens: 128_000,
            },
            call_counter: AtomicU32::new(0),
        }
    }

    /// Create from a full AppConfig (extracts pricing from providers config).
    pub fn from_app_config(config: &crate::config::AppConfig) -> Self {
        let mut pricing = HashMap::new();
        for provider_config in config.providers.values() {
            for (model, price) in &provider_config.pricing {
                pricing.insert(model.clone(), (price.input, price.output));
            }
        }
        Self::new(pricing)
    }

    /// Compute cost from token usage and pricing table.
    fn compute_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> Cost {
        match self.pricing.get(model) {
            Some((input_per_1m, output_per_1m)) => {
                let input_cost = (input_tokens as u64 * input_per_1m) / 1_000_000;
                let output_cost = (output_tokens as u64 * output_per_1m) / 1_000_000;
                Cost(input_cost + output_cost)
            }
            None => Cost(0),
        }
    }

    /// Extract the user's text from messages.
    fn extract_user_text(&self, messages: &[Message]) -> String {
        messages
            .iter()
            .filter(|m| m.role == Role::User)
            .filter_map(|m| match &m.content {
                MessageContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Count approximate input tokens (mock: 1 token per 4 chars).
    fn count_input_tokens(&self, messages: &[Message]) -> u32 {
        let total_chars: usize = messages
            .iter()
            .map(|m| match &m.content {
                MessageContent::Text { text } => text.len(),
                MessageContent::ToolCall { tool_name, .. } => tool_name.len() + 10,
                MessageContent::ToolResult { .. } => 20,
            })
            .sum();
        (total_chars / 4).max(1) as u32
    }

    /// Generate a unique call ID.
    fn next_call_id(&self) -> String {
        format!("call_{}", self.call_counter.fetch_add(1, Ordering::SeqCst))
    }
}

#[async_trait]
impl LLMAdapter for MockAdapter {
    async fn send(
        &self,
        messages: Vec<Message>,
        _config: ModelConfig,
    ) -> Result<Vec<ResponseEvent>, LLMError> {
        let user_text = self.extract_user_text(&messages);
        let input_tokens = self.count_input_tokens(&messages);
        let mut events = Vec::new();

        // If the user asks to "read the file", emit a tool call
        if user_text.to_lowercase().contains("read the file") {
            let call_id = self.next_call_id();
            events.push(ResponseEvent::ToolCall(ToolInvocation {
                call_id,
                tool_name: "filesystem".to_string(),
                arguments: serde_json::json!({"path": "/tmp/example.txt"}),
            }));
        }

        // Stream text chunks
        let response_text = format!("Mock response to: {user_text}");
        let words: Vec<&str> = response_text.split_whitespace().collect();
        let chunk_size = 3.max(words.len() / 4);
        for chunk in words.chunks(chunk_size) {
            events.push(ResponseEvent::Chunk(chunk.join(" ") + " "));
        }

        // Compute output tokens (mock: 1 token per 4 chars of response)
        let output_tokens = (response_text.len() / 4).max(1) as u32;
        let cost = self.compute_cost("gpt-4o-mini", input_tokens, output_tokens);

        events.push(ResponseEvent::Done(Usage {
            input_tokens,
            output_tokens,
            cost,
        }));

        Ok(events)
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.capabilities.clone()
    }
}
