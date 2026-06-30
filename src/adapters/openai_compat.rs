//! OpenAI-compatible LLM adapter.
//!
//! Works with any API that implements the OpenAI chat completions format:
//! OpenAI, OpenRouter, Ollama (with OpenAI compatibility), vLLM, etc.
//!
//! Behind the `providers` feature flag (requires reqwest).

use async_trait::async_trait;
use std::collections::HashMap;

use crate::adapters::{
    LLMAdapter, LLMError, ModelCapabilities, ModelConfig, ResponseEvent, ToolInvocation, Usage,
};
use crate::core::{Cost, Message, MessageContent, ModelId, Role};

/// OpenAI-compatible adapter. Uses reqwest for HTTP calls.
///
/// The base_url, api_key_env, and pricing come from config.
/// One adapter instance per provider (OpenAI, OpenRouter, LM Studio, etc.).
///
/// `base_url` should include the API version prefix (e.g.
/// `https://api.openai.com/v1` or `http://localhost:1234/v1`). The adapter
/// appends `/chat/completions` — it does NOT add another `/v1`.
pub struct OpenAiCompatAdapter {
    base_url: String,
    api_key_env: Option<String>,
    pricing: HashMap<String, (u64, u64)>,
    client: reqwest::Client,
}

impl OpenAiCompatAdapter {
    /// Create a new adapter for a provider.
    ///
    /// Pass `None` for `api_key_env` on providers that don't require auth
    /// (e.g. a local LM Studio server).
    pub fn new(
        base_url: String,
        api_key_env: Option<String>,
        pricing: HashMap<String, (u64, u64)>,
    ) -> Self {
        Self {
            base_url,
            api_key_env,
            pricing,
            client: reqwest::Client::new(),
        }
    }

    /// Create from provider config.
    pub fn from_provider_config(
        base_url: &str,
        api_key_env: Option<&str>,
        pricing: &HashMap<String, crate::config::PricingConfig>,
    ) -> Self {
        let pricing_map: HashMap<String, (u64, u64)> = pricing
            .iter()
            .map(|(k, v)| (k.clone(), (v.input, v.output)))
            .collect();
        Self::new(
            base_url.to_string(),
            api_key_env.map(|s| s.to_string()),
            pricing_map,
        )
    }

    /// Get the API key from the environment, if one is configured.
    /// Returns None when no env var is configured or it isn't set — the
    /// request is then sent without an Authorization header, which is correct
    /// for local providers like LM Studio that don't require auth.
    fn get_api_key(&self) -> Option<String> {
        let env_name = self.api_key_env.as_ref()?;
        std::env::var(env_name).ok()
    }

    /// Compute cost from token usage and pricing table.
    fn compute_cost(&self, model: &str, input_tokens: u32, output_tokens: u32) -> Cost {
        match self.pricing.get(model) {
            Some((input_per_1m, output_per_1m)) => {
                let input_cost = (input_tokens as f64 * *input_per_1m as f64) / 1_000_000.0;
                let output_cost = (output_tokens as f64 * *output_per_1m as f64) / 1_000_000.0;
                let total = input_cost + output_cost;
                Cost(if total > 0.0 { total.ceil() as u64 } else { 0 })
            }
            None => Cost(0),
        }
    }

    /// Convert messages to OpenAI format.
    fn convert_messages(messages: &[Message]) -> Vec<serde_json::Value> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                let content = match &m.content {
                    MessageContent::Text { text } => text.clone(),
                    MessageContent::ToolCall {
                        tool_name,
                        arguments,
                        ..
                    } => serde_json::json!({ "tool_name": tool_name, "arguments": arguments })
                        .to_string(),
                    MessageContent::ToolResult { result, .. } => result.to_string(),
                };
                serde_json::json!({ "role": role, "content": content })
            })
            .collect()
    }

    /// Parse the model name from a ModelId (strip provider prefix).
    /// For OpenRouter, the model field is the full model path (e.g. "anthropic/claude-3.5-sonnet").
    /// For OpenAI, it's just the model name (e.g. "gpt-4o").
    fn model_name(model: &ModelId) -> &str {
        &model.model
    }
}

#[async_trait]
impl LLMAdapter for OpenAiCompatAdapter {
    async fn send(
        &self,
        model: &ModelId,
        messages: Vec<Message>,
        config: ModelConfig,
    ) -> Result<Vec<ResponseEvent>, LLMError> {
        let model_name = Self::model_name(model);
        let url = format!("{}/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": model_name,
            "messages": Self::convert_messages(&messages),
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": false,  // TODO: streaming via SSE
        });

        if let Some(format) = config.response_format {
            if format == crate::adapters::ResponseFormat::Json {
                body["response_format"] = serde_json::json!({"type": "json_object"});
            }
        }

        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);

        // Attach Authorization header only when an API key is available.
        // Local providers (LM Studio, Ollama) don't require auth.
        if let Some(api_key) = self.get_api_key() {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request
            .send()
            .await
            .map_err(|e| LLMError::Network(e.to_string()))?;

        if response.status() == 401 {
            return Err(LLMError::Auth("authentication failed".to_string()));
        }

        if response.status() == 429 {
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(1000);
            return Err(LLMError::RateLimited {
                retry_after_ms: retry_after,
            });
        }

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(LLMError::Model(format!("HTTP {status}: {text}")));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LLMError::Parse(e.to_string()))?;

        let mut events = Vec::new();

        // Extract choices
        if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
            for choice in choices {
                if let Some(content) = choice
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                {
                    events.push(ResponseEvent::Chunk(content.to_string()));
                }

                // Check for tool calls in the response
                if let Some(tool_calls) = choice
                    .get("message")
                    .and_then(|m| m.get("tool_calls"))
                    .and_then(|t| t.as_array())
                {
                    for tc in tool_calls {
                        let call_id = tc
                            .get("id")
                            .and_then(|i| i.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let tool_name = tc
                            .pointer("/function/name")
                            .and_then(|n| n.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let arguments = tc
                            .pointer("/function/arguments")
                            .and_then(|a| a.as_str())
                            .and_then(|s| serde_json::from_str(s).ok())
                            .unwrap_or(serde_json::json!({}));

                        events.push(ResponseEvent::ToolCall(ToolInvocation {
                            call_id,
                            tool_name,
                            arguments,
                        }));
                    }
                }
            }
        }

        // Extract usage
        let input_tokens = json
            .pointer("/usage/prompt_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;
        let output_tokens = json
            .pointer("/usage/completion_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32;
        let cost = self.compute_cost(model_name, input_tokens, output_tokens);

        events.push(ResponseEvent::Done(Usage {
            input_tokens,
            output_tokens,
            cost,
        }));

        Ok(events)
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            streaming: true,
            tool_calling: true,
            structured_output: true,
            vision: false,
            max_context_tokens: 128_000,
        }
    }
}

/// Build adapters from config. Returns a map of provider name -> adapter.
/// Only called when the `providers` feature is enabled.
pub fn build_adapters_from_config(
    config: &crate::config::AppConfig,
) -> HashMap<String, Box<dyn LLMAdapter>> {
    let mut adapters: HashMap<String, Box<dyn LLMAdapter>> = HashMap::new();

    for (provider_name, provider_config) in &config.providers {
        let adapter = OpenAiCompatAdapter::from_provider_config(
            &provider_config.base_url,
            provider_config.api_key_env.as_deref(),
            &provider_config.pricing,
        );
        adapters.insert(provider_name.clone(), Box::new(adapter));
    }

    // Always have a "mock" provider for testing
    let mock = crate::adapters::MockAdapter::from_app_config(config);
    adapters.insert("mock".to_string(), Box::new(mock));

    adapters
}
