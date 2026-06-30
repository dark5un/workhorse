//! OpenAI-compatible LLM adapter.
//!
//! Works with any API that implements the OpenAI chat completions format:
//! OpenAI, OpenRouter, Ollama (with OpenAI compatibility), vLLM, etc.
//!
//! Behind the `providers` feature flag (requires reqwest).

use async_trait::async_trait;
use std::collections::HashMap;

use crate::adapters::{
    LLMAdapter, LLMError, ModelCapabilities, ModelConfig, ModelInfo, ResponseEvent, ToolInvocation,
    Usage,
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

    /// Check if base_url points to a local server (localhost or 127.0.0.1).
    /// Used to decide whether to try the LM Studio v0 API endpoint.
    fn is_local_url(&self) -> bool {
        self.base_url.contains("localhost") || self.base_url.contains("127.0.0.1")
    }

    /// Discover models via the standard OpenAI-compatible /v1/models endpoint.
    ///
    /// Works for all providers. OpenRouter responses include `context_length`
    /// and `top_provider.max_completion_tokens`; standard OpenAI only returns
    /// model IDs (context info is None).
    async fn discover_openai_compat(&self) -> Result<Vec<ModelInfo>, LLMError> {
        let url = format!("{}/models", self.base_url);

        let mut request = self.client.get(&url).header("Content-Type", "application/json");

        if let Some(api_key) = self.get_api_key() {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request
            .send()
            .await
            .map_err(|e| LLMError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            // Truncate to avoid dumping huge HTML 404 pages in logs
            let snippet = if text.len() > 200 { &text[..200] } else { &text };
            return Err(LLMError::Model(format!("HTTP {status}: {snippet}")));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| LLMError::Parse(e.to_string()))?;

        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| LLMError::Parse("missing 'data' array in models response".into()))?;

        let mut models = Vec::new();
        for entry in data {
            let id = entry
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("unknown")
                .to_string();

            // OpenRouter format: context_length + top_provider.max_completion_tokens
            let max_context = entry
                .get("context_length")
                .and_then(|c| c.as_u64())
                .map(|v| v as usize);

            let max_output = entry
                .pointer("/top_provider/max_completion_tokens")
                .and_then(|m| m.as_u64())
                .map(|v| v as usize);

            // Skip embedding models (they're not chat completion models)
            let is_embedding = entry
                .get("type")
                .and_then(|t| t.as_str())
                .is_some_and(|t| t == "embedding")
                || id.contains("embedding");

            if !is_embedding {
                models.push(ModelInfo {
                    model_id: id,
                    max_context_tokens: max_context,
                    max_output_tokens: max_output,
                });
            }
        }

        Ok(models)
    }

    /// Discover models via LM Studio's proprietary /api/v0/models endpoint.
    ///
    /// This endpoint provides richer metadata than the OpenAI-compatible one:
    /// `max_context_length` (theoretical limit) and `loaded_context_length`
    /// (what's actually allocated in memory). The effective context is the
    /// minimum of the two.
    ///
    /// Only exists on LM Studio. Returns an error for other providers.
    async fn discover_lm_studio_v0(&self) -> Result<Vec<ModelInfo>, LLMError> {
        // Derive the v0 URL from base_url by stripping /v1 and appending /api/v0/models.
        // e.g. http://localhost:1234/v1 -> http://localhost:1234/api/v0/models
        let v0_url = self
            .base_url
            .trim_end_matches("/v1")
            .trim_end_matches('/')
            .to_string()
            + "/api/v0/models";

        let response = self
            .client
            .get(&v0_url)
            .send()
            .await
            .map_err(|e| LLMError::Network(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(LLMError::Model(format!("HTTP {status}: {text}")));
        }

        let json: serde_json::Value =
            response.json().await.map_err(|e| LLMError::Parse(e.to_string()))?;

        let data = json
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| LLMError::Parse("missing 'data' array in v0 models response".into()))?;

        let mut models = Vec::new();
        for entry in data {
            let id = entry
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("unknown")
                .to_string();

            // Skip embedding models
            let model_type = entry.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if model_type == "embedding" || id.contains("embedding") {
                continue;
            }

            let max_context = entry
                .get("max_context_length")
                .and_then(|m| m.as_u64())
                .map(|v| v as usize);

            // loaded_context_length is what LM Studio actually allocated.
            // The effective context is the min of max and loaded.
            let loaded_context = entry
                .get("loaded_context_length")
                .and_then(|l| l.as_u64())
                .map(|v| v as usize);

            let effective_context = match (max_context, loaded_context) {
                (Some(max), Some(loaded)) => Some(max.min(loaded)),
                (Some(max), None) => Some(max),
                (None, Some(loaded)) => Some(loaded),
                (None, None) => None,
            };

            // LM Studio doesn't report max_output_tokens directly.
            // Use the effective context as a reasonable upper bound.
            let max_output = effective_context;

            models.push(ModelInfo {
                model_id: id,
                max_context_tokens: effective_context,
                max_output_tokens: max_output,
            });
        }

        Ok(models)
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
            // Stop sequences: prevent models from generating fake conversation
            // turns. Many chat models (especially reasoning models served by
            // LM Studio/Ollama) will emit role tokens like <|user|> and keep
            // generating indefinitely. These stops cut that off cleanly.
            "stop": [
                "<|user|>",
                "<|im_end|>",
                "<|end|>",
                "<|eot_id|>",
            ],
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

    async fn discover_models(&self) -> Result<Vec<ModelInfo>, LLMError> {
        // Try the OpenAI-compatible /v1/models endpoint first (works for all
        // providers). Then try LM Studio's proprietary /api/v0/models endpoint
        // which has richer metadata. Merge results: v0 takes precedence when
        // available.
        let mut models = self.discover_openai_compat().await?;

        // LM Studio v0 API: try /api/v0/models (only exists on LM Studio).
        // Only attempt for local URLs to avoid hitting foreign servers with
        // a proprietary endpoint path.
        if self.is_local_url() {
            match self.discover_lm_studio_v0().await {
                Ok(v0_models) => {
                    // Merge: v0 results take precedence (they have context info).
                    // Keep OpenAI-compat results for models not in v0.
                    let v0_ids: std::collections::HashSet<String> =
                        v0_models.iter().map(|m| m.model_id.clone()).collect();
                    models.retain(|m| !v0_ids.contains(&m.model_id));
                    models.extend(v0_models);
                }
                Err(LLMError::Network(_)) => {
                    // Not LM Studio or server not running — fine, skip.
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LM Studio v0 discovery failed");
                }
            }
        }

        Ok(models)
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
