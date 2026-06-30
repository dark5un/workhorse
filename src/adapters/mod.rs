//! LLM adapter abstraction: trait + types for provider-agnostic LLM calls.
//!
//! Providers implement `LLMAdapter`. The harness never calls provider-specific
//! SDKs directly. Adapters normalize provider-specific tool-call schemas into
//! the common `ResponseEvent` type.

pub mod mock;
pub mod retry;

#[cfg(feature = "providers")]
pub mod openai_compat;

pub use mock::MockAdapter;
pub use retry::{RetryError, RetryPolicy};

#[cfg(feature = "providers")]
pub use openai_compat::{OpenAiCompatAdapter, build_adapters_from_config};

// Re-export ModelInfo for consumers.
pub use self::ModelInfo as DiscoveredModelInfo;

use async_trait::async_trait;
use std::collections::HashMap;
use thiserror::Error;

use crate::config::AppConfig;
use crate::core::{Cost, Message, ModelId};

/// Abstract LLM adapter. Providers implement this trait.
///
/// Uses `&self` -- adapters manage mutable state (HTTP client pools, etc.)
/// via interior mutability (Arc<Mutex<...>>).
#[async_trait]
pub trait LLMAdapter: Send + Sync {
    /// Send a completion request, returning response events.
    /// The `model` field in ModelConfig specifies which model to call.
    async fn send(
        &self,
        model: &ModelId,
        messages: Vec<Message>,
        config: ModelConfig,
    ) -> Result<Vec<ResponseEvent>, LLMError>;

    fn capabilities(&self) -> ModelCapabilities;

    /// Discover available models and their metadata from the provider's API.
    ///
    /// Returns a list of `ModelInfo` entries. If the provider's API doesn't
    /// expose context window or max output token info, those fields will be
    /// `None` and the caller should fall back to config defaults.
    ///
    /// Default implementation returns an empty vec (no discovery).
    async fn discover_models(&self) -> Result<Vec<ModelInfo>, LLMError> {
        Ok(vec![])
    }
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

/// Discovered model metadata from a provider's API.
///
/// Populated at startup by querying the provider's models endpoint.
/// Used to auto-configure `max_tokens` so users don't have to set it manually.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name as the provider expects it (without provider prefix).
    /// E.g. "zai-org/glm-4.7-flash", "gpt-4o".
    pub model_id: String,
    /// Maximum context window the model supports (input + output combined).
    /// From LM Studio: `max_context_length` (or `loaded_context_length` if smaller).
    /// From OpenRouter: `context_length`.
    pub max_context_tokens: Option<usize>,
    /// Maximum output tokens the model can generate in a single response.
    /// From OpenRouter: `top_provider.max_completion_tokens`.
    /// From LM Studio: inferred from `loaded_context_length`.
    pub max_output_tokens: Option<usize>,
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
    #[error("provider '{0}' not configured")]
    ProviderNotConfigured(String),
}

/// Factory that maps provider names to adapter instances.
///
/// Built from AppConfig. Each provider config has base_url and api_key_env.
/// The factory returns the appropriate adapter for a given ModelId.
pub struct AdapterFactory {
    adapters: HashMap<String, Box<dyn LLMAdapter>>,
}

impl AdapterFactory {
    /// Build an adapter factory from app config.
    /// MockAdapter is used for all providers when no real adapter is available
    /// (i.e., the `providers` feature is not enabled).
    pub fn from_config(config: &AppConfig) -> Self {
        #[cfg(feature = "providers")]
        {
            let adapters = build_adapters_from_config(config);
            Self { adapters }
        }

        #[cfg(not(feature = "providers"))]
        {
            let mut adapters: HashMap<String, Box<dyn LLMAdapter>> = HashMap::new();
            let mock = MockAdapter::from_app_config(config);
            for provider_name in config.providers.keys() {
                adapters.insert(provider_name.clone(), Box::new(mock.clone()));
            }
            adapters.insert("mock".to_string(), Box::new(mock));
            Self { adapters }
        }
    }

    /// Get the adapter for a given model ID's provider.
    pub fn get_adapter(&self, model: &ModelId) -> Result<&dyn LLMAdapter, LLMError> {
        self.adapters
            .get(&model.provider)
            .map(|a| a.as_ref())
            .ok_or_else(|| LLMError::ProviderNotConfigured(model.provider.clone()))
    }

    /// Discover models from all configured providers.
    ///
    /// Queries each adapter's `discover_models()` endpoint. Providers that
    /// are unreachable or don't support discovery are silently skipped
    /// (logged via tracing). Returns a map of `provider/model` -> ModelInfo.
    ///
    /// This is called at startup to auto-configure `max_tokens` based on
    /// what the provider actually supports, so users don't have to set it
    /// manually in config.
    pub async fn discover_all(&self) -> HashMap<String, ModelInfo> {
        let mut discovered = HashMap::new();

        for (provider_name, adapter) in &self.adapters {
            match adapter.discover_models().await {
                Ok(models) => {
                    let count = models.len();
                    for model in models {
                        let key = format!("{provider_name}/{}", model.model_id);
                        tracing::debug!(
                            provider = %provider_name,
                            model = %model.model_id,
                            max_context = ?model.max_context_tokens,
                            max_output = ?model.max_output_tokens,
                            "discovered model"
                        );
                        discovered.insert(key, model);
                    }
                    tracing::info!(
                        provider = %provider_name,
                        count,
                        "discovered models from provider"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %provider_name,
                        error = %e,
                        "model discovery failed, using config defaults"
                    );
                }
            }
        }

        discovered
    }

    /// Get the adapter for a model ID, falling back through the chain.
    ///
    /// Tries the primary model first. If its provider isn't configured,
    /// tries each fallback model in order. Returns the first available
    /// adapter + the model ID that matched.
    ///
    /// This lets users comment out providers in config without breaking
    /// routing — the harness just falls through to the next provider.
    pub fn get_adapter_with_fallback<'a>(
        &'a self,
        primary: &ModelId,
        fallback_chain: &[ModelId],
    ) -> Result<(&'a dyn LLMAdapter, ModelId), LLMError> {
        // Try primary first
        if let Some(adapter) = self.adapters.get(&primary.provider) {
            return Ok((adapter.as_ref(), primary.clone()));
        }

        // Fall through the chain
        for model in fallback_chain {
            if let Some(adapter) = self.adapters.get(&model.provider) {
                tracing::info!(
                    primary = %primary,
                    fallback = %model,
                    "primary provider not configured, using fallback"
                );
                return Ok((adapter.as_ref(), model.clone()));
            }
        }

        Err(LLMError::ProviderNotConfigured(format!(
            "no adapter available for {} or any of its {} fallbacks",
            primary,
            fallback_chain.len()
        )))
    }
}
