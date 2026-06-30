//! Router: maps complexity results to model specs via config-driven tier mappings.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{ComplexityResult, Cost};

/// Routes a complexity result to a model specification.
///
/// The router does NOT pre-validate provider availability. It returns the
/// preferred model + fallback chain; the adapter attempts the primary, and
/// on failure falls through the chain.
#[async_trait]
pub trait Router: Send + Sync {
    async fn route(
        &self,
        complexity: &ComplexityResult,
        user_override: Option<&ModelId>,
    ) -> Result<ModelSpec, RoutingError>;
}

/// Canonical model identifier: provider + model name.
/// Parsed and validated at config load time. No bare strings in routing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId {
    pub provider: String,
    pub model: String,
}

impl ModelId {
    /// Parse "provider/model" format. Returns None if malformed.
    pub fn parse(s: &str) -> Option<Self> {
        let (provider, model) = s.split_once('/')?;
        let provider = provider.trim();
        let model = model.trim();
        if provider.is_empty() || model.is_empty() {
            return None;
        }
        Some(Self {
            provider: provider.to_string(),
            model: model.to_string(),
        })
    }

    /// Render as "provider/model".
    pub fn as_str(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.provider, self.model)
    }
}

impl std::str::FromStr for ModelId {
    type Err = RoutingError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or(RoutingError::InvalidModelId(s.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub model_id: ModelId,
    pub fallback_chain: Vec<ModelId>,
    pub budget_limit: Option<Cost>,
}

#[derive(Debug, Error)]
pub enum RoutingError {
    #[error("invalid model id: {0}")]
    InvalidModelId(String),
    #[error("no models configured for tier")]
    NoModelsForTier,
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("config error: {0}")]
    Config(String),
}
