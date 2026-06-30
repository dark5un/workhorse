//! Router: maps complexity results to model specs via config-driven tier mappings.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

use super::{ComplexityResult, ComplexityTier, Cost};
use crate::config::HeuristicConfig;

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

// ============================================================
// Config-driven Router Implementation
// ============================================================

/// Config-driven router that maps ComplexityTier to ModelSpec using
/// tier→model mappings from config. All model IDs and fallback chains
/// come from config. The router does not pre-validate provider availability.
pub struct ConfigRouter {
    /// Map from ComplexityTier to ordered list of ModelIds.
    /// First model is primary; rest form the fallback chain.
    tier_models: HashMap<ComplexityTier, Vec<ModelId>>,
    /// Budget limit from session config (applies to all routes).
    budget_limit: Option<Cost>,
}

impl ConfigRouter {
    /// Create a router from heuristic config (tier→model mappings).
    pub fn new(config: &HeuristicConfig, budget_limit: Option<Cost>) -> Result<Self, RoutingError> {
        let mut tier_models = HashMap::new();

        for (key, tier_config) in &config.tiers {
            let tier = ComplexityTier::from_config_key(key)
                .ok_or_else(|| RoutingError::Config(format!("unknown tier key: {key}")))?;

            let models: Vec<ModelId> = tier_config
                .models
                .iter()
                .filter_map(|s| ModelId::parse(s))
                .collect();

            if models.is_empty() {
                return Err(RoutingError::NoModelsForTier);
            }

            tier_models.insert(tier, models);
        }

        Ok(Self {
            tier_models,
            budget_limit,
        })
    }

    /// Create from a full AppConfig (convenience for tests).
    pub fn from_app_config(config: &crate::config::AppConfig) -> Result<Self, RoutingError> {
        let budget_limit = if config.session.cost_tracking.enabled {
            Some(Cost::from_usd(config.session.cost_tracking.hard_limit_usd))
        } else {
            None
        };
        Self::new(&config.analyzer.heuristic, budget_limit)
    }
}

#[async_trait]
impl Router for ConfigRouter {
    async fn route(
        &self,
        complexity: &ComplexityResult,
        user_override: Option<&ModelId>,
    ) -> Result<ModelSpec, RoutingError> {
        // User override bypasses routing entirely
        if let Some(model_id) = user_override {
            return Ok(ModelSpec {
                model_id: model_id.clone(),
                fallback_chain: vec![],
                budget_limit: self.budget_limit,
            });
        }

        // Look up models for the complexity tier
        let models = self
            .tier_models
            .get(&complexity.tier)
            .ok_or(RoutingError::NoModelsForTier)?;

        if models.is_empty() {
            return Err(RoutingError::NoModelsForTier);
        }

        // First model is primary; rest form the fallback chain
        let model_id = models[0].clone();
        let fallback_chain = models[1..].to_vec();

        Ok(ModelSpec {
            model_id,
            fallback_chain,
            budget_limit: self.budget_limit,
        })
    }
}
