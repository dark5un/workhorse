//! Learning router: wraps ConfigRouter and reorders the fallback chain
//! based on historical model performance from the RankingEngine.
//!
//! When ranking is enabled:
//! 1. ConfigRouter produces ModelSpec (config order)
//! 2. LearningRouter reorders the primary + fallback chain by score
//! 3. Models with < min_samples ratings keep their config position
//! 4. Epsilon-greedy exploration: with probability exploration_rate,
//!    pick a random under-sampled model from the chain

use async_trait::async_trait;

use super::ranking::RankingEngine;
use super::{ComplexityResult, ComplexityTier, Cost, ModelId, ModelSpec, RoutingError};
use crate::config::HeuristicConfig;

/// Router that wraps ConfigRouter with a learning layer.
pub struct LearningRouter {
    /// Inner config-driven router.
    inner: super::router::ConfigRouter,
    /// Ranking engine for model effectiveness data (shared with Session).
    ranking: std::sync::Arc<RankingEngine>,
}

impl LearningRouter {
    /// Create a learning router from config and a ranking engine.
    pub fn new(
        config: &HeuristicConfig,
        budget_limit: Option<Cost>,
        ranking: std::sync::Arc<RankingEngine>,
    ) -> Result<Self, RoutingError> {
        let inner = super::router::ConfigRouter::new(config, budget_limit)?;
        Ok(Self { inner, ranking })
    }

    /// Create from a full AppConfig (convenience).
    pub fn from_app_config(
        config: &crate::config::AppConfig,
        ranking: std::sync::Arc<RankingEngine>,
    ) -> Result<Self, RoutingError> {
        let budget_limit = if config.session.cost_tracking.enabled {
            Some(Cost::from_usd(config.session.cost_tracking.hard_limit_usd))
        } else {
            None
        };
        Self::new(&config.analyzer.heuristic, budget_limit, ranking)
    }

    /// Reorder a model list by ranking score (descending).
    /// Models with insufficient samples stay in their original relative order
    /// at the end, while models with enough samples are sorted by score.
    fn reorder_by_score(
        models: &[ModelId],
        tier: ComplexityTier,
        task_type: super::TaskType,
        ranking: &RankingEngine,
    ) -> Vec<ModelId> {
        let min_samples = ranking.min_samples();

        let mut scored: Vec<(ModelId, f64, u32)> = Vec::new();
        let mut unscored: Vec<ModelId> = Vec::new();

        for model in models {
            match ranking.get_score(model, tier, task_type) {
                Ok(score) => {
                    let has_enough = ranking
                        .get_rankings(tier, task_type)
                        .map(|entries| {
                            entries
                                .iter()
                                .find(|e| e.model_id == model.as_str())
                                .is_some_and(|e| e.sample_count >= min_samples)
                        })
                        .unwrap_or(false);

                    if has_enough {
                        scored.push((model.clone(), score, 0));
                    } else {
                        unscored.push(model.clone());
                    }
                }
                Err(_) => {
                    unscored.push(model.clone());
                }
            }
        }

        // Sort scored models by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Scored models first (sorted), then unscored (original order)
        scored
            .into_iter()
            .map(|(m, _, _)| m)
            .chain(unscored)
            .collect()
    }

    /// Epsilon-greedy exploration: with probability exploration_rate,
    /// pick a random model from the chain that has < min_samples ratings.
    fn maybe_explore(
        models: &[ModelId],
        tier: ComplexityTier,
        task_type: super::TaskType,
        ranking: &RankingEngine,
    ) -> Option<ModelId> {
        let rate = ranking.exploration_rate();
        if rate <= 0.0 || models.len() <= 1 {
            return None;
        }

        // Roll the dice
        let rand_val = simple_random_f64();
        if rand_val >= rate {
            return None;
        }

        // Find models with insufficient samples
        let min_samples = ranking.min_samples();
        let under_sampled: Vec<&ModelId> = models
            .iter()
            .filter(|m| {
                ranking
                    .get_rankings(tier, task_type)
                    .map(|entries| {
                        !entries
                            .iter()
                            .any(|e| e.model_id == m.as_str() && e.sample_count >= min_samples)
                    })
                    .unwrap_or(true)
            })
            .collect();

        if under_sampled.is_empty() {
            return None;
        }

        // Pick a random under-sampled model
        let idx = (simple_random_f64() * under_sampled.len() as f64) as usize;
        let idx = idx.min(under_sampled.len() - 1);
        Some(under_sampled[idx].clone())
    }
}

#[async_trait]
impl super::Router for LearningRouter {
    async fn route(
        &self,
        complexity: &ComplexityResult,
        user_override: Option<&ModelId>,
    ) -> Result<ModelSpec, RoutingError> {
        // First, get the config-driven route
        let spec = self.inner.route(complexity, user_override).await?;

        // If ranking is disabled, return the config-driven spec as-is
        if !self.ranking.is_enabled() {
            return Ok(spec);
        }

        // Reorder the primary + fallback chain by score
        let all_models = std::iter::once(spec.model_id.clone())
            .chain(spec.fallback_chain.iter().cloned())
            .collect::<Vec<_>>();

        let reordered = Self::reorder_by_score(
            &all_models,
            complexity.tier,
            complexity.task_type,
            &self.ranking,
        );

        if reordered.is_empty() {
            return Ok(spec);
        }

        // Check for exploration
        if let Some(explore_model) = Self::maybe_explore(
            &reordered,
            complexity.tier,
            complexity.task_type,
            &self.ranking,
        ) {
            tracing::info!(
                model = %explore_model.as_str(),
                tier = ?complexity.tier,
                "exploration: selecting under-sampled model"
            );
            // Move the explored model to primary, rest become fallback
            let fallback: Vec<ModelId> = reordered
                .into_iter()
                .filter(|m| m != &explore_model)
                .collect();
            return Ok(ModelSpec {
                model_id: explore_model,
                fallback_chain: fallback,
                budget_limit: spec.budget_limit,
            });
        }

        let model_id = reordered[0].clone();
        let fallback_chain = reordered[1..].to_vec();

        tracing::info!(
            primary = %model_id.as_str(),
            fallback_count = fallback_chain.len(),
            tier = ?complexity.tier,
            "learning router reordered fallback chain"
        );

        Ok(ModelSpec {
            model_id,
            fallback_chain,
            budget_limit: spec.budget_limit,
        })
    }
}

/// Simple pseudo-random f64 in [0, 1) without adding a dependency.
/// Uses thread-local state seeded from system time.
fn simple_random_f64() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new({
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(42);
            seed.max(1)
        });
    }

    STATE.with(|s| {
        let mut x = s.get();
        // xorshift64
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f64) / (u64::MAX as f64)
    })
}
