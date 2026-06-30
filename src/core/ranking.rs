//! Model effectiveness ranking engine.
//!
//! Collects explicit user ratings (1-5) for model responses, groups them
//! by complexity tier, and uses Bayesian-smoothed scores to rank models.
//! Rankings are stored in SQLite and can be scoped globally or per-project.

use rusqlite::params;
use std::sync::Mutex;
use thiserror::Error;

use crate::core::{ComplexityTier, ModelId};

/// Errors for ranking operations.
#[derive(Debug, Error)]
pub enum RankingError {
    #[error("storage error: {0}")]
    Storage(String),
    #[error("invalid rating: {0} (must be 1-5)")]
    InvalidRating(u32),
    #[error("no last response to rate")]
    NoLastResponse,
}

/// Configuration for the ranking engine. All values come from config.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RankingConfig {
    pub enabled: bool,
    pub min_samples: u32,
    pub prior: f64,
    pub decay: f64,
    pub exploration_rate: f64,
    pub scope: String,
}

impl Default for RankingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_samples: 3,
            prior: 3.0,
            decay: 0.95,
            exploration_rate: 0.1,
            scope: "project".to_string(),
        }
    }
}

/// Scope for ratings: global or per-project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    Global,
    Project(String),
}

impl Scope {
    /// Create a project scope from the current working directory.
    pub fn from_cwd() -> Self {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        // Use a simple hash of the path to avoid storing full paths
        let hash = fnv1a_hash(&cwd);
        Self::Project(format!("proj_{hash:016x}"))
    }

    fn db_value(&self) -> Option<String> {
        match self {
            Self::Global => None,
            Self::Project(id) => Some(id.clone()),
        }
    }
}

fn fnv1a_hash(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A single model's ranking entry.
#[derive(Debug, Clone)]
pub struct RankingEntry {
    pub model_id: String,
    pub score: f64,
    pub sample_count: u32,
}

/// The ranking engine. Wraps a SQLite connection for persistence.
pub struct RankingEngine {
    conn: Mutex<rusqlite::Connection>,
    config: RankingConfig,
    scope: Scope,
    /// Whether ranking is active for the current session.
    /// Overrides config when the user does /ranking on|off.
    session_enabled: Mutex<Option<bool>>,
}

impl RankingEngine {
    /// Create a new ranking engine with the given config and SQLite connection.
    pub fn new(conn: rusqlite::Connection, config: RankingConfig) -> Self {
        let scope = if config.scope == "global" {
            Scope::Global
        } else {
            Scope::from_cwd()
        };

        // Create the ratings table if it doesn't exist
        let _ = conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS model_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                tier TEXT NOT NULL,
                rating INTEGER NOT NULL,
                cost_cents INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                project_scope TEXT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_ratings_lookup
                ON model_ratings(model_id, tier, project_scope);",
        );

        Self {
            conn: Mutex::new(conn),
            config,
            scope,
            session_enabled: Mutex::new(None),
        }
    }

    /// Check if ranking is currently active (session override or config default).
    pub fn is_enabled(&self) -> bool {
        let guard = self.session_enabled.lock().unwrap();
        guard.unwrap_or(self.config.enabled)
    }

    /// Enable or disable ranking for this session.
    pub fn set_session_enabled(&self, enabled: bool) {
        *self.session_enabled.lock().unwrap() = Some(enabled);
    }

    /// Get the current scope.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// Record a rating for a model response.
    pub fn record_rating(
        &self,
        model_id: &ModelId,
        tier: ComplexityTier,
        rating: u32,
        cost_cents: Option<u64>,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    ) -> Result<(), RankingError> {
        if !(1..=5).contains(&rating) {
            return Err(RankingError::InvalidRating(rating));
        }

        let tier_str = tier_str(tier);
        let scope_val = self.scope.db_value();

        let conn = self
            .conn
            .lock()
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        conn.execute(
            "INSERT INTO model_ratings (model_id, tier, rating, cost_cents, input_tokens, output_tokens, project_scope)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                model_id.as_str(),
                tier_str,
                rating as i64,
                cost_cents.map(|c| c as i64),
                input_tokens.map(|t| t as i64),
                output_tokens.map(|t| t as i64),
                scope_val,
            ],
        )
        .map_err(|e| RankingError::Storage(e.to_string()))?;

        Ok(())
    }

    /// Get the Bayesian-smoothed score for a model+tier.
    /// Returns prior (3.0) if no ratings exist.
    pub fn get_score(&self, model_id: &ModelId, tier: ComplexityTier) -> Result<f64, RankingError> {
        let tier_str = tier_str(tier);
        let scope_val = self.scope.db_value();

        let conn = self
            .conn
            .lock()
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        // Bayesian-smoothed score with recency decay:
        // score = (SUM(rating * decay^days) + prior * prior_weight)
        //       / (SUM(decay^days) + prior_weight)
        // where prior_weight = min_samples (acts as that many prior ratings
        // at the neutral value)
        let prior_weight = self.config.min_samples as f64;
        let prior = self.config.prior;
        let decay = self.config.decay;

        // Fetch raw ratings + timestamps, compute decay in Rust (SQLite lacks pow())
        let mut stmt = conn
            .prepare(
                "SELECT rating, julianday('now') - julianday(timestamp) as days_ago
                 FROM model_ratings
                 WHERE model_id = ? AND tier = ? AND (project_scope IS ? OR project_scope = ?)",
            )
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(
                params![model_id.as_str(), tier_str, &scope_val, &scope_val],
                |row| {
                    let rating: f64 = row.get(0)?;
                    let days_ago: f64 = row.get(1).unwrap_or(0.0);
                    Ok((rating, days_ago))
                },
            )
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut count = 0u32;

        for row in rows {
            let (rating, days_ago) = row.map_err(|e| RankingError::Storage(e.to_string()))?;
            let weight = decay.powf(days_ago);
            weighted_sum += rating * weight;
            weight_sum += weight;
            count += 1;
        }

        if count == 0 {
            return Ok(prior);
        }

        let score = (weighted_sum + prior * prior_weight) / (weight_sum + prior_weight);
        Ok(score)
    }

    /// Get rankings for a tier, sorted by score descending.
    pub fn get_rankings(&self, tier: ComplexityTier) -> Result<Vec<RankingEntry>, RankingError> {
        let tier_str = tier_str(tier);
        let scope_val = self.scope.db_value();
        let prior_weight = self.config.min_samples as f64;
        let prior = self.config.prior;
        let decay = self.config.decay;

        let conn = self
            .conn
            .lock()
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        // Fetch raw data grouped by model_id, compute scores in Rust
        let mut stmt = conn
            .prepare(
                "SELECT model_id, rating, julianday('now') - julianday(timestamp) as days_ago
                 FROM model_ratings
                 WHERE tier = ? AND (project_scope IS ? OR project_scope = ?)",
            )
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![tier_str, &scope_val, &scope_val], |row| {
                let model_id: String = row.get(0)?;
                let rating: f64 = row.get(1)?;
                let days_ago: f64 = row.get(2).unwrap_or(0.0);
                Ok((model_id, rating, days_ago))
            })
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        // Aggregate per model
        use std::collections::HashMap;
        let mut model_data: HashMap<String, (f64, f64, u32)> = HashMap::new();

        for row in rows {
            let (model_id, rating, days_ago) =
                row.map_err(|e| RankingError::Storage(e.to_string()))?;
            let weight = decay.powf(days_ago);
            let entry = model_data.entry(model_id).or_insert((0.0, 0.0, 0));
            entry.0 += rating * weight; // weighted_sum
            entry.1 += weight; // weight_sum
            entry.2 += 1; // count
        }

        let mut results: Vec<RankingEntry> = model_data
            .into_iter()
            .map(
                |(model_id, (weighted_sum, weight_sum, count))| RankingEntry {
                    score: (weighted_sum + prior * prior_weight) / (weight_sum + prior_weight),
                    model_id,
                    sample_count: count,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(results)
    }

    /// Reset ratings for the current scope (or global if scope is Global).
    pub fn reset_ratings(&self, scope: &Scope) -> Result<u64, RankingError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| RankingError::Storage(e.to_string()))?;

        let deleted = match scope {
            Scope::Global => conn
                .execute("DELETE FROM model_ratings", [])
                .map_err(|e| RankingError::Storage(e.to_string()))?,
            Scope::Project(id) => conn
                .execute("DELETE FROM model_ratings WHERE project_scope = ?", [id])
                .map_err(|e| RankingError::Storage(e.to_string()))?,
        };

        Ok(deleted as u64)
    }

    /// Get config reference.
    pub fn config(&self) -> &RankingConfig {
        &self.config
    }

    /// Get minimum samples threshold.
    pub fn min_samples(&self) -> u32 {
        self.config.min_samples
    }

    /// Get exploration rate.
    pub fn exploration_rate(&self) -> f64 {
        self.config.exploration_rate
    }
}

fn tier_str(tier: ComplexityTier) -> &'static str {
    match tier {
        ComplexityTier::Simple => "simple",
        ComplexityTier::Medium => "medium",
        ComplexityTier::Complex => "complex",
        ComplexityTier::Expert => "expert",
    }
}
