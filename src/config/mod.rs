//! Config: figment-based layered config loading (YAML < env < CLI flags).
//!
//! All thresholds, keywords, model IDs, fallbacks, timeouts, budgets, and
//! defaults come from config or env vars. No compiled-in config values.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

use crate::core::ModelId;
use figment::providers::Format;

/// Top-level application config, loaded from config/ directory.
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub analyzer: AnalyzerConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    pub session: SessionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzerConfig {
    pub heuristic: HeuristicConfig,
    pub classifier: Option<ClassifierConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeuristicConfig {
    pub enabled: bool,
    pub tiers: HashMap<String, TierConfig>,
    pub fallback_policy: String,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TierConfig {
    pub thresholds: ThresholdConfig,
    #[serde(default)]
    pub keywords: Vec<String>,
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ThresholdConfig {
    pub min_tokens: u64,
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClassifierConfig {
    pub enabled: bool,
    pub model: String,
    pub fallback_on_error: bool,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ToolsConfig {
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    #[serde(default)]
    pub defaults: ToolsDefaults,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub transport: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default = "default_sandbox")]
    pub sandbox: String,
}

fn default_sandbox() -> String {
    "consent".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolsDefaults {
    #[serde(default = "default_sandbox")]
    pub sandbox: String,
    #[serde(default = "default_temp_dir")]
    pub session_temp_dir: String,
}

fn default_temp_dir() -> String {
    "/tmp/harness-${SESSION_ID}".to_string()
}

impl Default for ToolsDefaults {
    fn default() -> Self {
        Self {
            sandbox: default_sandbox(),
            session_temp_dir: default_temp_dir(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key_env: String,
    #[serde(default)]
    pub pricing: HashMap<String, PricingConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PricingConfig {
    pub input: u64,
    pub output: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionConfig {
    pub storage: String,
    pub path: String,
    pub context_window: ContextWindowConfig,
    pub system_prompt_file: String,
    pub cost_tracking: CostTrackingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextWindowConfig {
    pub strategy: String,
    pub max_tokens: usize,
    pub sticky_system_prompt: bool,
    #[serde(default)]
    pub summarize: Option<SummarizeConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SummarizeConfig {
    pub model: String,
    pub trigger_at_pct: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CostTrackingConfig {
    pub enabled: bool,
    pub warn_at_usd: f64,
    pub hard_limit_usd: f64,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config file not found: {0}")]
    NotFound(String),
    #[error("parse error at {path}: {message}")]
    Parse { path: String, message: String },
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid value for {field}: {reason}")]
    InvalidValue { field: String, reason: String },
}

/// Config file names within the config directory.
const CONFIG_FILES: &[&str] = &[
    "routing.yaml",
    "tools.yaml",
    "providers.yaml",
    "session.yaml",
];

/// Load config from a directory using figment (YAML files < env vars).
///
/// Layered config: YAML files provide defaults, env vars (prefixed with
/// `HARNESS_` and split on `__`) override.
///
/// # Errors
/// - `NotFound`: config directory or required files missing
/// - `Parse`: figment extraction or deserialization failure
/// - `MissingField` / `InvalidValue`: validation failure
pub fn load_config(config_dir: &str) -> Result<AppConfig, ConfigError> {
    let dir = Path::new(config_dir);
    if !dir.is_dir() {
        return Err(ConfigError::NotFound(format!(
            "config directory not found: {config_dir}"
        )));
    }

    let mut figment = figment::Figment::new();

    for file_name in CONFIG_FILES {
        let path = dir.join(file_name);
        if !path.exists() {
            return Err(ConfigError::NotFound(format!(
                "config file not found: {}",
                path.display()
            )));
        }
        figment = figment.merge(figment::providers::Yaml::file(path));
    }

    // Env var overrides: HARNESS_SESSION__STORAGE=json -> session.storage = "json"
    figment = figment.merge(figment::providers::Env::prefixed("HARNESS_").split("__"));

    let config: AppConfig = figment.extract().map_err(|e| {
        let path = e
            .metadata
            .as_ref()
            .map(|m| m.name.to_string())
            .unwrap_or_default();
        ConfigError::Parse {
            path,
            message: e.to_string(),
        }
    })?;

    config.validate()?;
    Ok(config)
}

impl AppConfig {
    /// Validate config: check required fields, model ID formats, etc.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Tiers must be configured
        if self.analyzer.heuristic.tiers.is_empty() {
            return Err(ConfigError::MissingField(
                "analyzer.heuristic.tiers".to_string(),
            ));
        }

        // Validate model IDs in each tier
        for (tier_name, tier_config) in &self.analyzer.heuristic.tiers {
            for model_str in &tier_config.models {
                if ModelId::parse(model_str).is_none() {
                    return Err(ConfigError::InvalidValue {
                        field: format!("analyzer.heuristic.tiers.{tier_name}.models"),
                        reason: format!(
                            "invalid model ID '{model_str}' (expected 'provider/model')"
                        ),
                    });
                }
            }
        }

        // Validate fallback_policy
        if self.analyzer.heuristic.fallback_policy != "sequential" {
            return Err(ConfigError::InvalidValue {
                field: "analyzer.heuristic.fallback_policy".to_string(),
                reason: format!(
                    "unsupported fallback policy '{}' (only 'sequential' is supported)",
                    self.analyzer.heuristic.fallback_policy
                ),
            });
        }

        Ok(())
    }
}
