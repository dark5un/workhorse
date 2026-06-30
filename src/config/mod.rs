//! Config: figment-based layered config loading (YAML < env < CLI flags).
//!
//! All thresholds, keywords, model IDs, fallbacks, timeouts, budgets, and
//! defaults come from config or env vars. No compiled-in config values.

use serde::Deserialize;
use std::collections::HashMap;
use thiserror::Error;

/// Top-level application config, loaded from config/ directory.
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub analyzer: AnalyzerConfig,
    pub tools: ToolsConfig,
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
    pub keywords: Vec<String>,
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

#[derive(Debug, Clone, Deserialize)]
pub struct ToolsConfig {
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

#[derive(Debug, Clone, Deserialize)]
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

/// Load config from the config/ directory. Stub -- Phase 1 implements this.
pub fn load_config(_config_dir: &str) -> Result<AppConfig, ConfigError> {
    Err(ConfigError::NotFound(
        "config loader not yet implemented (Phase 1)".to_string(),
    ))
}
