//! Prompt complexity analyzer -- two-stage design (heuristic + classifier).

use async_trait::async_trait;
use thiserror::Error;

/// Analyzes prompt complexity to inform routing decisions.
///
/// Two stages (configurable):
/// 1. Heuristic: length + keyword + structural rules from config. No LLM call.
/// 2. Classifier (optional): cheap LLM call returning structured tier + confidence.
///
/// If the classifier fails, falls back to the heuristic result.
#[async_trait]
pub trait PromptAnalyzer: Send + Sync {
    async fn analyze(&self, prompt: &str) -> Result<ComplexityResult, AnalysisError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexityResult {
    pub tier: ComplexityTier,
    pub confidence: f32,
    pub signals: Vec<String>,
    pub source: AnalysisSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComplexityTier {
    Simple,
    Medium,
    Complex,
    Expert,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisSource {
    Heuristic,
    Classifier { model: String },
    FallbackHeuristic { reason: String },
}

#[derive(Debug, Error)]
pub enum AnalysisError {
    #[error("tokenization failed: {0}")]
    Tokenization(String),
    #[error("no tier matched for prompt")]
    NoTierMatched,
    #[error("config error: {0}")]
    Config(String),
    #[error("classifier error: {0}")]
    Classifier(String),
}
