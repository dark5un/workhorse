//! Prompt complexity analyzer -- two-stage design (heuristic + classifier).
//!
//! Phase 1 implements the heuristic stage. The classifier stage (Phase 5)
//! will add an optional LLM-based classification step.

use async_trait::async_trait;
use std::collections::HashMap;
use thiserror::Error;

use crate::config::{AnalyzerConfig, HeuristicConfig};

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

impl ComplexityTier {
    /// Parse a tier from a config key string (e.g. "simple", "medium").
    pub fn from_config_key(key: &str) -> Option<Self> {
        match key.to_lowercase().as_str() {
            "simple" => Some(Self::Simple),
            "medium" => Some(Self::Medium),
            "complex" => Some(Self::Complex),
            "expert" => Some(Self::Expert),
            _ => None,
        }
    }

    /// Numeric rank for ordering: Simple=0 ... Expert=3.
    pub fn rank(&self) -> u8 {
        match self {
            Self::Simple => 0,
            Self::Medium => 1,
            Self::Complex => 2,
            Self::Expert => 3,
        }
    }

    /// All tiers in ascending order.
    pub fn all_ascending() -> [Self; 4] {
        [Self::Simple, Self::Medium, Self::Complex, Self::Expert]
    }

    /// All tiers in descending order.
    pub fn all_descending() -> [Self; 4] {
        [Self::Expert, Self::Complex, Self::Medium, Self::Simple]
    }
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

// ============================================================
// Heuristic Analyzer Implementation
// ============================================================

/// Heuristic-based prompt complexity analyzer.
///
/// Uses tiktoken (cl100k_base) for token counting. All thresholds, keywords,
/// and model IDs come from config. The matching algorithm (case-insensitive
/// substring, structural detection) is compiled logic.
pub struct HeuristicAnalyzer {
    config: HeuristicConfig,
    bpe: tiktoken_rs::CoreBPE,
}

impl HeuristicAnalyzer {
    /// Create a new heuristic analyzer from analyzer config.
    pub fn new(config: AnalyzerConfig) -> Result<Self, AnalysisError> {
        let bpe =
            tiktoken_rs::cl100k_base().map_err(|e| AnalysisError::Tokenization(e.to_string()))?;
        Ok(Self {
            config: config.heuristic,
            bpe,
        })
    }

    /// Create from a full AppConfig (convenience for tests).
    pub fn from_app_config(config: &crate::config::AppConfig) -> Result<Self, AnalysisError> {
        Self::new(config.analyzer.clone())
    }

    /// Count tokens using tiktoken cl100k_base (reference tokenizer).
    fn count_tokens(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    /// Detect structural signals in the prompt.
    fn detect_structural_signals(&self, prompt: &str) -> Vec<StructuralSignal> {
        let mut signals = Vec::new();

        // Code blocks (```)
        if prompt.contains("```") {
            signals.push(StructuralSignal::CodeBlock);
        }

        // JSON detection (starts with { or [)
        let trimmed = prompt.trim_start();
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            signals.push(StructuralSignal::Json);
        }

        // Multi-step instructions (3+ lines starting with a number)
        let numbered_lines = prompt
            .lines()
            .filter(|l| {
                l.trim_start()
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_digit())
            })
            .count();
        if numbered_lines >= 3 {
            signals.push(StructuralSignal::MultiStep);
        }

        signals
    }

    /// Evaluate a prompt against config tiers and return the result.
    fn evaluate(&self, prompt: &str) -> Result<ComplexityResult, AnalysisError> {
        let token_count = self.count_tokens(prompt);
        let prompt_lower = prompt.to_lowercase();
        let structural_signals = self.detect_structural_signals(prompt);

        // Collect signals per tier
        let mut tier_scores: HashMap<ComplexityTier, Vec<String>> = HashMap::new();

        for (key, tier_config) in &self.config.tiers {
            let tier = ComplexityTier::from_config_key(key)
                .ok_or_else(|| AnalysisError::Config(format!("unknown tier key: {key}")))?;

            let mut signals = Vec::new();

            // Length check
            let min = tier_config.thresholds.min_tokens;
            let max = tier_config.thresholds.max_tokens.unwrap_or(u64::MAX);
            if token_count as u64 >= min && token_count as u64 <= max {
                signals.push(format!("length:{token_count}tokens"));
            }

            // Keyword matching (case-insensitive substring)
            for kw in &tier_config.keywords {
                if prompt_lower.contains(&kw.to_lowercase()) {
                    signals.push(format!("keyword:{kw}"));
                }
            }

            // Structural detection
            for sig in &structural_signals {
                let signal_str = match sig {
                    StructuralSignal::CodeBlock => {
                        if tier == ComplexityTier::Complex || tier == ComplexityTier::Expert {
                            Some("structural:code_block")
                        } else {
                            None
                        }
                    }
                    StructuralSignal::Json => {
                        if tier == ComplexityTier::Complex {
                            Some("structural:json")
                        } else {
                            None
                        }
                    }
                    StructuralSignal::MultiStep => {
                        if tier == ComplexityTier::Expert {
                            Some("structural:multi_step")
                        } else {
                            None
                        }
                    }
                };
                if let Some(s) = signal_str {
                    signals.push(s.to_string());
                }
            }

            if !signals.is_empty() {
                tier_scores.insert(tier, signals);
            }
        }

        // Determine winning tier
        let has_keyword_match = tier_scores
            .values()
            .flatten()
            .any(|s| s.starts_with("keyword:"));

        if has_keyword_match {
            // Keyword-driven: pick the highest-scoring tier with keyword matches.
            // Tie -> higher tier wins.
            let mut best_tier = ComplexityTier::Simple;
            let mut best_score = 0usize;

            for tier in ComplexityTier::all_descending() {
                if let Some(sigs) = tier_scores.get(&tier) {
                    let kw_count = sigs.iter().filter(|s| s.starts_with("keyword:")).count();
                    if kw_count > 0 {
                        let total = sigs.len();
                        if total > best_score {
                            best_score = total;
                            best_tier = tier;
                        }
                    }
                }
            }

            let final_signals = tier_scores.get(&best_tier).cloned().unwrap_or_default();
            let confidence = if best_score >= 2 { 0.9 } else { 0.7 };

            Ok(ComplexityResult {
                tier: best_tier,
                confidence,
                signals: final_signals,
                source: AnalysisSource::Heuristic,
            })
        } else {
            // No keyword match: use length-based classification.
            // If length points to Simple, bump to Medium (safe default --
            // can't confidently classify as "simple" without keyword signal).
            let length_tier = ComplexityTier::all_ascending()
                .iter()
                .filter(|t| {
                    tier_scores
                        .get(t)
                        .is_some_and(|sigs| sigs.iter().any(|s| s.starts_with("length:")))
                })
                .copied()
                .max_by_key(|t| t.rank())
                .unwrap_or(ComplexityTier::Medium);

            let bumped = if length_tier == ComplexityTier::Simple {
                ComplexityTier::Medium
            } else {
                length_tier
            };

            let mut final_signals = tier_scores
                .get(&length_tier)
                .cloned()
                .unwrap_or_else(|| vec![format!("length:{token_count}tokens")]);

            // Add structural signals to the output even if they don't match a tier
            for sig in &structural_signals {
                let s = match sig {
                    StructuralSignal::CodeBlock => "structural:code_block",
                    StructuralSignal::Json => "structural:json",
                    StructuralSignal::MultiStep => "structural:multi_step",
                };
                if !final_signals.iter().any(|f| f == s) {
                    final_signals.push(s.to_string());
                }
            }

            Ok(ComplexityResult {
                tier: bumped,
                confidence: 0.4,
                signals: final_signals,
                source: AnalysisSource::Heuristic,
            })
        }
    }
}

#[async_trait]
impl PromptAnalyzer for HeuristicAnalyzer {
    async fn analyze(&self, prompt: &str) -> Result<ComplexityResult, AnalysisError> {
        self.evaluate(prompt)
    }
}

/// Internal structural signal type.
enum StructuralSignal {
    CodeBlock,
    Json,
    MultiStep,
}

// ============================================================
// Classifier Model Trait + Classification Response
// ============================================================

/// Response from a classifier model.
#[derive(Debug, Clone)]
pub struct ClassificationResponse {
    pub tier: ComplexityTier,
    pub confidence: f32,
    pub reasoning: String,
}

/// Trait for models that classify prompt complexity via an LLM call.
///
/// Real implementations use an LLMAdapter with structured output (JSON mode).
/// Test implementations return deterministic responses.
#[async_trait]
pub trait ClassifierModel: Send + Sync {
    async fn classify(&self, prompt: &str) -> Result<ClassificationResponse, AnalysisError>;
}

// ============================================================
// Classifier Analyzer (two-stage: heuristic + classifier)
// ============================================================

/// Two-stage analyzer: runs the heuristic first, then optionally calls
/// a classifier model to override the result.
///
/// If the classifier succeeds, its result is used with `AnalysisSource::Classifier`.
/// If the classifier fails and `fallback_on_error` is true, the heuristic
/// result is used with `AnalysisSource::FallbackHeuristic`.
/// If `fallback_on_error` is false, the classifier error propagates.
pub struct ClassifierAnalyzer {
    heuristic: HeuristicAnalyzer,
    model: Box<dyn ClassifierModel>,
    model_name: String,
    fallback_on_error: bool,
}

impl ClassifierAnalyzer {
    pub fn new(
        heuristic: HeuristicAnalyzer,
        model: Box<dyn ClassifierModel>,
        model_name: String,
        fallback_on_error: bool,
    ) -> Self {
        Self {
            heuristic,
            model,
            model_name,
            fallback_on_error,
        }
    }

    /// Create from app config. Uses the classifier config section.
    /// Returns None if classifier is not enabled in config.
    pub fn from_app_config(
        config: &crate::config::AppConfig,
        model: Box<dyn ClassifierModel>,
    ) -> Option<Self> {
        let classifier_config = config.analyzer.classifier.as_ref()?;
        if !classifier_config.enabled {
            return None;
        }
        let heuristic = HeuristicAnalyzer::from_app_config(config).ok()?;
        Some(Self::new(
            heuristic,
            model,
            classifier_config.model.clone(),
            classifier_config.fallback_on_error,
        ))
    }
}

#[async_trait]
impl PromptAnalyzer for ClassifierAnalyzer {
    async fn analyze(&self, prompt: &str) -> Result<ComplexityResult, AnalysisError> {
        // Run heuristic first (always available)
        let heuristic_result = self.heuristic.analyze(prompt).await?;

        // Try classifier
        match self.model.classify(prompt).await {
            Ok(response) => {
                // Classifier succeeds: override heuristic result
                Ok(ComplexityResult {
                    tier: response.tier,
                    confidence: response.confidence,
                    signals: {
                        let mut sigs = heuristic_result.signals.clone();
                        sigs.push(format!("classifier_reasoning:{}", response.reasoning));
                        sigs
                    },
                    source: AnalysisSource::Classifier {
                        model: self.model_name.clone(),
                    },
                })
            }
            Err(e) => {
                // Classifier fails
                if self.fallback_on_error {
                    Ok(ComplexityResult {
                        tier: heuristic_result.tier,
                        confidence: heuristic_result.confidence,
                        signals: heuristic_result.signals.clone(),
                        source: AnalysisSource::FallbackHeuristic {
                            reason: e.to_string(),
                        },
                    })
                } else {
                    Err(e)
                }
            }
        }
    }
}
