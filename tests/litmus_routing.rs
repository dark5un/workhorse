//! Litmus tests: routing & complexity analysis contracts (AGENTS.md 3.1, 3.2, 5).
//!
//! Phase 1 (heuristic analyzer) and Phase 2 (router) tests are enabled.
//! Phase 5 (classifier) tests remain ignored.

use myharness::config::AppConfig;
use myharness::core::{
    AnalysisError, AnalysisSource, ComplexityResult, ComplexityTier, ConfigRouter, Cost,
    HeuristicAnalyzer, ModelId, ModelSpec, PromptAnalyzer, Router, RoutingError,
};

// ============================================================
// Heuristic analyzer contracts (Phase 1) -- ENABLED
// ============================================================

#[tokio::test]
async fn heuristic_analyzer_classifies_simple_prompt() {
    // "hello" is a keyword for simple tier; short prompt
    let analyzer = create_heuristic_analyzer();
    let result = analyzer.analyze("hello").await.unwrap();
    assert_eq!(result.tier, ComplexityTier::Simple);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert_eq!(result.source, AnalysisSource::Heuristic);
}

#[tokio::test]
async fn heuristic_analyzer_classifies_complex_prompt() {
    let analyzer = create_heuristic_analyzer();
    // "debug" keyword + long enough to hit complex tier
    let prompt = "debug this architecture: ".to_string() + &"x".repeat(200);
    let result = analyzer.analyze(&prompt).await.unwrap();
    assert_eq!(result.tier, ComplexityTier::Complex);
}

#[tokio::test]
async fn heuristic_analyzer_confidence_in_valid_range() {
    let analyzer = create_heuristic_analyzer();
    for prompt in &[
        "hi",
        "analyze this",
        "debug this complex system plan",
        "reason about this",
    ] {
        let result = analyzer.analyze(prompt).await.unwrap();
        assert!(
            result.confidence >= 0.0 && result.confidence <= 1.0,
            "confidence {} out of range for prompt: {}",
            result.confidence,
            prompt
        );
    }
}

#[tokio::test]
async fn heuristic_analyzer_signals_explain_decision() {
    let analyzer = create_heuristic_analyzer();
    let result = analyzer.analyze("debug this").await.unwrap();
    assert!(
        !result.signals.is_empty(),
        "analyzer must produce signals explaining the decision"
    );
}

// ============================================================
// Token counting contract (Phase 1) -- ENABLED
// ============================================================

#[tokio::test]
async fn analyzer_uses_real_token_counting() {
    let analyzer = create_heuristic_analyzer();
    let result = analyzer.analyze("tokenization").await.unwrap();
    let has_token_signal = result
        .signals
        .iter()
        .any(|s| s.contains("token") || s.contains("length"));
    assert!(has_token_signal, "signals should reference token count");
}

// ============================================================
// No hardcoded config values contract (AGENTS.md 9) -- ENABLED
// ============================================================

#[tokio::test]
async fn analyzer_reads_keywords_from_config() {
    let config_a = config_with_keyword("hello");
    let config_b = config_with_keyword("greetings");

    let analyzer_a = create_analyzer_from_config(config_a);
    let analyzer_b = create_analyzer_from_config(config_b);

    let result_a = analyzer_a.analyze("hello").await.unwrap();
    let result_b = analyzer_b.analyze("hello").await.unwrap();

    assert_eq!(result_a.tier, ComplexityTier::Simple);
    assert_ne!(result_b.tier, ComplexityTier::Simple);
}

// ============================================================
// Router contracts (Phase 2) -- ENABLED
// ============================================================

#[tokio::test]
async fn router_selects_model_for_each_tier() {
    let router = create_router();
    for tier in [
        ComplexityTier::Simple,
        ComplexityTier::Medium,
        ComplexityTier::Complex,
        ComplexityTier::Expert,
    ] {
        let complexity = ComplexityResult {
            tier,
            confidence: 0.9,
            signals: vec![],
            source: AnalysisSource::Heuristic,
        };
        let spec = router.route(&complexity, None).await.unwrap();
        assert!(
            !spec.fallback_chain.is_empty(),
            "fallback chain must be non-empty for tier {:?}",
            tier
        );
    }
}

#[tokio::test]
async fn router_user_override_bypasses_routing() {
    let router = create_router();
    let complexity = ComplexityResult {
        tier: ComplexityTier::Simple,
        confidence: 0.99,
        signals: vec![],
        source: AnalysisSource::Heuristic,
    };
    let override_model = ModelId::parse("anthropic/claude-opus").unwrap();
    let spec = router
        .route(&complexity, Some(&override_model))
        .await
        .unwrap();
    assert_eq!(spec.model_id, override_model);
}

#[tokio::test]
async fn router_model_spec_uses_model_id_not_bare_string() {
    let router = create_router();
    let complexity = ComplexityResult {
        tier: ComplexityTier::Medium,
        confidence: 0.8,
        signals: vec![],
        source: AnalysisSource::Heuristic,
    };
    let spec = router.route(&complexity, None).await.unwrap();
    assert!(!spec.model_id.provider.is_empty());
    assert!(!spec.model_id.model.is_empty());
}

#[tokio::test]
async fn router_budget_limit_uses_cost_type() {
    let router = create_router();
    let complexity = ComplexityResult {
        tier: ComplexityTier::Complex,
        confidence: 0.85,
        signals: vec![],
        source: AnalysisSource::Heuristic,
    };
    let spec = router.route(&complexity, None).await.unwrap();
    if let Some(budget) = spec.budget_limit {
        let _cents: u64 = budget.0;
    }
}

// ============================================================
// Classifier stage contracts (Phase 5) -- still ignored
// ============================================================

#[ignore = "Phase 5: classifier stage not yet implemented"]
#[tokio::test]
async fn classifier_overrides_heuristic_result() {
    let analyzer = create_classifier_analyzer();
    let result = analyzer
        .analyze("hello, please debug this complex distributed system")
        .await;
    let result = result.unwrap();
    assert!(matches!(result.source, AnalysisSource::Classifier { .. }));
}

#[ignore = "Phase 5: classifier stage not yet implemented"]
#[tokio::test]
async fn classifier_falls_back_to_heuristic_on_failure() {
    let analyzer = create_failing_classifier_analyzer();
    let result = analyzer.analyze("debug this").await.unwrap();
    assert!(matches!(
        result.source,
        AnalysisSource::FallbackHeuristic { .. }
    ));
}

// ============================================================
// Real implementations
// ============================================================

fn create_heuristic_analyzer() -> Box<dyn PromptAnalyzer> {
    let config = myharness::config::load_config("config").unwrap();
    Box::new(HeuristicAnalyzer::from_app_config(&config).unwrap())
}

fn create_analyzer_from_config(config: AppConfig) -> Box<dyn PromptAnalyzer> {
    Box::new(HeuristicAnalyzer::from_app_config(&config).unwrap())
}

fn create_router() -> Box<dyn Router> {
    let config = myharness::config::load_config("config").unwrap();
    Box::new(ConfigRouter::from_app_config(&config).unwrap())
}

// ============================================================
// Stubs for future phases
// ============================================================

fn create_classifier_analyzer() -> Box<dyn PromptAnalyzer> {
    unimplemented!("Phase 5")
}

fn create_failing_classifier_analyzer() -> Box<dyn PromptAnalyzer> {
    unimplemented!("Phase 5")
}

fn config_with_keyword(keyword: &str) -> AppConfig {
    let yaml = format!(
        r#"
analyzer:
  heuristic:
    enabled: true
    tiers:
      simple:
        thresholds: {{ min_tokens: 0, max_tokens: 50 }}
        keywords: ["{kw}"]
        models: ["openai/gpt-4o-mini"]
      medium:
        thresholds: {{ min_tokens: 51, max_tokens: 200 }}
        keywords: ["analyze"]
        models: ["openai/gpt-4o"]
      complex:
        thresholds: {{ min_tokens: 201, max_tokens: 4096 }}
        keywords: ["debug"]
        models: ["anthropic/claude-opus"]
      expert:
        thresholds: {{ min_tokens: 4097, max_tokens: null }}
        keywords: ["reason"]
        models: ["custom/70b"]
    fallback_policy: "sequential"
    timeout_seconds: 30
  classifier: null
tools:
  mcp_servers: []
  defaults:
    sandbox: "consent"
    session_temp_dir: "/tmp/harness-test"
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
session:
  storage: "sqlite"
  path: "~/.harness/sessions.db"
  context_window:
    strategy: "sliding_window"
    max_tokens: 128000
    sticky_system_prompt: true
  system_prompt_file: "config/system_prompt.md"
  cost_tracking:
    enabled: true
    warn_at_usd: 5.0
    hard_limit_usd: 20.0
"#,
        kw = keyword
    );
    serde_yaml::from_str(&yaml).unwrap()
}

// Suppress unused warnings for types only referenced in ignored tests
#[allow(dead_code)]
fn _suppress_unused() {
    let _ = AnalysisError::Tokenization(String::new());
    let _ = RoutingError::NoModelsForTier;
    let _ = ModelSpec {
        model_id: ModelId::parse("a/b").unwrap(),
        fallback_chain: vec![],
        budget_limit: Some(Cost(100)),
    };
}
