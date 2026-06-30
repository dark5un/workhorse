//! Litmus tests: cross-cutting type contracts and trait invariants.
//!
//! These tests encode the architectural contracts from AGENTS.md that must
//! hold across all future refactoring. They test TYPES and TRAITS, not
//! specific implementations.

use myharness::adapters::{LLMAdapter, LLMError, ResponseEvent, ToolInvocation, Usage};
use myharness::config::AppConfig;
use myharness::core::{
    AnalysisError, AnalysisSource, ComplexityResult, ComplexityTier, Cost, ModelId, PromptAnalyzer,
    Router, RoutingError, SessionController, SessionError,
};
use myharness::tools::{SandboxLevel, Tool, ToolContent, ToolError, ToolResult};

// ============================================================
// ModelId contracts (AGENTS.md 3.2, 9)
// ============================================================

#[test]
fn model_id_parses_valid_provider_model() {
    let id = ModelId::parse("openai/gpt-4o").unwrap();
    assert_eq!(id.provider, "openai");
    assert_eq!(id.model, "gpt-4o");
}

#[test]
fn model_id_rejects_malformed_strings() {
    assert!(ModelId::parse("no-slash").is_none());
    assert!(ModelId::parse("/leading-slash").is_none());
    assert!(ModelId::parse("trailing-slash/").is_none());
    assert!(ModelId::parse("").is_none());
    assert!(ModelId::parse("  /  ").is_none());
}

#[test]
fn model_id_display_round_trips_through_fromstr() {
    let id = ModelId::parse("anthropic/claude-sonnet").unwrap();
    assert_eq!(id.as_str(), "anthropic/claude-sonnet");
    assert_eq!(id.to_string(), "anthropic/claude-sonnet");
    let parsed: ModelId = "anthropic/claude-sonnet".parse().unwrap();
    assert_eq!(parsed, id);
}

#[test]
fn model_id_fromstr_rejects_invalid() {
    let result: Result<ModelId, RoutingError> = "no-slash".parse();
    assert!(result.is_err());
}

// ============================================================
// Cost newtype contracts (AGENTS.md 3.2, 3.3)
// ============================================================

#[test]
fn cost_from_usd_converts_to_cents() {
    assert_eq!(Cost::from_usd(1.50).0, 150);
    assert_eq!(Cost::from_usd(0.01).0, 1);
    assert_eq!(Cost::from_usd(0.0).0, 0);
}

#[test]
fn cost_as_usd_converts_back() {
    let cost = Cost(375);
    assert!((cost.as_usd() - 3.75).abs() < 0.001);
}

#[test]
fn cost_add_accumulates() {
    let a = Cost::from_usd(1.50);
    let b = Cost::from_usd(2.25);
    let total = a.add(b);
    assert_eq!(total.0, 375);
}

// ============================================================
// Dyn-compatibility contracts (AGENTS.md 3.0)
// All async traits must work as Box<dyn Trait> via #[async_trait].
// ============================================================

#[test]
fn all_traits_are_dyn_compatible() {
    fn _assert_send_sync<T: ?Sized + Send + Sync>() {}

    _assert_send_sync::<dyn PromptAnalyzer>();
    _assert_send_sync::<dyn Router>();
    _assert_send_sync::<dyn LLMAdapter>();
    _assert_send_sync::<dyn Tool>();
    _assert_send_sync::<dyn SessionController>();

    // If this compiles, all traits are dyn-compatible.
}

// ============================================================
// Error type contracts (AGENTS.md 9)
// Every module error type implements std::error::Error + Send + Sync.
// ============================================================

#[test]
fn all_error_types_implement_std_error() {
    fn _assert_error<T: std::error::Error + Send + Sync + 'static>() {}

    _assert_error::<AnalysisError>();
    _assert_error::<RoutingError>();
    _assert_error::<LLMError>();
    _assert_error::<ToolError>();
    _assert_error::<myharness::config::ConfigError>();
    _assert_error::<SessionError>();
}

// ============================================================
// Enum variant contracts (AGENTS.md 3.1, 3.3, 3.4, 7)
// ============================================================

#[test]
fn complexity_tier_has_four_levels() {
    assert_eq!(
        vec![
            ComplexityTier::Simple,
            ComplexityTier::Medium,
            ComplexityTier::Complex,
            ComplexityTier::Expert,
        ]
        .len(),
        4
    );
    // All tiers are distinct
    assert_ne!(ComplexityTier::Simple, ComplexityTier::Medium);
    assert_ne!(ComplexityTier::Medium, ComplexityTier::Complex);
    assert_ne!(ComplexityTier::Complex, ComplexityTier::Expert);
}

#[test]
fn analysis_source_covers_all_sources() {
    let _heuristic = AnalysisSource::Heuristic;
    let _classifier = AnalysisSource::Classifier {
        model: "openai/gpt-4o-mini".to_string(),
    };
    let _fallback = AnalysisSource::FallbackHeuristic {
        reason: "classifier timeout".to_string(),
    };
}

#[test]
fn response_event_covers_chunk_toolcall_done() {
    let _chunk = ResponseEvent::Chunk("hello".to_string());
    let _tool_call = ResponseEvent::ToolCall(ToolInvocation {
        call_id: "call_1".to_string(),
        tool_name: "filesystem".to_string(),
        arguments: serde_json::json!({}),
    });
    let _done = ResponseEvent::Done(Usage {
        input_tokens: 100,
        output_tokens: 50,
        cost: Cost(5),
    });
}

#[test]
fn tool_content_matches_mcp_types() {
    let _text = ToolContent::Text("result".to_string());
    let _image = ToolContent::Image {
        mime_type: "image/png".to_string(),
        data: vec![0x89, 0x50, 0x4E, 0x47],
    };
    let _resource = ToolContent::Resource {
        uri: "file:///tmp/test".to_string(),
        mime_type: "text/plain".to_string(),
    };
}

#[test]
fn sandbox_level_has_four_variants() {
    assert_eq!(
        vec![
            SandboxLevel::Consent,
            SandboxLevel::Wasmtime,
            SandboxLevel::Docker,
            SandboxLevel::None,
        ]
        .len(),
        4
    );
    assert_ne!(SandboxLevel::Consent, SandboxLevel::None);
}

// ============================================================
// Config schema contracts (AGENTS.md 4)
// Config structs must be deserializable from YAML.
// ============================================================

#[test]
fn config_structs_are_deserializable() {
    fn _assert_deserialize<T: serde::de::DeserializeOwned>() {}
    _assert_deserialize::<AppConfig>();
}

#[test]
fn full_config_deserializes_from_valid_yaml() {
    let yaml = r#"
analyzer:
  heuristic:
    enabled: true
    tiers:
      simple:
        thresholds: { min_tokens: 0, max_tokens: 50 }
        keywords: ["hello", "translate"]
        models: ["openai/gpt-4o-mini"]
      medium:
        thresholds: { min_tokens: 51, max_tokens: 200 }
        keywords: ["analyze"]
        models: ["openai/gpt-4o"]
      complex:
        thresholds: { min_tokens: 201, max_tokens: 4096 }
        keywords: ["debug"]
        models: ["anthropic/claude-opus"]
      expert:
        thresholds: { min_tokens: 4097, max_tokens: null }
        keywords: ["reason"]
        models: ["custom/70b"]
    fallback_policy: "sequential"
    timeout_seconds: 30
  classifier:
    enabled: true
    model: "openai/gpt-4o-mini"
    fallback_on_error: true
    timeout_seconds: 10
tools:
  mcp_servers:
    - name: "filesystem"
      transport: "subprocess"
      command: "mcp-server-filesystem"
      sandbox: "consent"
  defaults:
    sandbox: "consent"
    session_temp_dir: "/tmp/harness-test"
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key_env: "OPENAI_API_KEY"
    pricing:
      gpt-4o:
        input: 250
        output: 500
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
"#;
    let config: AppConfig = serde_yaml::from_str(yaml).expect("valid config must deserialize");
    assert!(config.analyzer.heuristic.enabled);
    assert_eq!(config.analyzer.heuristic.tiers.len(), 4);
    assert_eq!(
        config.analyzer.classifier.as_ref().unwrap().model,
        "openai/gpt-4o-mini"
    );
    assert_eq!(config.tools.mcp_servers.len(), 1);
    assert_eq!(config.providers.len(), 1);
    assert_eq!(config.session.storage, "sqlite");
}

#[test]
fn config_rejects_missing_required_field() {
    let yaml = r#"
analyzer:
  heuristic:
    enabled: true
    tiers: {}
    # missing fallback_policy and timeout_seconds
  classifier: null
tools:
  mcp_servers: []
providers: {}
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
"#;
    let result: Result<AppConfig, _> = serde_yaml::from_str(yaml);
    assert!(result.is_err(), "config with missing fields must fail");
}

// ============================================================
// ToolResult / ToolContent contracts (AGENTS.md 3.4)
// ============================================================

#[test]
fn tool_result_with_error_flag() {
    let ok = ToolResult {
        content: vec![ToolContent::Text("done".to_string())],
        is_error: false,
    };
    let err = ToolResult {
        content: vec![ToolContent::Text("permission denied".to_string())],
        is_error: true,
    };
    assert!(!ok.is_error);
    assert!(err.is_error);
}

// ============================================================
// ComplexityResult shape (AGENTS.md 3.1)
// ============================================================

#[test]
fn complexity_result_carries_source_and_signals() {
    let result = ComplexityResult {
        tier: ComplexityTier::Complex,
        confidence: 0.85,
        signals: vec!["keyword:debug".to_string(), "length:250".to_string()],
        source: AnalysisSource::Heuristic,
        task_type: myharness::core::TaskType::General,
    };
    assert_eq!(result.tier, ComplexityTier::Complex);
    assert!((result.confidence - 0.85).abs() < 0.001);
    assert_eq!(result.signals.len(), 2);
    assert_eq!(result.source, AnalysisSource::Heuristic);
}
