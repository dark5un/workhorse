//! Litmus tests: model effectiveness ranking system (RANKING_DESIGN.md).
//!
//! Phase B tests: rating recording, Bayesian scoring, scope, slash commands.

use myharness::core::{ComplexityTier, ModelId, RankingConfig, RankingEngine, Scope};
use std::sync::atomic::{AtomicU64, Ordering};

static DB_COUNTER: AtomicU64 = AtomicU64::new(0);

fn create_ranking_engine() -> RankingEngine {
    create_ranking_engine_with_config(RankingConfig::default())
}

fn create_ranking_engine_with_config(config: RankingConfig) -> RankingEngine {
    let id = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("myharness-ranking-test-{id}.db"));
    let _ = std::fs::remove_file(&path);
    let conn = rusqlite::Connection::open(path).unwrap();
    RankingEngine::new(conn, config)
}

fn test_model(provider: &str, model: &str) -> ModelId {
    ModelId {
        provider: provider.to_string(),
        model: model.to_string(),
    }
}

// ============================================================
// Rating recording contracts
// ============================================================

#[test]
fn record_rating_and_retrieve_score() {
    let engine = create_ranking_engine();
    let model = test_model("openai", "gpt-4o");

    engine
        .record_rating(&model, ComplexityTier::Complex, 4, None, None, None)
        .unwrap();

    let score = engine.get_score(&model, ComplexityTier::Complex).unwrap();
    // With 1 rating of 4 and prior 3.0, score should be between 3.0 and 4.0
    assert!(
        score > 3.0 && score < 4.0,
        "score {score} should be between prior (3.0) and rating (4.0)"
    );
}

#[test]
fn record_multiple_ratings_score_converges() {
    let engine = create_ranking_engine();
    let model = test_model("openrouter", "llama-3-70b");

    // Record 10 ratings of 5
    for _ in 0..10 {
        engine
            .record_rating(&model, ComplexityTier::Expert, 5, None, None, None)
            .unwrap();
    }

    let score = engine.get_score(&model, ComplexityTier::Expert).unwrap();
    // With 10 ratings of 5, score should be close to 5 but pulled toward prior
    assert!(
        score > 4.5,
        "score {score} should converge toward 5.0 with 10 five-star ratings"
    );
}

#[test]
fn reject_invalid_rating() {
    let engine = create_ranking_engine();
    let model = test_model("openai", "gpt-4o");

    let result = engine.record_rating(&model, ComplexityTier::Simple, 0, None, None, None);
    assert!(result.is_err(), "rating 0 must be rejected");

    let result = engine.record_rating(&model, ComplexityTier::Simple, 6, None, None, None);
    assert!(result.is_err(), "rating 6 must be rejected");
}

#[test]
fn unrated_model_returns_prior_score() {
    let engine = create_ranking_engine();
    let model = test_model("anthropic", "claude-opus");

    let score = engine.get_score(&model, ComplexityTier::Complex).unwrap();
    assert!(
        (score - 3.0).abs() < 0.001,
        "unrated model should return prior (3.0), got {score}"
    );
}

// ============================================================
// Rankings table contracts
// ============================================================

#[test]
fn rankings_sorted_by_score_descending() {
    let engine = create_ranking_engine();

    // Model A: 5 ratings of 5
    let model_a = test_model("openai", "gpt-4o");
    for _ in 0..5 {
        engine
            .record_rating(&model_a, ComplexityTier::Complex, 5, None, None, None)
            .unwrap();
    }

    // Model B: 5 ratings of 2
    let model_b = test_model("anthropic", "claude-haiku");
    for _ in 0..5 {
        engine
            .record_rating(&model_b, ComplexityTier::Complex, 2, None, None, None)
            .unwrap();
    }

    let rankings = engine.get_rankings(ComplexityTier::Complex).unwrap();
    assert!(rankings.len() >= 2);
    // Model A should rank higher than Model B
    assert_eq!(rankings[0].model_id, model_a.as_str());
    assert!(rankings[0].score > rankings[1].score);
}

#[test]
fn rankings_empty_for_unrated_tier() {
    let engine = create_ranking_engine();
    let rankings = engine.get_rankings(ComplexityTier::Simple).unwrap();
    assert!(rankings.is_empty());
}

#[test]
fn rankings_include_sample_count() {
    let engine = create_ranking_engine();
    let model = test_model("openai", "gpt-4o");

    for _ in 0..3 {
        engine
            .record_rating(&model, ComplexityTier::Medium, 4, None, None, None)
            .unwrap();
    }

    let rankings = engine.get_rankings(ComplexityTier::Medium).unwrap();
    assert_eq!(rankings.len(), 1);
    assert_eq!(rankings[0].sample_count, 3);
}

// ============================================================
// Scope contracts
// ============================================================

#[test]
fn reset_ratings_clears_current_scope() {
    let engine = create_ranking_engine();
    let model = test_model("openai", "gpt-4o");

    engine
        .record_rating(&model, ComplexityTier::Complex, 4, None, None, None)
        .unwrap();
    assert!(
        !engine
            .get_rankings(ComplexityTier::Complex)
            .unwrap()
            .is_empty()
    );

    let deleted = engine.reset_ratings(engine.scope()).unwrap();
    assert_eq!(deleted, 1);
    assert!(
        engine
            .get_rankings(ComplexityTier::Complex)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn reset_ratings_global_clears_all() {
    let engine = create_ranking_engine();
    let model = test_model("openai", "gpt-4o");

    engine
        .record_rating(&model, ComplexityTier::Complex, 5, None, None, None)
        .unwrap();
    engine
        .record_rating(&model, ComplexityTier::Simple, 3, None, None, None)
        .unwrap();

    let deleted = engine.reset_ratings(&Scope::Global).unwrap();
    assert_eq!(deleted, 2);
    assert!(
        engine
            .get_rankings(ComplexityTier::Complex)
            .unwrap()
            .is_empty()
    );
    assert!(
        engine
            .get_rankings(ComplexityTier::Simple)
            .unwrap()
            .is_empty()
    );
}

// ============================================================
// Enable/disable contracts
// ============================================================

#[test]
fn ranking_disabled_by_default() {
    let engine = create_ranking_engine();
    assert!(
        !engine.is_enabled(),
        "ranking should be disabled by default"
    );
}

#[test]
fn ranking_can_be_enabled_per_session() {
    let engine = create_ranking_engine();
    engine.set_session_enabled(true);
    assert!(
        engine.is_enabled(),
        "ranking should be enabled after session override"
    );
}

#[test]
fn ranking_can_be_disabled_per_session() {
    let config = RankingConfig {
        enabled: true,
        ..Default::default()
    };
    let engine = create_ranking_engine_with_config(config);
    assert!(engine.is_enabled());

    engine.set_session_enabled(false);
    assert!(
        !engine.is_enabled(),
        "session override should disable ranking"
    );
}

// ============================================================
// Config deserialization contract
// ============================================================

#[test]
fn ranking_config_deserializes_from_yaml() {
    let yaml = r#"
enabled: true
min_samples: 5
prior: 3.5
decay: 0.9
exploration_rate: 0.15
scope: "global"
"#;
    let config: RankingConfig = serde_yaml::from_str(yaml).unwrap();
    assert!(config.enabled);
    assert_eq!(config.min_samples, 5);
    assert!((config.prior - 3.5).abs() < 0.001);
    assert!((config.decay - 0.9).abs() < 0.001);
    assert!((config.exploration_rate - 0.15).abs() < 0.001);
    assert_eq!(config.scope, "global");
}

// ============================================================
// Session integration contracts
// ============================================================

#[tokio::test]
async fn session_rate_command_rates_last_response() {
    use myharness::core::{Session, SessionController};

    let config = myharness::config::load_config("config").unwrap();
    let id = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir()
        .join(format!("myharness-ranking-session-{id}.db"))
        .to_str()
        .unwrap()
        .to_string();
    let _ = std::fs::remove_file(&path);

    let mut session = Session::new(config, &path, "test").unwrap();

    // Send a prompt to get a response from a model
    session.process("hello").await.unwrap();

    // Rate the last response
    let result = session.process("/rate 4").await.unwrap();
    assert!(!result.events.is_empty());
}

#[tokio::test]
async fn session_ranking_status_command() {
    use myharness::core::{Session, SessionController};

    let config = myharness::config::load_config("config").unwrap();
    let id = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir()
        .join(format!("myharness-ranking-status-{id}.db"))
        .to_str()
        .unwrap()
        .to_string();
    let _ = std::fs::remove_file(&path);

    let mut session = Session::new(config, &path, "test").unwrap();

    // Check ranking status
    let result = session.process("/ranking status").await.unwrap();
    assert!(!result.events.is_empty());

    // Enable ranking
    session.process("/ranking on").await.unwrap();

    // Check status again
    let result = session.process("/ranking status").await.unwrap();
    let text = match &result.events[0] {
        myharness::core::SessionEvent::Text(t) => t.clone(),
        _ => panic!("expected text event"),
    };
    assert!(
        text.contains("enabled"),
        "status should show enabled after /ranking on"
    );
}

#[tokio::test]
async fn session_ratings_command_shows_table() {
    use myharness::core::{Session, SessionController};

    let config = myharness::config::load_config("config").unwrap();
    let id = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir()
        .join(format!("myharness-ratings-table-{id}.db"))
        .to_str()
        .unwrap()
        .to_string();
    let _ = std::fs::remove_file(&path);

    let mut session = Session::new(config, &path, "test").unwrap();

    // No ratings yet
    let result = session.process("/ratings").await.unwrap();
    let text = match &result.events[0] {
        myharness::core::SessionEvent::Text(t) => t.clone(),
        _ => panic!("expected text event"),
    };
    assert!(
        text.contains("No ratings"),
        "should show no ratings message"
    );
}

#[tokio::test]
async fn session_reset_ratings_command() {
    use myharness::core::{Session, SessionController};

    let config = myharness::config::load_config("config").unwrap();
    let id = DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir()
        .join(format!("myharness-reset-ratings-{id}.db"))
        .to_str()
        .unwrap()
        .to_string();
    let _ = std::fs::remove_file(&path);

    let mut session = Session::new(config, &path, "test").unwrap();

    // Rate a model first
    session.process("hello").await.unwrap();
    session.process("/rate 5").await.unwrap();

    // Reset ratings
    let result = session.process("/reset-ratings").await.unwrap();
    let text = match &result.events[0] {
        myharness::core::SessionEvent::Text(t) => t.clone(),
        _ => panic!("expected text event"),
    };
    assert!(text.contains("Reset"), "should show reset message");
}
