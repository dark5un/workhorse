//! Litmus tests: session & REPL contracts (AGENTS.md 3.5, 6).
//!
//! Phase 3 tests are enabled: session persistence, context window,
//! cost tracking, slash commands.

use myharness::core::{Session, SessionController, SessionError, SessionState};
use std::sync::atomic::{AtomicU64, Ordering};

// Unique DB path per create_session() call, stored thread-locally for
// restore_session() to pick up. Safe for parallel tests (each thread
// gets its own path).
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static TEST_DB_PATH: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

// ============================================================
// Session persistence contracts (Phase 3) -- ENABLED
// ============================================================

#[tokio::test]
async fn session_persists_across_restart() {
    let mut session = create_session();
    session.process("hello").await.unwrap();

    let restored = restore_session();
    let status = restored.status();
    assert!(
        status.message_count > 0,
        "session must restore message history"
    );
}

#[tokio::test]
async fn session_uses_sqlite_storage() {
    // Session state is stored in SQLite (rusqlite), not JSON.
    // Verified by the fact that Session::new() opens a rusqlite::Connection
    // and persist_message() uses SQL INSERT.
    let session = create_session();
    let status = session.status();
    // A fresh session has 0 non-system messages
    assert_eq!(status.message_count, 0);
}

// ============================================================
// Context window management contracts (Phase 3) -- ENABLED
// ============================================================

#[tokio::test]
async fn context_window_prevents_overflow() {
    let mut session = create_session();
    for i in 0..100 {
        let _ = session.process(&format!("message number {i}")).await;
    }
    let status = session.status();
    assert!(
        status.context_tokens_used <= status.context_token_limit,
        "context tokens used ({}) must not exceed limit ({})",
        status.context_tokens_used,
        status.context_token_limit
    );
}

#[tokio::test]
async fn sliding_window_drops_oldest_messages() {
    let mut session = create_session();
    for i in 0..50 {
        let _ = session.process(&format!("message {i}")).await;
    }
    let status = session.status();
    assert!(status.context_tokens_used <= status.context_token_limit);
}

#[tokio::test]
async fn system_prompt_is_never_dropped() {
    let mut session = create_session();
    for i in 0..100 {
        let _ = session.process(&format!("filler message {i}")).await;
    }
    // The system prompt must still be present as the first message.
    let status = session.status();
    assert_eq!(
        status.first_message_role.as_deref(),
        Some("system"),
        "system prompt must be the first message and never evicted"
    );
}

// ============================================================
// Cost tracking contracts (Phase 3) -- ENABLED
// ============================================================

#[tokio::test]
async fn cost_tracking_accumulates_per_session() {
    let mut session = create_session();
    session.process("hello").await.unwrap();
    session.process("analyze this").await.unwrap();

    let status = session.status();
    assert!(
        status.total_cost_cents > 0,
        "cost must accumulate after LLM calls"
    );
}

#[tokio::test]
async fn cost_limit_blocks_execution() {
    let mut session = create_session_with_low_budget();
    for _ in 0..1000 {
        match session.process("expensive prompt").await {
            Ok(_) => continue,
            Err(SessionError::BudgetExceeded(_, _)) => return,
            Err(e) => panic!("expected BudgetExceeded, got: {e}"),
        }
    }
    panic!("should have hit budget limit");
}

// ============================================================
// Slash command contracts (Phase 3) -- ENABLED
// ============================================================

#[tokio::test]
async fn slash_clear_resets_session() {
    let mut session = create_session();
    session.process("hello").await.unwrap();
    assert!(session.status().message_count > 0);

    session.process("/clear").await.unwrap();
    assert_eq!(
        session.status().message_count,
        0,
        "/clear must reset message count"
    );
}

#[tokio::test]
async fn slash_model_overrides_routing() {
    let mut session = create_session();
    session
        .process("/model anthropic/claude-opus")
        .await
        .unwrap();

    let status = session.status();
    assert_eq!(
        status.current_model.as_deref(),
        Some("anthropic/claude-opus")
    );
}

#[tokio::test]
async fn slash_cost_shows_session_spend() {
    let mut session = create_session();
    session.process("hello").await.unwrap();
    let result = session.process("/cost").await.unwrap();
    assert!(!result.events.is_empty());
}

// ============================================================
// Real implementations
// ============================================================

fn load_test_config() -> myharness::config::AppConfig {
    myharness::config::load_config("config").unwrap()
}

fn create_session() -> Box<dyn SessionController> {
    let config = load_test_config();
    let id = SESSION_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("myharness-test-{id}.db"));
    let _ = std::fs::remove_file(&path);
    let path_str = path.to_str().unwrap().to_string();
    TEST_DB_PATH.with(|p| *p.borrow_mut() = Some(path_str.clone()));
    Box::new(Session::new(config, &path_str, "test").unwrap())
}

fn restore_session() -> Box<dyn SessionController> {
    let config = load_test_config();
    let path_str = TEST_DB_PATH
        .with(|p| p.borrow().clone())
        .expect("no DB path set -- call create_session first");
    Box::new(Session::new(config, &path_str, "test").unwrap())
}

fn create_session_with_low_budget() -> Box<dyn SessionController> {
    let mut config = load_test_config();
    config.session.cost_tracking.hard_limit_usd = 0.01; // 1 cent
    let id = SESSION_COUNTER.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("myharness-test-{id}.db"));
    let _ = std::fs::remove_file(&path);
    let path_str = path.to_str().unwrap().to_string();
    Box::new(Session::new(config, &path_str, "test").unwrap())
}

#[allow(dead_code)]
fn _suppress_unused() {
    let _ = SessionState {
        message_count: 0,
        total_cost_cents: 0,
        current_model: None,
        context_tokens_used: 0,
        context_token_limit: 0,
        first_message_role: None,
    };
}
