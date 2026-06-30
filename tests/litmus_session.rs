//! Litmus tests: session & REPL contracts (AGENTS.md 3.5, 6).
//!
//! These tests are #[ignore] until Phase 3 (interactive REPL) is implemented.

use myharness::core::{SessionController, SessionError, SessionState};

// ============================================================
// Session persistence contracts (Phase 3)
// ============================================================

#[ignore = "Phase 3: session storage not yet implemented"]
#[tokio::test]
async fn session_persists_across_restart() {
    // After writing to a session and "restarting" (creating a new controller
    // pointing at the same storage), the message history must be recovered.
    let mut session = create_session();
    session.process("hello").await.unwrap();

    let restored = restore_session();
    let status = restored.status();
    assert!(
        status.message_count > 0,
        "session must restore message history"
    );
}

#[ignore = "Phase 3: SQLite backend not yet implemented"]
#[tokio::test]
async fn session_uses_sqlite_storage() {
    // Session state must be stored in SQLite (rusqlite), not JSON.
    // This is verified by checking the storage backend type or the file format.
}

// ============================================================
// Context window management contracts (Phase 3)
// ============================================================

#[ignore = "Phase 3: context window management not yet implemented"]
#[tokio::test]
async fn context_window_prevents_overflow() {
    let mut session = create_session();
    // Fill context to exceed max_tokens
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

#[ignore = "Phase 3: sliding window not yet implemented"]
#[tokio::test]
async fn sliding_window_drops_oldest_messages() {
    let mut session = create_session();
    for i in 0..50 {
        let _ = session.process(&format!("message {i}")).await;
    }
    // After exceeding context limit, oldest messages should be dropped.
    // The system prompt must NOT be dropped (sticky).
    let status = session.status();
    assert!(status.context_tokens_used <= status.context_token_limit);
}

#[ignore = "Phase 3: system prompt stickiness not yet implemented"]
#[tokio::test]
async fn system_prompt_is_never_dropped() {
    let mut session = create_session();
    // Fill context to trigger eviction
    for i in 0..100 {
        let _ = session.process(&format!("filler message {i}")).await;
    }
    // The system prompt must still be present in the context.
    // This is verified by checking the first message role is System.
}

// ============================================================
// Cost tracking contracts (Phase 3)
// ============================================================

#[ignore = "Phase 3: cost tracking not yet implemented"]
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

#[ignore = "Phase 3: cost limit not yet implemented"]
#[tokio::test]
async fn cost_limit_blocks_execution() {
    let mut session = create_session_with_low_budget();
    // After hitting the hard limit, further processing should return BudgetExceeded
    for _ in 0..1000 {
        match session.process("expensive prompt").await {
            Ok(_) => continue,
            Err(SessionError::BudgetExceeded(_, _)) => return, // expected
            Err(e) => panic!("expected BudgetExceeded, got: {e}"),
        }
    }
    panic!("should have hit budget limit");
}

// ============================================================
// Slash command contracts (Phase 3)
// ============================================================

#[ignore = "Phase 3: slash commands not yet implemented"]
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

#[ignore = "Phase 3: slash commands not yet implemented"]
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

#[ignore = "Phase 3: slash commands not yet implemented"]
#[tokio::test]
async fn slash_cost_shows_session_spend() {
    let mut session = create_session();
    session.process("hello").await.unwrap();
    // /cost should show the current session spend without making an LLM call
    let result = session.process("/cost").await.unwrap();
    // The output should contain cost information
    assert!(!result.events.is_empty());
}

// ============================================================
// Mock implementations
// ============================================================

fn create_session() -> Box<dyn SessionController> {
    unimplemented!("Phase 3")
}

fn restore_session() -> Box<dyn SessionController> {
    unimplemented!("Phase 3")
}

fn create_session_with_low_budget() -> Box<dyn SessionController> {
    unimplemented!("Phase 3")
}

#[allow(dead_code)]
fn _suppress_unused() {
    let _ = SessionState {
        message_count: 0,
        total_cost_cents: 0,
        current_model: None,
        context_tokens_used: 0,
        context_token_limit: 0,
    };
}
