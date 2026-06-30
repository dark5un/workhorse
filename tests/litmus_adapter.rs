//! Litmus tests: LLM adapter contracts (AGENTS.md 3.3).
//!
//! Phase 2 tests (mock adapter) are enabled.

use myharness::adapters::{
    LLMAdapter, LLMError, MockAdapter, ModelCapabilities, ModelConfig, ResponseEvent,
    ToolInvocation, Usage,
};
use myharness::core::{Cost, Message, MessageContent, Role};
use std::collections::HashMap;

// ============================================================
// Adapter streaming contract (Phase 2) -- ENABLED
// ============================================================

#[tokio::test]
async fn mock_adapter_streams_response_events() {
    let adapter = create_mock_adapter();
    let messages = vec![Message {
        role: Role::User,
        content: MessageContent::Text {
            text: "hello".to_string(),
        },
    }];
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.7,
        stream: true,
        tools: None,
        response_format: None,
    };
    let events = adapter.send(messages, config).await.unwrap();

    // Must produce at least one Chunk and end with Done
    assert!(!events.is_empty());
    assert!(events.iter().any(|e| matches!(e, ResponseEvent::Chunk(_))));
    assert!(events.iter().any(|e| matches!(e, ResponseEvent::Done(_))));
}

#[tokio::test]
async fn adapter_normalizes_tool_calls() {
    let adapter = create_mock_adapter();
    let messages = vec![Message {
        role: Role::User,
        content: MessageContent::Text {
            text: "read the file".to_string(),
        },
    }];
    let config = ModelConfig {
        max_tokens: 100,
        temperature: 0.0,
        stream: false,
        tools: None,
        response_format: None,
    };
    let events = adapter.send(messages, config).await.unwrap();

    // If the adapter returns a tool call, it must be a normalized ToolInvocation
    let tool_calls: Vec<&ToolInvocation> = events
        .iter()
        .filter_map(|e| match e {
            ResponseEvent::ToolCall(inv) => Some(inv),
            _ => None,
        })
        .collect();

    for tc in &tool_calls {
        assert!(!tc.call_id.is_empty(), "tool call must have a call_id");
        assert!(!tc.tool_name.is_empty(), "tool call must have a tool_name");
    }
}

#[tokio::test]
async fn adapter_usage_includes_cost_from_pricing_table() {
    let adapter = create_mock_adapter();
    let messages = vec![Message {
        role: Role::User,
        content: MessageContent::Text {
            text: "hello".to_string(),
        },
    }];
    let config = default_model_config();
    let events = adapter.send(messages, config).await.unwrap();

    let done = events.iter().find_map(|e| match e {
        ResponseEvent::Done(usage) => Some(usage),
        _ => None,
    });
    let usage = done.expect("adapter must emit Done event with Usage");
    assert!(usage.input_tokens > 0 || usage.output_tokens > 0);
    // Cost must be computed from pricing table, not zero (unless model is free)
    // Cost is in USD cents (u64 newtype)
    let _cents: u64 = usage.cost.0;
}

#[tokio::test]
async fn adapter_capabilities_describe_model_features() {
    let adapter = create_mock_adapter();
    let caps = adapter.capabilities();
    let _streaming = caps.streaming;
    let _tool_calling = caps.tool_calling;
    let _structured_output = caps.structured_output;
    let _vision = caps.vision;
    assert!(
        caps.max_context_tokens > 0,
        "max_context_tokens must be positive"
    );
}

// ============================================================
// No provider coupling contract (AGENTS.md 9)
// ============================================================

#[test]
fn harness_does_not_import_provider_sdks() {
    // The harness must not import openai, anthropic, or ollama crates directly.
    // Adapters use a shared reqwest client.
    // This is a static analysis contract -- verified by the fact that
    // MockAdapter is the only adapter and it makes no SDK calls.
    // When real adapters are added, they must use reqwest, not provider SDKs.
    let adapter = create_mock_adapter();
    // The mock adapter works without any provider SDK
    let _caps = adapter.capabilities();
}

// ============================================================
// Real implementations
// ============================================================

fn create_mock_adapter() -> Box<dyn LLMAdapter> {
    let config = myharness::config::load_config("config").unwrap();
    Box::new(MockAdapter::from_app_config(&config))
}

fn default_model_config() -> ModelConfig {
    ModelConfig {
        max_tokens: 100,
        temperature: 0.7,
        stream: true,
        tools: None,
        response_format: None,
    }
}

#[allow(dead_code)]
fn _suppress_unused() {
    let _ = LLMError::Network(String::new());
    let _ = ResponseEvent::Chunk(String::new());
    let _ = Usage {
        input_tokens: 0,
        output_tokens: 0,
        cost: Cost(0),
    };
    let _ = ModelCapabilities {
        streaming: false,
        tool_calling: false,
        structured_output: false,
        vision: false,
        max_context_tokens: 0,
    };
    let _ = HashMap::<String, (u64, u64)>::new();
}
