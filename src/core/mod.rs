//! Core types: shared identifiers, complexity analysis, routing, session.
//!
//! This module defines the traits and types that other modules depend on.
//! Trait definitions live here; concrete implementations live in child modules.

pub mod analyzer;
pub mod router;
pub mod session;

pub use analyzer::{
    AnalysisError, AnalysisSource, ComplexityResult, ComplexityTier, PromptAnalyzer,
};
pub use router::{ModelId, ModelSpec, Router, RoutingError};
pub use session::{SessionController, SessionError, SessionState};

use serde::{Deserialize, Serialize};

/// Monetary cost in USD cents. Newtype prevents mixing with raw token counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Cost(pub u64);

impl Cost {
    pub fn from_usd(usd: f64) -> Self {
        Self((usd * 100.0).round() as u64)
    }

    pub fn as_usd(&self) -> f64 {
        self.0 as f64 / 100.0
    }

    pub fn add(&self, other: Cost) -> Cost {
        Cost(self.0 + other.0)
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text {
        text: String,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        call_id: String,
        result: serde_json::Value,
    },
}
