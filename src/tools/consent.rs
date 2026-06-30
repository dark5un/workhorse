//! Consent sandbox: user approval for destructive operations.
//!
//! When sandbox=consent, destructive tool calls (file write, shell exec)
//! are intercepted and the user is asked for approval before executing.
//! Like Claude Code's permission model.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// The decision returned by a consent callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsentDecision {
    /// User approved the operation.
    Allow,
    /// User denied the operation.
    Deny,
    /// User approved and wants to allow all future operations from this tool.
    AllowAlways,
}

/// Callback trait for consent prompts.
/// Implementations decide how to ask the user (stdin, GUI, auto-approve for tests).
pub trait ConsentCallback: Send + Sync {
    fn request_consent(
        &self,
        tool_name: &str,
        operation: &str,
        args: &serde_json::Value,
    ) -> ConsentDecision;
}

/// Consent sandbox that intercepts tool calls and asks for approval.
pub struct ConsentSandbox {
    callback: Box<dyn ConsentCallback>,
    /// Track how many consent requests were made (for testing).
    consent_requests: AtomicU32,
    /// Whether any operation was denied (for testing).
    last_decision: AtomicBool, // true = allowed
}

impl ConsentSandbox {
    pub fn new(callback: Box<dyn ConsentCallback>) -> Self {
        Self {
            callback,
            consent_requests: AtomicU32::new(0),
            last_decision: AtomicBool::new(true),
        }
    }

    /// Check if an operation should be allowed.
    /// Returns Ok(()) if allowed, Err if denied.
    pub fn check(
        &self,
        tool_name: &str,
        operation: &str,
        args: &serde_json::Value,
    ) -> Result<(), crate::tools::ToolError> {
        self.consent_requests.fetch_add(1, Ordering::SeqCst);
        let decision = self.callback.request_consent(tool_name, operation, args);
        match decision {
            ConsentDecision::Allow | ConsentDecision::AllowAlways => {
                self.last_decision.store(true, Ordering::SeqCst);
                Ok(())
            }
            ConsentDecision::Deny => {
                self.last_decision.store(false, Ordering::SeqCst);
                Err(crate::tools::ToolError::PermissionDenied(format!(
                    "user denied {operation} by tool '{tool_name}'"
                )))
            }
        }
    }

    /// Number of consent requests made (for testing).
    pub fn consent_request_count(&self) -> u32 {
        self.consent_requests.load(Ordering::SeqCst)
    }

    /// Whether the last operation was allowed (for testing).
    pub fn last_operation_allowed(&self) -> bool {
        self.last_decision.load(Ordering::SeqCst)
    }
}

/// Auto-approve all operations (for testing).
pub struct AutoApprove;

impl ConsentCallback for AutoApprove {
    fn request_consent(
        &self,
        _tool_name: &str,
        _operation: &str,
        _args: &serde_json::Value,
    ) -> ConsentDecision {
        ConsentDecision::Allow
    }
}

/// Auto-deny all operations (for testing).
pub struct AutoDeny;

impl ConsentCallback for AutoDeny {
    fn request_consent(
        &self,
        _tool_name: &str,
        _operation: &str,
        _args: &serde_json::Value,
    ) -> ConsentDecision {
        ConsentDecision::Deny
    }
}
