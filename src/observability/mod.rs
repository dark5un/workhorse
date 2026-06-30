//! Observability: tracing-based structured logging and telemetry.

use tracing_subscriber::EnvFilter;

/// Initialize the tracing subscriber with JSON output and env filter.
pub fn init() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(true)
        .init();
}
