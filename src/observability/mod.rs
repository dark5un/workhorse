//! Observability: tracing-based structured logging and telemetry.
//!
//! Provides JSON-formatted structured logging via `tracing` and
//! `tracing-subscriber`. OpenTelemetry export is supported via the
//! `otel` feature flag (tracing-opentelemetry).

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Initialize the tracing subscriber.
///
/// - Default: pretty-printed logs to stdout with env filter (`RUST_LOG`).
/// - `HARNESS_LOG_JSON=1`: JSON-formatted logs to stdout.
/// - `otel` feature: exports spans/traces via OpenTelemetry.
pub fn init() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("workhorse=info,warn"));

    let use_json = std::env::var("HARNESS_LOG_JSON").is_ok();

    #[cfg(feature = "otel")]
    {
        use opentelemetry_sdk::trace::TracerProvider;
        let tracer = opentelemetry_sdk::trace::TracerProvider::builder()
            .build()
            .tracer("workhorse");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        if use_json {
            let fmt_layer = fmt::layer().json().with_target(true);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .with(otel_layer)
                .init();
        } else {
            let fmt_layer = fmt::layer().with_target(true);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .with(otel_layer)
                .init();
        }
    }

    #[cfg(not(feature = "otel"))]
    {
        if use_json {
            let fmt_layer = fmt::layer().json().with_target(true);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .init();
        } else {
            let fmt_layer = fmt::layer().with_target(true);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(fmt_layer)
                .init();
        }
    }
}

/// Shut down tracing providers. Call before program exit to flush
/// any pending telemetry data.
pub fn shutdown() {
    #[cfg(feature = "otel")]
    {
        opentelemetry::global::shutdown_tracer_provider();
    }
}
