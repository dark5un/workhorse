//! Retry / backoff infrastructure for adapter fallback chains.
//!
//! Uses exponential backoff with jitter. Retry policy is per-provider
//! configurable (max retries, base delay). The `backoff` crate provides
//! the ExponentialBackoff builder.

use backoff::backoff::Backoff;
use std::time::Duration;

/// Retry policy for a provider. All values come from config.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
}

/// Error type for retry operations.
#[derive(Debug, thiserror::Error)]
#[error("operation failed after {max_retries} retries: {last_error}")]
pub struct RetryError {
    pub max_retries: u32,
    pub last_error: String,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// Create a backoff::ExponentialBackoff from this policy.
    pub fn to_backoff(&self) -> backoff::ExponentialBackoff {
        backoff::ExponentialBackoffBuilder::new()
            .with_initial_interval(self.base_delay)
            .with_max_interval(self.max_delay)
            .with_max_elapsed_time(Some(Duration::from_secs(120)))
            .build()
    }

    /// Run a fallible async operation with retry + backoff.
    ///
    /// Returns the first Ok result, or the last error after exhausting retries.
    /// Retries up to `max_retries` times with exponential backoff + jitter.
    pub async fn run_with_retry<T, E, F, Fut>(&self, mut op: F) -> Result<T, E>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut backoff = self.to_backoff();
        let mut attempt = 0u32;
        loop {
            match op().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempt += 1;
                    if attempt > self.max_retries {
                        tracing::error!(error = ?e, attempts = attempt, "operation failed, no more retries");
                        return Err(e);
                    }
                    match backoff.next_backoff() {
                        Some(delay) => {
                            tracing::warn!(
                                error = ?e,
                                attempt = attempt,
                                max_retries = self.max_retries,
                                retry_after_ms = delay.as_millis(),
                                "operation failed, retrying"
                            );
                            tokio::time::sleep(delay).await;
                        }
                        None => {
                            tracing::error!(error = ?e, "backoff exhausted");
                            return Err(e);
                        }
                    }
                }
            }
        }
    }
}
