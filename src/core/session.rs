//! Session controller: REPL loop, context window management, state persistence.
//!
//! Phase 3 implements the full Session struct with SQLite storage,
//! sliding window context management, cost tracking, and slash commands.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;
use thiserror::Error;

use crate::adapters::{AdapterFactory, ModelConfig, ModelInfo, ResponseEvent, Usage};
use crate::config::AppConfig;
use crate::core::analyzer::HeuristicAnalyzer;
use crate::core::learning_router::LearningRouter;
use crate::core::ranking::{RankingEngine, Scope};
use crate::core::{
    ComplexityTier, Cost, Message, MessageContent, ModelId, PromptAnalyzer, Role, Router,
};
use crate::tools::ToolResult;

/// Controls the interactive session: processing input, managing state, reset.
#[async_trait]
pub trait SessionController: Send + Sync {
    async fn process(&mut self, input: &str) -> Result<SessionOutput, SessionError>;
    async fn reset(&mut self);
    fn status(&self) -> SessionState;
}

/// Output from processing a single input turn.
#[derive(Debug, Clone)]
pub struct SessionOutput {
    pub events: Vec<SessionEvent>,
    pub usage: Option<Usage>,
    /// Which model was used for this turn (None for slash commands).
    pub model_used: Option<ModelId>,
}

#[derive(Debug, Clone)]
pub enum SessionEvent {
    Text(String),
    ToolCall(crate::adapters::ToolInvocation),
    ToolResult(ToolResult),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct SessionState {
    pub message_count: usize,
    pub total_cost_cents: u64,
    pub current_model: Option<String>,
    pub context_tokens_used: usize,
    pub context_token_limit: usize,
    /// Role of the first message in context (e.g., "system" if system prompt is present).
    pub first_message_role: Option<String>,
}

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("adapter error: {0}")]
    Adapter(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("context window exhausted")]
    ContextExhausted,
    #[error("budget exceeded: ${0:.2} spent, limit ${1:.2}")]
    BudgetExceeded(f64, f64),
}

/// Concrete session implementation with SQLite persistence.
pub struct Session {
    config: AppConfig,
    messages: Vec<Message>,
    conn: Mutex<rusqlite::Connection>,
    adapter_factory: AdapterFactory,
    analyzer: Box<dyn PromptAnalyzer>,
    router: Box<dyn Router>,
    bpe: tiktoken_rs::CoreBPE,
    current_model: Option<ModelId>,
    total_cost: Cost,
    session_id: String,
    system_prompt: String,
    /// The model used in the last LLM call (for /rate command).
    last_model_used: Option<ModelId>,
    /// The complexity tier of the last prompt (for /rate command).
    last_tier: Option<ComplexityTier>,
    /// Ranking engine for model effectiveness tracking (shared with LearningRouter).
    ranking: std::sync::Arc<RankingEngine>,
    /// Discovered model metadata from provider APIs. Populated lazily on first
    /// `process()` call. Keyed by `provider/model` (canonical ModelId format).
    model_info: std::sync::Mutex<HashMap<String, ModelInfo>>,
    /// Whether model discovery has been attempted (avoids retrying on every call).
    discovery_attempted: std::sync::atomic::AtomicBool,
}

impl Session {
    /// Create a new session with the given config, DB path, and session ID.
    ///
    /// If the DB already contains messages for this session_id, they are loaded.
    /// The system prompt is loaded from `config.session.system_prompt_file`.
    pub fn new(config: AppConfig, db_path: &str, session_id: &str) -> Result<Self, SessionError> {
        let conn = rusqlite::Connection::open(db_path)
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );",
        )
        .map_err(|e| SessionError::Storage(e.to_string()))?;

        // Load system prompt
        let system_prompt =
            std::fs::read_to_string(&config.session.system_prompt_file).unwrap_or_default();

        // Build initial messages: system prompt + loaded messages
        let mut messages = Vec::new();
        if !system_prompt.is_empty() {
            messages.push(Message {
                role: Role::System,
                content: MessageContent::Text {
                    text: system_prompt.clone(),
                },
            });
        }

        // Load existing messages from DB
        let loaded = Self::load_messages_from_db(&conn, session_id)?;
        messages.extend(loaded);

        // Create adapter factory, analyzer
        let adapter_factory = AdapterFactory::from_config(&config);
        let analyzer: Box<dyn PromptAnalyzer> = Box::new(
            HeuristicAnalyzer::from_app_config(&config)
                .map_err(|e| SessionError::Storage(e.to_string()))?,
        );
        let bpe = tiktoken_rs::cl100k_base().map_err(|e| SessionError::Storage(e.to_string()))?;

        // Create shared ranking engine
        let ranking_conn = rusqlite::Connection::open(db_path)
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        let ranking_config = config.session.ranking.clone();
        let ranking = std::sync::Arc::new(RankingEngine::new(ranking_conn, ranking_config));

        // Create learning router (wraps ConfigRouter with ranking-based reordering)
        let router: Box<dyn Router> = Box::new(
            LearningRouter::from_app_config(&config, ranking.clone())
                .map_err(|e| SessionError::Storage(e.to_string()))?,
        );

        Ok(Self {
            config,
            messages,
            conn: Mutex::new(conn),
            adapter_factory,
            analyzer,
            router,
            bpe,
            current_model: None,
            total_cost: Cost(0),
            session_id: session_id.to_string(),
            system_prompt,
            last_model_used: None,
            last_tier: None,
            ranking,
            model_info: Mutex::new(HashMap::new()),
            discovery_attempted: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Load messages from the DB for the given session ID.
    fn load_messages_from_db(
        conn: &rusqlite::Connection,
        session_id: &str,
    ) -> Result<Vec<Message>, SessionError> {
        let mut stmt = conn
            .prepare("SELECT message_json FROM messages WHERE session_id = ? ORDER BY id ASC")
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([session_id], |row| {
                let json: String = row.get(0)?;
                Ok(json)
            })
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        let mut messages = Vec::new();
        for row in rows {
            let json = row.map_err(|e| SessionError::Storage(e.to_string()))?;
            let msg: Message =
                serde_json::from_str(&json).map_err(|e| SessionError::Storage(e.to_string()))?;
            messages.push(msg);
        }
        Ok(messages)
    }

    /// Persist a message to the DB.
    fn persist_message(&self, message: &Message) -> Result<(), SessionError> {
        let json =
            serde_json::to_string(message).map_err(|e| SessionError::Storage(e.to_string()))?;
        let conn = self
            .conn
            .lock()
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO messages (session_id, message_json) VALUES (?, ?)",
            rusqlite::params![&self.session_id, &json],
        )
        .map_err(|e| SessionError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Delete all messages for this session from the DB.
    fn clear_db(&self) -> Result<(), SessionError> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| SessionError::Storage(e.to_string()))?;
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?",
            [&self.session_id],
        )
        .map_err(|e| SessionError::Storage(e.to_string()))?;
        Ok(())
    }

    /// Count total tokens in all messages using tiktoken cl100k_base.
    fn count_context_tokens(&self) -> usize {
        self.messages
            .iter()
            .map(|m| match &m.content {
                MessageContent::Text { text } => self.bpe.encode_with_special_tokens(text).len(),
                MessageContent::ToolCall {
                    tool_name,
                    arguments,
                    ..
                } => {
                    let s = format!("{tool_name} {arguments}");
                    self.bpe.encode_with_special_tokens(&s).len()
                }
                MessageContent::ToolResult { result, .. } => self
                    .bpe
                    .encode_with_special_tokens(&result.to_string())
                    .len(),
            })
            .sum()
    }

    /// Apply sliding window context management: drop oldest non-system messages
    /// until token count is within the limit. System prompt is never dropped.
    fn apply_context_window(&mut self) {
        let limit = self.config.session.context_window.max_tokens;
        let mut token_count = self.count_context_tokens();

        while token_count > limit && self.messages.len() > 1 {
            // Find the first non-system message and remove it
            let idx = self.messages.iter().position(|m| m.role != Role::System);
            if let Some(idx) = idx {
                self.messages.remove(idx);
                token_count = self.count_context_tokens();
            } else {
                break;
            }
        }
    }

    /// Ensure the system prompt is the first message.
    fn ensure_system_prompt(&mut self) {
        if self.system_prompt.is_empty() {
            return;
        }
        let has_system = self
            .messages
            .first()
            .is_some_and(|m| m.role == Role::System);
        if !has_system {
            self.messages.insert(
                0,
                Message {
                    role: Role::System,
                    content: MessageContent::Text {
                        text: self.system_prompt.clone(),
                    },
                },
            );
        }
    }

    /// Handle a slash command. Returns the output and whether to quit.
    fn handle_slash_command(
        &mut self,
        cmd: &str,
        args: &str,
    ) -> Result<SessionOutput, SessionError> {
        let events = match cmd {
            "help" => vec![SessionEvent::Text(
                "Commands: /help, /tools, /model <id>, /clear, /budget <tokens>, /cost,\n  /rate [1-5], /rate <model_id> [1-5], /ratings [tier], /reset-ratings [global],\n  /ranking on|off|status, /quit".to_string(),
            )],
            "tools" => vec![SessionEvent::Text(
                "No tools registered. (Phase 4)".to_string(),
            )],
            "model" => {
                let model_str = args.trim();
                if model_str.is_empty() {
                    vec![SessionEvent::Text(format!(
                        "Current model: {}",
                        self.current_model
                            .as_ref()
                            .map(|m| m.as_str())
                            .unwrap_or_else(|| "auto (router-selected)".to_string())
                    ))]
                } else {
                    match ModelId::parse(model_str) {
                        Some(id) => {
                            self.current_model = Some(id.clone());
                            vec![SessionEvent::Text(format!(
                                "Model override set to: {}",
                                id.as_str()
                            ))]
                        }
                        None => vec![SessionEvent::Error(format!(
                            "Invalid model ID: '{model_str}' (expected 'provider/model')"
                        ))],
                    }
                }
            }
            "clear" => {
                self.messages.clear();
                self.total_cost = Cost(0);
                self.current_model = None;
                self.last_model_used = None;
                self.last_tier = None;
                self.clear_db()?;
                vec![SessionEvent::Text("Session cleared.".to_string())]
            }
            "budget" => {
                let tokens = args.trim();
                if tokens.is_empty() {
                    vec![SessionEvent::Text(format!(
                        "Context budget: {} tokens",
                        self.config.session.context_window.max_tokens
                    ))]
                } else {
                    vec![SessionEvent::Text(format!(
                        "Budget set to: {tokens} tokens (not yet enforced)"
                    ))]
                }
            }
            "cost" => {
                vec![SessionEvent::Text(format!(
                    "Session cost: ${:.2} ({} cents)\nBudget limit: ${:.2}",
                    self.total_cost.as_usd(),
                    self.total_cost.0,
                    self.config.session.cost_tracking.hard_limit_usd
                ))]
            }
            "rate" => {
                self.handle_rate_command(args)?
            }
            "ratings" => {
                self.handle_ratings_command(args)?
            }
            "reset-ratings" => {
                self.handle_reset_ratings_command(args)?
            }
            "ranking" => {
                self.handle_ranking_command(args)?
            }
            "quit" => {
                vec![SessionEvent::Text("Goodbye.".to_string())]
            }
            _ => vec![SessionEvent::Error(format!(
                "Unknown command: /{cmd}. Type /help for available commands."
            ))],
        };

        Ok(SessionOutput {
            events,
            usage: None,
            model_used: None,
        })
    }

    /// Handle /rate command.
    fn handle_rate_command(&self, args: &str) -> Result<Vec<SessionEvent>, SessionError> {
        let args = args.trim();
        if args.is_empty() {
            return Ok(vec![SessionEvent::Error(
                "Usage: /rate [1-5] or /rate <model_id> [1-5]".to_string(),
            )]);
        }

        // Two forms: "/rate 5" (rate last response) or "/rate openrouter/llama-3-70b 4"
        if let Some(rating_str) = args.strip_prefix(|c: char| c.is_ascii_digit()) {
            // Just a number: rate the last response
            let _ = rating_str;
            let rating: u32 = args.parse().unwrap_or(0);
            let model = self
                .last_model_used
                .as_ref()
                .ok_or_else(|| SessionError::Storage("no last response to rate".to_string()))?;
            let tier = self.last_tier.unwrap_or(ComplexityTier::Medium);

            self.ranking
                .record_rating(
                    model,
                    tier,
                    crate::core::TaskType::General,
                    rating,
                    None,
                    None,
                    None,
                )
                .map_err(|e| SessionError::Storage(e.to_string()))?;

            return Ok(vec![SessionEvent::Text(format!(
                "Rated {} for {} (tier: {:?})",
                rating,
                model.as_str(),
                tier
            ))]);
        }

        // Model ID + rating
        let parts: Vec<&str> = args.split_whitespace().collect();
        if parts.len() != 2 {
            return Ok(vec![SessionEvent::Error(
                "Usage: /rate <model_id> [1-5]".to_string(),
            )]);
        }

        let model = ModelId::parse(parts[0]).ok_or_else(|| {
            SessionError::Storage(format!(
                "Invalid model ID: '{}' (expected 'provider/model')",
                parts[0]
            ))
        })?;
        let rating: u32 = parts[1].parse().unwrap_or(0);
        let tier = self.last_tier.unwrap_or(ComplexityTier::Medium);

        self.ranking
            .record_rating(
                &model,
                tier,
                crate::core::TaskType::General,
                rating,
                None,
                None,
                None,
            )
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        Ok(vec![SessionEvent::Text(format!(
            "Rated {} for {} (tier: {:?})",
            rating,
            model.as_str(),
            tier
        ))])
    }

    /// Handle /ratings command.
    fn handle_ratings_command(&self, args: &str) -> Result<Vec<SessionEvent>, SessionError> {
        let tier = if args.trim().is_empty() {
            self.last_tier.unwrap_or(ComplexityTier::Medium)
        } else {
            match args.trim().to_lowercase().as_str() {
                "simple" => ComplexityTier::Simple,
                "medium" => ComplexityTier::Medium,
                "complex" => ComplexityTier::Complex,
                "expert" => ComplexityTier::Expert,
                _ => {
                    return Ok(vec![SessionEvent::Error(format!(
                        "Unknown tier: '{args}'. Use: simple, medium, complex, expert"
                    ))]);
                }
            }
        };

        let entries = self
            .ranking
            .get_rankings(tier, crate::core::TaskType::General)
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        if entries.is_empty() {
            let scope = match self.ranking.scope() {
                Scope::Global => "global".to_string(),
                Scope::Project(id) => format!("project:{id}"),
            };
            return Ok(vec![SessionEvent::Text(format!(
                "No ratings for tier {:?} (scope: {scope})",
                tier
            ))]);
        }

        let mut text = format!("Rankings for tier {:?}:\n", tier);
        text.push_str("  Model                          Score   Samples\n");
        text.push_str("  -----                          -----   -------\n");
        for entry in &entries {
            text.push_str(&format!(
                "  {:<30} {:.2}   {}\n",
                entry.model_id, entry.score, entry.sample_count
            ));
        }

        let scope = match self.ranking.scope() {
            Scope::Global => "global".to_string(),
            Scope::Project(id) => format!("project:{id}"),
        };
        text.push_str(&format!("\nScope: {scope}"));

        Ok(vec![SessionEvent::Text(text)])
    }

    /// Handle /reset-ratings command.
    fn handle_reset_ratings_command(&self, args: &str) -> Result<Vec<SessionEvent>, SessionError> {
        let scope = if args.trim() == "global" {
            Scope::Global
        } else {
            self.ranking.scope().clone()
        };

        let deleted = self
            .ranking
            .reset_ratings(&scope)
            .map_err(|e| SessionError::Storage(e.to_string()))?;

        Ok(vec![SessionEvent::Text(format!(
            "Reset {deleted} ratings ({scope_label})",
            scope_label = match &scope {
                Scope::Global => "global scope".to_string(),
                Scope::Project(id) => format!("project:{id}"),
            }
        ))])
    }

    /// Handle /ranking command.
    fn handle_ranking_command(&self, args: &str) -> Result<Vec<SessionEvent>, SessionError> {
        let args = args.trim();
        match args {
            "on" => {
                self.ranking.set_session_enabled(true);
                Ok(vec![SessionEvent::Text(
                    "Ranking enabled for this session.".to_string(),
                )])
            }
            "off" => {
                self.ranking.set_session_enabled(false);
                Ok(vec![SessionEvent::Text(
                    "Ranking disabled for this session.".to_string(),
                )])
            }
            "status" | "" => {
                let enabled = self.ranking.is_enabled();
                let scope = match self.ranking.scope() {
                    Scope::Global => "global".to_string(),
                    Scope::Project(id) => format!("project:{id}"),
                };
                let config_enabled = self.ranking.config().enabled;
                let session_override = if enabled != config_enabled {
                    format!(
                        " (session override: {})",
                        if enabled { "on" } else { "off" }
                    )
                } else {
                    String::new()
                };
                Ok(vec![SessionEvent::Text(format!(
                    "Ranking: {}{session_override}\nScope: {scope}\nMin samples: {}\nExploration: {:.0}%",
                    if enabled { "enabled" } else { "disabled" },
                    self.ranking.min_samples(),
                    self.ranking.exploration_rate() * 100.0,
                ))])
            }
            _ => Ok(vec![SessionEvent::Error(format!(
                "Unknown ranking subcommand: '{args}'. Use: on, off, status"
            ))]),
        }
    }

    /// Check if the budget has been exceeded.
    fn check_budget(&self) -> Result<(), SessionError> {
        if !self.config.session.cost_tracking.enabled {
            return Ok(());
        }
        let limit_cents = Cost::from_usd(self.config.session.cost_tracking.hard_limit_usd).0;
        if self.total_cost.0 >= limit_cents {
            return Err(SessionError::BudgetExceeded(
                self.total_cost.as_usd(),
                self.config.session.cost_tracking.hard_limit_usd,
            ));
        }
        Ok(())
    }
}

#[async_trait]
impl SessionController for Session {
    async fn process(&mut self, input: &str) -> Result<SessionOutput, SessionError> {
        let input = input.trim();

        // Handle slash commands
        if let Some(stripped) = input.strip_prefix('/') {
            let (cmd, args) = stripped.split_once(' ').unwrap_or((stripped, ""));
            return self.handle_slash_command(cmd, args);
        }

        // Check budget before making LLM call
        self.check_budget()?;

        // Ensure system prompt is present
        self.ensure_system_prompt();

        // Add user message
        let user_message = Message {
            role: Role::User,
            content: MessageContent::Text {
                text: input.to_string(),
            },
        };
        self.messages.push(user_message.clone());
        self.persist_message(&user_message)?;

        // Analyze prompt complexity
        let complexity = self
            .analyzer
            .analyze(input)
            .await
            .map_err(|e| SessionError::Adapter(e.to_string()))?;

        // Route to model
        let model_spec = self
            .router
            .route(&complexity, self.current_model.as_ref())
            .await
            .map_err(|e| SessionError::Adapter(e.to_string()))?;

        // Track for /rate command
        self.last_model_used = Some(model_spec.model_id.clone());
        self.last_tier = Some(complexity.tier);

        // Lazily discover model metadata from provider APIs on first call.
        // This populates self.model_info with context window and max output
        // token info so we don't need the user to configure max_tokens.
        if !self
            .discovery_attempted
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            let discovered = self.adapter_factory.discover_all().await;
            if !discovered.is_empty() {
                let mut info = self.model_info.lock().map_err(|e| {
                    SessionError::Storage(e.to_string())
                })?;
                *info = discovered;
            }
        }

        // Get adapter for the selected model, falling back through the chain.
        // This lets users comment out providers in config (e.g. when they
        // don't have an API key) without breaking routing.
        let (adapter, effective_model_id) = self
            .adapter_factory
            .get_adapter_with_fallback(&model_spec.model_id, &model_spec.fallback_chain)
            .map_err(|e| SessionError::Adapter(e.to_string()))?;

        // If we fell back to a different model, update tracking
        if effective_model_id != model_spec.model_id {
            self.last_model_used = Some(effective_model_id.clone());
            tracing::info!(
                primary = %model_spec.model_id,
                effective = %effective_model_id,
                "using fallback model"
            );
        }

        // Determine max_tokens: config override > discovered info > default
        let max_tokens = self
            .config
            .session
            .generation
            .max_tokens
            .unwrap_or_else(|| {
                // Look up discovered info for this model
                let info = self.model_info.lock().unwrap_or_else(|e| e.into_inner());
                let key = effective_model_id.as_str();
                info.get(&key)
                    .and_then(|m| m.max_output_tokens)
                    .map(|v| v as u32)
                    .unwrap_or(4096)
            });

        // Build model config from session.yaml generation settings + discovery
        let model_config = ModelConfig {
            max_tokens,
            temperature: self.config.session.generation.temperature,
            stream: true,
            tools: None,
            response_format: None,
        };

        // Call adapter
        let events = adapter
            .send(&effective_model_id, self.messages.clone(), model_config)
            .await
            .map_err(|e| SessionError::Adapter(e.to_string()))?;

        // Process events: collect text, tool calls, usage
        let mut session_events = Vec::new();
        let mut response_text = String::new();
        let mut usage = None;

        for event in &events {
            match event {
                ResponseEvent::Chunk(text) => {
                    response_text.push_str(text);
                    session_events.push(SessionEvent::Text(text.clone()));
                }
                ResponseEvent::ToolCall(inv) => {
                    session_events.push(SessionEvent::ToolCall(inv.clone()));
                }
                ResponseEvent::Done(u) => {
                    self.total_cost = self.total_cost.add(u.cost);
                    usage = Some(u.clone());
                }
            }
        }

        // Add assistant message
        if !response_text.is_empty() {
            let assistant_message = Message {
                role: Role::Assistant,
                content: MessageContent::Text {
                    text: response_text,
                },
            };
            self.messages.push(assistant_message.clone());
            self.persist_message(&assistant_message)?;
        }

        // Apply context window management
        self.apply_context_window();

        Ok(SessionOutput {
            events: session_events,
            usage,
            model_used: Some(effective_model_id),
        })
    }

    async fn reset(&mut self) {
        self.messages.clear();
        self.total_cost = Cost(0);
        self.current_model = None;
        let _ = self.clear_db();
        self.ensure_system_prompt();
    }

    fn status(&self) -> SessionState {
        let non_system_count = self
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .count();

        SessionState {
            message_count: non_system_count,
            total_cost_cents: self.total_cost.0,
            current_model: self.current_model.as_ref().map(|m| m.as_str()),
            context_tokens_used: self.count_context_tokens(),
            context_token_limit: self.config.session.context_window.max_tokens,
            first_message_role: self.messages.first().map(|m| role_str(&m.role)),
        }
    }
}

fn role_str(role: &Role) -> String {
    match role {
        Role::System => "system".to_string(),
        Role::User => "user".to_string(),
        Role::Assistant => "assistant".to_string(),
        Role::Tool => "tool".to_string(),
    }
}
