# Model Effectiveness Ranking -- Design Document

## 1. Overview

A learning layer on top of the existing config-driven router. The harness
collects explicit user ratings (1-5) for model responses, groups them by
complexity tier and task type, and uses Bayesian-smoothed scores to reorder
fallback chains. The user can toggle ranking on/off per session and in config.

## 2. Requirements

- Which model answered is visible to the user after the response
- User can rate responses: `/rate [1-5]`
- User can show rankings: `/ratings`
- User can reset ratings: `/reset-ratings` (global or per-project)
- User can set ratings manually: `/rate openrouter/llama-3-70b 4`
- User can toggle ranking: `/ranking on|off`
- Config can disable ranking globally: `session.ranking.enabled: false`
- Ratings are scoped: global (all projects) or per-project (working directory)
- Ranking is off by default until the user opts in

## 3. Architecture Changes

### 3.1 Adapter Factory

Currently Session holds a single `Box<dyn LLMAdapter>` (MockAdapter). This
must become a factory that maps provider name to adapter instance, so that
when the router selects `openrouter/llama-3-70b`, the session knows which
adapter to call.

```
AdapterFactory
  ├── "mock" -> MockAdapter
  ├── "openai" -> OpenAiCompatAdapter (base_url from config)
  ├── "openrouter" -> OpenAiCompatAdapter (base_url from config)
  └── "anthropic" -> AnthropicAdapter (Phase 6+ / providers feature)
```

The factory is built from `config.providers`. Each provider config has
`base_url` and `api_key_env`. The adapter uses `reqwest` for HTTP.

### 3.2 Model Visibility in SessionOutput

`SessionOutput` gains a `model_used: Option<ModelId>` field. After the
adapter responds, the session sets this field. The CLI prints it after
the response text:

```
> debug this architecture
[response text...]

[model: openrouter/anthropic/claude-3.5-sonnet | cost: $0.03 | tokens: 450/120]
```

### 3.3 Ranking Engine (src/core/ranking.rs)

```
RankingEngine
  ├── store: RankingStore (SQLite)
  ├── enabled: bool
  ├── project_scope: Option<String> (working dir hash, or None for global)
  ├── min_samples: u32 (default 3, cold-start threshold)
  ├── prior: f64 (default 3.0, Bayesian prior)
  ├── decay: f64 (default 0.95, per-day recency decay)
  └── exploration_rate: f64 (default 0.1, epsilon-greedy)
```

Methods:
- `record_rating(model_id, tier, task_type, rating, cost, tokens)` -> persists
- `get_score(model_id, tier, task_type) -> f64` (Bayesian-smoothed)
- `get_rankings(tier, task_type) -> Vec<(ModelId, f64, u32)>` (sorted)
- `reorder_chain(models: Vec<ModelId>, tier, task_type) -> Vec<ModelId>`
- `reset_ratings(scope: Scope)` -> deletes ratings
- `set_rating(model_id, tier, rating)` -> manual override
- `is_enabled() -> bool`

### 3.4 Learning Router (src/core/router.rs)

Wraps `ConfigRouter`. When ranking is enabled:

1. ConfigRouter produces ModelSpec (config order)
2. LearningRouter calls `ranking.reorder_chain(spec.fallback_chain, tier, task_type)`
3. The primary model is also subject to reordering
4. If exploration_rate triggers (10% chance), pick a random model from the
   chain that has < min_samples ratings (exploration)
5. Log the reordering decision via tracing

### 3.5 Task Type Extension

The classifier (Phase 5) already classifies tier. Extend `ClassificationResponse`
to include `task_type: TaskType`:

```
enum TaskType {
    Code,
    Creative,
    Analysis,
    Qa,
    Translation,
    General,
}
```

The heuristic analyzer defaults to `General`. The classifier model returns
both tier and task_type. Rankings are stored per (tier, task_type) pair.

### 3.6 SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS model_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    tier TEXT NOT NULL,
    task_type TEXT NOT NULL DEFAULT 'general',
    rating INTEGER NOT NULL,        -- 1-5
    cost_cents INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    project_scope TEXT,             -- NULL = global, else working dir hash
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_ratings_lookup
    ON model_ratings(model_id, tier, task_type, project_scope);
```

### 3.7 Configuration

`config/session.yaml` additions:

```yaml
session:
  ranking:
    enabled: false          # off by default
    min_samples: 3          # cold-start threshold
    prior: 3.0              # Bayesian prior (neutral)
    decay: 0.95             # per-day recency decay
    exploration_rate: 0.1   # epsilon-greedy exploration
    scope: "project"        # "global" or "project"
```

### 3.8 Slash Commands

| Command | Description |
|---------|-------------|
| `/rate [1-5]` | Rate the last response |
| `/rate <model_id> [1-5]` | Manually rate a specific model |
| `/ratings` | Show ranking table for current tier+task_type |
| `/ratings <tier>` | Show rankings for a specific tier |
| `/reset-ratings` | Reset ratings for current scope |
| `/reset-ratings global` | Reset all ratings globally |
| `/ranking on` | Enable ranking for this session |
| `/ranking off` | Disable ranking for this session |
| `/ranking status` | Show current ranking status |

## 4. Scoring Formula

```
score(model, tier, task_type) =
    (SUM(rating_i * decay^(days_since_i)))
    /
    (SUM(decay^(days_since_i)) + prior_weight)

where:
    prior_weight = prior / max_rating * sample_count_threshold
    decay = 0.95 per day
    prior = 3.0 (on a 1-5 scale)
```

A model with no ratings gets score = prior (3.0).
A model with 10 ratings averaging 4.5 gets score ~= 4.3 (pulled toward
prior by the Bayesian smoothing, but converges with more data).

## 5. Cold Start

- New models start at prior (3.0)
- Need >= min_samples (default 3) ratings before influencing routing
- Until then, config order is preserved
- Exploration: 10% of the time, pick a random under-sampled model from
  the chain to build ratings

## 6. Scope

- `global`: all ratings stored without project_scope, shared across all projects
- `project`: ratings stored with project_scope = hash(cwd), per-project
- `/reset-ratings` resets current scope only
- `/reset-ratings global` resets everything
- Config `scope` sets the default; `/ranking scope global|project` can override per session

## 7. Implementation Phases

### Phase A: Adapter Factory + Model Visibility
- AdapterFactory that maps provider -> adapter
- ModelConfig gains `model: String` field
- SessionOutput gains `model_used: Option<ModelId>`
- CLI prints model after response
- MockAdapter still works; OpenAiCompatAdapter stub (returns error)

### Phase B: Ranking Engine + Feedback
- RankingEngine + RankingStore (SQLite)
- `/rate`, `/ratings`, `/reset-ratings`, `/ranking` commands
- Config schema for ranking section
- Rankings visible to user

### Phase C: Learning Router
- LearningRouter wraps ConfigRouter
- Reorders fallback chain based on rankings
- Epsilon-greedy exploration
- Tracing logs for reordering decisions

### Phase D: Task Type Extension
- TaskType enum
- Classifier returns task_type
- Rankings keyed by (tier, task_type)
- Heuristic analyzer defaults to General

### Phase E: OpenRouter Adapter
- OpenAiCompatAdapter implementation (behind `providers` feature)
- SSE streaming parsing
- Function-calling normalization
- Config for OpenRouter provider
- End-to-end test with real OpenRouter API (manual)

## 8. Non-Goals (for now)

- Implicit feedback (rephrasing detection, session continuation signals)
- Cost-efficiency scoring (quality-only for v1)
- Model version tracking
- Per-user normalization (single-user CLI tool)
- Web UI for ratings dashboard
