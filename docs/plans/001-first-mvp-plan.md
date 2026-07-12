# TurnMux MVP Plan

## Goal

Build the first usable `TurnMux` MVP: a Telegram-controlled session router for terminal coding agents that can start, resume, monitor, and interrupt live `Claude Code` and `Codex CLI` sessions running inside `tmux`.

The product identity for MVP is:

- one Telegram bot
- one Telegram topic/thread per live session
- one `tmux` window per session
- one provider per session (`claude` or `codex`)
- persistent local state so sessions survive bot restarts

This is not yet a generic multi-channel orchestration platform. The first release should be narrow, reliable, and boring enough to run every day.

## Product Definition

### Core user story

While away from the keyboard, a user can:

1. open a Telegram topic
2. start or resume a live coding-agent session for a local repo
3. receive replies and progress updates from the running agent
4. send follow-up messages into the same session
5. interrupt or kill the session if needed
6. return to desktop later and attach to the same `tmux` window

### MVP promise

TurnMux lets a user keep one or more real local CLI-agent sessions alive and operable from Telegram without losing the ability to return to the same terminal session on desktop.

## MVP Scope

### In scope

- Telegram transport only
- Telegram forum topics mode as the primary UX
- `tmux` as the execution substrate
- `claude` and `codex` provider adapters
- start new session in a chosen repo path
- resume a recent session for the chosen provider/path
- forward plain text messages into the live CLI session
- forward non-control slash lines like `/clear` to the underlying CLI as raw input
- monitor provider transcripts/logs and send assistant updates to Telegram
- persistent binding storage
- interrupt and kill controls
- basic history view for the current bound session
- deployable on a developer laptop or VPS with a stable filesystem

### Explicitly out of scope for MVP

- voice messages
- screenshots and terminal image rendering
- inline approval handling for provider permission prompts
- `AskUserQuestion`-style inline keyboards
- filesystem directory browser in Telegram
- support for Discord/Slack/Signal/etc.
- web dashboard
- multi-user RBAC
- provider-agnostic plugin system
- share/export/public transcript publishing
- advanced transcript summarization

## Constraints and Assumptions

### Operational constraints

- TurnMux must not own the source of truth for the session. The terminal session remains the source of truth.
- The user must be able to reattach to the exact same `tmux` window outside TurnMux.
- The bot must survive restarts without losing topic bindings.
- The implementation should be simple enough to debug locally with shell tools.

### Security assumptions

For MVP, TurnMux will not implement rich approval routing for provider permission prompts.

That means deployed sessions must use launch commands that do not block on an interactive approval UI. Examples:

- Claude: a non-interactive permission configuration chosen by the user
- Codex: `--ask-for-approval never` with an explicit sandbox mode chosen by the user

This is a deliberate MVP tradeoff. Permission UX can be added later, but it should not block the first usable release.

### Product assumptions

- Single operator or a very small trusted user list
- One machine running the bot and the CLI sessions
- Telegram supergroup with topics is the default environment
- Repo paths will be provided as text or chosen from a small configured allowlist, not from a Telegram filesystem browser

## MVP Architecture

### High-level model

TurnMux has five layers:

1. `transport`
   Telegram bot updates, topic routing, reply formatting, command parsing
2. `application`
   Session lifecycle orchestration, onboarding flow, provider selection, state transitions
3. `providers`
   `claude` and `codex` adapters for launch, resume discovery, transcript parsing
4. `runtime`
   `tmux` session/window management and raw message injection
5. `state`
   Persistent storage for bindings, offsets, configuration, and monitor progress

### Recommended stack

- Python 3.12+
- `python-telegram-bot` for Telegram transport
- standard-library `sqlite3` for MVP state
- standard-library `subprocess` for `tmux` and provider process interactions
- optional `pydantic` for config/state DTO validation if needed

Python is the right MVP language because:

- `ccbot` already proves the shape in Python
- transcript parsing is file-heavy and easy in Python
- Telegram ecosystem is mature
- shipping one executable service quickly matters more than maximal performance

## State Model

Use `~/.turnmux/` as the runtime home directory.

### Files

- `~/.turnmux/config.toml`
- `~/.turnmux/state.db`
- `~/.turnmux/logs/turnmux.log`

### Core tables

#### `bindings`

Maps Telegram topic to live session.

Suggested columns:

- `id`
- `chat_id`
- `thread_id`
- `provider`
- `repo_path`
- `tmux_session_name`
- `tmux_window_id`
- `tmux_window_name`
- `provider_session_id`
- `transcript_path`
- `status` (`pending_start`, `active`, `stopped`, `missing`)
- `created_at`
- `updated_at`

#### `monitor_offsets`

Tracks how far transcript monitoring has progressed.

Suggested columns:

- `binding_id`
- `byte_offset`
- `last_event_ts`
- `last_message_hash`

#### `pending_launches`

Tracks in-flight startup flows before provider session ID is discovered.

Suggested columns:

- `id`
- `binding_id`
- `provider`
- `repo_path`
- `started_at`
- `discovery_deadline_at`

#### `settings`

Small key/value store for MVP-level config overrides if needed.

## Topic Model

### Primitive

One Telegram topic maps to one TurnMux binding.

### Why this matters

- topic history is the external session list
- no need for separate `/list` UX in MVP
- easy mental model: one topic = one live conversation window
- aligns with the proven `ccbot` pattern

## Provider Adapters

Define a small provider interface up front.

```python
class ProviderAdapter(Protocol):
    name: str

    def build_start_command(self, repo_path: str) -> list[str]: ...
    def list_resumable_sessions(self, repo_path: str) -> list[ProviderSession]: ...
    def build_resume_command(self, session_id: str, repo_path: str) -> list[str]: ...
    def discover_session(self, repo_path: str, started_after: datetime) -> ProviderSession | None: ...
    def parse_new_events(self, transcript_path: Path, offset: int) -> ParseBatch: ...
```

### Claude adapter

MVP responsibilities:

- build start command from configured `CLAUDE_COMMAND`
- inspect Claude session storage under `~/.claude/projects/...`
- optionally use a TurnMux hook installer later, but do not make it a hard MVP dependency
- parse Claude transcript JSONL enough to extract:
  - user messages
  - assistant text
  - thinking text if trivially available
  - local command output when obviously present

### Codex adapter

MVP responsibilities:

- build start command from configured `CODEX_COMMAND`, including `--no-alt-screen`
- inspect `~/.codex/session_index.jsonl` and `~/.codex/sessions/...`
- support resume via `codex resume` or `codex exec resume` selection flows
- parse Codex rollout JSONL enough to extract:
  - assistant commentary messages
  - final/follow-up assistant messages
  - function/tool call summaries if easy
  - shell command output when available

### Important asymmetry

Claude and Codex should share the same app-level lifecycle, but transcript parsing will remain provider-specific.

Do not force a fake common transcript schema too early. Normalize only the minimum needed for Telegram delivery:

- `role`
- `content_type`
- `text`
- `timestamp`
- `is_final`

## tmux Runtime Layer

TurnMux should own one dedicated `tmux` session, default name: `turnmux`.

### Required runtime capabilities

- create window at repo path
- send raw text + Enter
- send Escape / Ctrl-C
- capture pane text for fallback diagnostics
- kill window
- list windows

### MVP execution rule

Each bound topic gets a dedicated `tmux` window. No pane multiplexing in MVP.

## Telegram UX

### Hard MVP commands

- `/start` — usage summary
- `/new` — begin new session onboarding in current topic
- `/resume` — choose a recent session for current topic
- `/status` — show current binding details
- `/history` — show recent message history for current session
- `/interrupt` — send interrupt to active session
- `/kill` — kill bound tmux window and clear binding

### Message behavior

- any plain text message in a bound topic is forwarded to the active CLI session
- any slash command not claimed by TurnMux control commands is forwarded raw to the CLI session

### Onboarding flow

Keep it simple for MVP:

1. user sends `/new`
2. bot asks provider: `claude` or `codex`
3. user sends repo path as text
4. bot asks: `fresh` or `resume`
5. if `resume`, adapter returns a short recent-session list
6. TurnMux creates the window and forwards the first user message after binding is active

Avoid building a Telegram directory browser in MVP.

## Monitoring and Delivery

### Monitor loop

Run a background monitor every 1-2 seconds.

For each active binding:

1. verify `tmux` window still exists
2. verify transcript path still exists
3. read new transcript bytes since last offset
4. parse provider-specific events
5. emit user-visible Telegram updates
6. persist new offset

### Delivery strategy

MVP delivery should optimize for readability, not fidelity.

Send:

- assistant text as plain Telegram messages
- lightweight tool/shell summaries if easy to parse
- optionally collapse noisy repeated updates

Do not attempt token-level streaming in MVP.

### Deduplication

Because transcript polling is eventually consistent, dedupe with:

- byte offsets
- last event timestamp
- last message hash

## Repo Layout Proposal

```text
turnmux/
  docs/
    plans/
      001-first-mvp-plan.md
  src/
    turnmux/
      __init__.py
      main.py
      config.py
      logging.py
      state/
        db.py
        models.py
      runtime/
        tmux.py
      providers/
        base.py
        claude.py
        codex.py
      app/
        bindings.py
        onboarding.py
        monitor.py
        history.py
      transport/
        telegram_bot.py
        commands.py
        formatters.py
  tests/
    providers/
    app/
    transport/
```

## Milestones

### Milestone 0: Bootstrap

Deliverables:

- repo skeleton
- dependency management
- config loading
- logging setup
- state DB creation

Acceptance criteria:

- `turnmux` starts with a config file and creates `~/.turnmux/state.db`
- logs are written predictably

### Milestone 1: tmux + binding core

Deliverables:

- create/list/kill windows
- binding records in SQLite
- manual smoke command to create a bound session row

Acceptance criteria:

- local script can create a `tmux` window for a repo path and persist metadata

### Milestone 2: provider adapters

Deliverables:

- `ClaudeAdapter`
- `CodexAdapter`
- resume discovery
- session discovery after startup

Acceptance criteria:

- local dev script can:
  - list resumable sessions for a repo
  - launch a fresh session
  - resolve the transcript path and provider session ID

### Milestone 3: Telegram transport

Deliverables:

- `/start`, `/new`, `/resume`, `/status`, `/interrupt`, `/kill`
- topic-bound message forwarding
- onboarding state machine

Acceptance criteria:

- a user can create a topic, bind it to a live session, and send messages successfully

### Milestone 4: monitor loop

Deliverables:

- transcript polling
- provider event parsing
- Telegram delivery
- dedupe logic

Acceptance criteria:

- assistant responses appear in Telegram with no duplicate spam
- process survives restart and resumes from stored offset

### Milestone 5: history and recovery

Deliverables:

- `/history`
- startup reconciliation for missing windows/transcripts
- cleanup behavior on killed windows

Acceptance criteria:

- bot restart does not orphan all active topics
- `/history` returns useful recent context

## Suggested MVP Cutline

The first public MVP should ship only after these flows work end-to-end:

1. new Codex session from Telegram topic
2. resume Codex session from Telegram topic
3. new Claude session from Telegram topic
4. resume Claude session from Telegram topic
5. interrupt active session
6. kill active session
7. restart bot and continue monitoring existing binding

If a feature does not help these flows, it should not block release.

## Testing Strategy

### Unit tests

- config parsing
- state transitions for bindings
- tmux command building
- transcript parser fixtures for Claude
- transcript parser fixtures for Codex
- dedupe logic for monitor batches

### Integration tests

- create binding -> create window -> persist binding
- simulated transcript growth updates offset correctly
- control commands route to the correct topic binding

### Manual tests

- create two topics, one Claude and one Codex
- run both in parallel
- verify messages stay in the correct topic
- interrupt one without affecting the other
- kill one topic and confirm the other continues
- restart bot process and verify active bindings recover

## Known Risks

### Codex session discovery ambiguity

Codex does not expose the same explicit startup hook model as Claude. Fresh-session discovery may require heuristics using recent transcript files, cwd matching, and start time.

Mitigation:

- TurnMux owns session launch timing
- discovery window is narrow
- startup remains one pending launch per topic
- discovery failures surface clearly and do not silently bind the wrong session

### Provider transcript drift

Both providers may evolve transcript formats over time.

Mitigation:

- keep parsing logic isolated per adapter
- store fixture transcripts in tests
- parse only fields required for MVP delivery

### Telegram command collisions

Some provider slash commands may collide with TurnMux control commands.

Mitigation:

- keep TurnMux control commands few and explicit
- everything else forwards raw to the active session

### Non-interactive permission requirement

Without approval routing, some users will launch providers in an unsafe mode.

Mitigation:

- document this clearly as an MVP constraint
- keep approval UX as the first post-MVP roadmap item

## Post-MVP Priorities

In order:

1. provider approval/prompt routing UI
2. filesystem browser / configured workspace picker
3. screenshots and pane rendering
4. richer history pagination
5. export/share transcript bundles
6. additional transports beyond Telegram

## Implementation Order Recommendation

Build in this order:

1. state + config
2. tmux runtime
3. Codex adapter
4. Claude adapter
5. Telegram transport
6. monitor loop
7. recovery + history

The reason to start with Codex before Claude is practical:

- Codex has the trickier discovery model
- if Codex works, the adapter boundary is probably correct
- Claude can then land as the cleaner second adapter

## MVP Exit Criteria

TurnMux MVP is done when:

- it can reliably control both providers from Telegram topics
- it keeps the terminal as the source of truth
- it survives restarts without losing active bindings
- the user can return to desktop and attach to the same `tmux` window
- there are no critical ambiguities about which Telegram topic maps to which live session
