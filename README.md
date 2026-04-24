# TurnMux

TurnMux binds a Telegram topic or private chat to a real local `tmux` window running Claude Code, Codex CLI, or OpenCode. It keeps orchestration state on the machine, resumes sessions from provider-native local transcripts, and does not depend on a TurnMux-managed backend.

## Requirements

- macOS. This is the only currently supported and tested target.
- Python 3.12+
- `tmux`
- a Telegram bot token
- at least one installed and authenticated provider CLI:
  - Claude Code: [Anthropic CLI docs](https://code.claude.com/docs/en/cli-reference)
  - Codex CLI: [OpenAI Codex CLI docs](https://developers.openai.com/codex/cli)
  - OpenCode: [OpenCode CLI docs](https://opencode.ai/docs/cli/)

Linux or WSL may work only as a best-effort foreground run if `tmux`, Python, Telegram networking, and the provider CLI all behave the same way. Native Windows is not supported.

## Install

Clone the repo, create a virtual environment, activate it, and install TurnMux from this checkout:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install .
```

This is the supported first-time install path.

## Quickstart

1. Generate a starter config:

   ```bash
   turnmux init-config
   ```

2. Edit `~/.turnmux/config.toml`:
   - replace `telegram_bot_token` and `allowed_user_ids`
   - keep `allowed_roots` narrow
   - configure at least one provider command

3. Bootstrap the runtime state:

   ```bash
   turnmux bootstrap
   ```

4. Validate the repo you want to expose:

   ```bash
   turnmux doctor --repo /absolute/path/to/repo
   ```

   Fix every `[error]` before continuing. The sample config from `turnmux init-config` intentionally fails `doctor` until you replace the placeholder token and user id.

5. Start TurnMux:

   Foreground:
   ```bash
   turnmux run
   ```

   macOS `launchd` service:
   ```bash
   turnmux service install
   turnmux service status
   ```

`turnmux init-config` writes a sample config with detected local defaults and comments out provider blocks it cannot find. `turnmux service status` reports the actual runtime home, heartbeat age, and launchd health for the installed service. The built-in `turnmux service ...` commands are macOS-only.

Contributions that harden Linux, WSL, or Windows support are welcome, but they are not part of the supported path today.

## Config Notes

Configure at least one provider. Common defaults:

```toml
claude_command = [
  "claude",
  "--dangerously-skip-permissions",
]

codex_command = [
  "codex",
  "--ask-for-approval",
  "on-request",
  "--sandbox",
  "danger-full-access",
  "--no-alt-screen",
]

# Optional: OpenCode provider.
# opencode_command = ["opencode"]
# opencode_model = "provider/model"

# Optional: relay Claude thinking blocks back into Telegram and /history.
# relay_claude_thinking = true

# Optional: voice/audio transcription via OpenAI.
# openai_api_key = "sk-..."
# openai_base_url = "https://api.openai.com/v1"
# openai_transcription_model = "gpt-4o-transcribe"
```

Notes:

- `allowed_roots` must contain absolute directories. TurnMux rejects repo paths outside them.
- TurnMux keeps runtime state in `~/.turnmux/`, including `config.toml`, `logs/turnmux.log`, `state.db`, and `heartbeat.json`.
- Telegram needs plain text delivery. Either disable privacy mode in `@BotFather` or add the bot as an admin in the target supergroup.
- Forum supergroups with topics are the best fit for parallel sessions. Private chats work too, but only support one live binding at a time.
- On first launch, TurnMux marks supported workspaces as trusted in local provider state so `tmux` startup can stay non-interactive.
- Claude thinking stays local by default. Set `relay_claude_thinking = true` only if you want it mirrored into Telegram and `/history`.
- Voice and audio messages are optional. If `openai_api_key` is configured, TurnMux transcribes them through the configured OpenAI-compatible `/audio/transcriptions` endpoint before forwarding the text to the CLI.
- Fresh bindings can remain in `pending_start` until the first prompt reaches the provider. That is normal.
- Telegram should be treated as a high-trust control surface. Do not paste raw API keys or other secrets into shared topics.

## Telegram Commands

- `/new` starts setup for a fresh session
- `/resume` resumes an existing provider session in the current topic
- `/status` shows the current binding
- `/history` prints recent transcript history
- `/interrupt` sends `Ctrl-C`
- `/kill` kills the bound `tmux` window
- `/cancel` cancels the current setup flow

## Updating

Use the same checkout and virtual environment you installed from:

```bash
git pull --ff-only
. .venv/bin/activate
python -m pip install --upgrade .
```

If you use the macOS service, restart it after updating:

```bash
turnmux service restart
```

If you run TurnMux in the foreground instead, stop the current process and start `turnmux run` again after updating.

## Troubleshooting

- Plain text commands do not arrive in Telegram topics: disable bot privacy mode or make the bot an admin in the supergroup.
- A repo is rejected at launch time: confirm it is inside `allowed_roots`, then rerun `turnmux doctor --repo /absolute/path/to/repo`.
- `turnmux doctor` warns that the service runtime home differs from this invocation: `turnmux service status` will show the runtime home and config that launchd is actually using. Restart or reinstall the service from the checkout and virtual environment you want it to follow.
- `turnmux service status` shows a missing or stale heartbeat: inspect the `logs dir` from the status output, then run `turnmux service restart`.
- The checkout updated but behavior did not: TurnMux does not hot-reload. Restart the process, and if you installed with `pip install .`, reinstall with `python -m pip install --upgrade .`.

## Development

If you are editing TurnMux itself, use an editable install instead:

```bash
python -m pip install -e .
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Security

See [SECURITY.md](SECURITY.md) for the trust model and deployment guidance.

## License

MIT. See [LICENSE](LICENSE).
