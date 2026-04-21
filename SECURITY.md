# Security Model

TurnMux is a high-trust local control plane for your own machine. It is designed for single-user or tightly controlled use, not for zero-trust or multi-tenant operation.

## What TurnMux Does

- launches local provider CLIs inside `tmux`
- reads local provider transcript/state files to resume sessions and mirror updates
- sends and receives control/data through the Telegram Bot API
- stores orchestration metadata in `~/.turnmux/state.db`
- can auto-mark local workspaces as trusted for supported providers so remote control stays non-interactive
- can optionally mirror Claude thinking blocks into Telegram if `relay_claude_thinking = true`

## What TurnMux Does Not Do

- it does not proxy provider API keys through its own backend
- by default it talks to the Telegram Bot API; if optional voice/audio transcription is enabled, it also sends media to the configured OpenAI-compatible transcription endpoint using your configured API key
- it does not upload your repositories to a TurnMux-managed service

## Main Trust Assumptions

- anyone who can read the bound Telegram topic may see prompts, assistant output, repo paths, and selected tool errors
- if `relay_claude_thinking = true`, Telegram readers may also see Claude thinking blocks
- if voice/audio transcription is enabled, those media attachments leave the machine and are sent to the configured transcription provider
- anyone who can control the bot for an allowed topic can drive a trusted local coding agent
- provider-specific trust settings may be modified locally to avoid interactive startup prompts

## Recommendations

- use TurnMux only in a dedicated private chat or a supergroup you control
- do not paste raw API keys or other secrets into Telegram topics
- keep `allowed_roots` narrow
- review provider command flags before enabling remote editing
- treat the machine running TurnMux as security-sensitive
