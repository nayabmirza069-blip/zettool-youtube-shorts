# ZetTool YouTube Shorts (GitHub Actions)

Auto-generates and uploads 1 Short per scheduled run (2x/day IST: 08:00 and 20:00),
running on GitHub servers - no PC needed. The PC can be off and videos still post.

## What happens on every run
1. Script built (Gemini, or SEO template fallback if quota is out)
2. Edge-TTS voice (en-GB-RyanNeural)
3. Pexels background footage + lower-third captions -> 1080x1920 Short (ffmpeg)
4. Uploaded to the connected YouTube channel (public) via YouTube Data API

## Required GitHub Secrets (Settings -> Secrets and variables -> Actions)
| Secret | Value |
|---|---|
| `SITE_CONFIG` | Full contents of `config.json` (holds Gemini key + site) |
| `YT_CONFIG` | Full contents of `youtube_config.json` (holds Pexels key + topics + privacy) |
| `YT_CLIENT_SECRET` | base64 of `client_secret.json` (Google OAuth desktop client) |
| `YT_TOKEN` | base64 of `youtube_token.json` (Google OAuth refresh token, scope youtube.upload) |

Generate the base64 values on Windows:
```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\Users\hp\zettool-agent\client_secret.json"))
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\Users\hp\zettool-agent\youtube_token.json"))
```

## Topic picking
Runs are stateless (no persistent disk), so topics rotate deterministically by date
across the seed_topic pool - no repeats until the whole list cycles.

## Local dev
```powershell
python youtube_agent.py --no-upload --force   # render only, no upload
```
Windows runs still use the local `_youtube_state.json`; set `YT_DETERMINISTIC_TOPIC=1`
to replicate the GitHub topic rotation locally.