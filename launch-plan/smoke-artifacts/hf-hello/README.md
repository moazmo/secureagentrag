---
title: SecureAgentRAG API (smoke)
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Phase 1 smoke test for the SecureAgentRAG production launch
---

# SecureAgentRAG API — Phase 1 Smoke

Hello-world FastAPI proving HF Docker SDK build + CPU Basic free tier + port 7860 reachability from Egypt.

Real backend lands in Phase 2 — see [launch plan](https://github.com/moazmo/secureagentrag/blob/deploy/prod-launch/launch-plan/03-backend-byok.md).

## Endpoints

- `GET /` — service info + uptime
- `GET /health` — health probe (used by the GitHub Actions keepalive cron)

## Source

[github.com/moazmo/secureagentrag](https://github.com/moazmo/secureagentrag) — branch `deploy/prod-launch`.
