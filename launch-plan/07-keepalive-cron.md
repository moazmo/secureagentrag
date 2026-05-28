# 07 — Phase 6: GitHub Actions Keepalive Cron

**Owner of this phase:** AI agent.
**Pre-requisite:** phase 3 (HF Space) live.

## Goal

A free GitHub Actions workflow that pings the HF Space `/health` endpoint every 24 hours, keeping it out of the 48-hour idle sleep state. Net effect: the Space is always warm for real visitors.

## File: `.github/workflows/keepalive.yml`

```yaml
name: keepalive

on:
  schedule:
    # Every day at 03:17 UTC. Avoid round numbers — those run minutes are
    # heavily contended by other repos and GitHub schedules suffer delays.
    - cron: '17 3 * * *'
  workflow_dispatch:                          # allow manual trigger

permissions:
  contents: read

jobs:
  ping:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Ping HF Space
        run: |
          curl --fail --silent --show-error \
               --max-time 60 \
               --retry 3 --retry-delay 10 \
               https://LeomordKaly-secureagentrag-api.hf.space/health
      
      - name: Ping Vercel frontend
        run: |
          curl --fail --silent --show-error \
               --max-time 30 \
               --retry 3 --retry-delay 5 \
               https://app.eilm.live/
```

## Budget

- 2 curls × ~30 s budget × 365 days = ~6 hours/year of GitHub Actions runtime
- Free quota: 2000 min/month = 24,000 min/year
- **Usage: 0.025% of quota.** Free forever.

## Failure handling

If either ping fails:

- The workflow exits non-zero
- GitHub sends an email notification to the owner
- The workflow's history page shows the failure with curl's error message

Owner can manually trigger via the Actions tab "Run workflow" button to test recovery.

## Why not Cron-job.org / similar third-party

- Third-party crons require yet another account
- GitHub Actions is already integrated with the repo
- Free quota is massive
- Failure notifications go to the same email as other repo notifications

## Acceptance criteria

- [ ] Workflow file checked in at `.github/workflows/keepalive.yml`
- [ ] Manual workflow_dispatch run succeeds
- [ ] Scheduled run succeeds within 24 hours of the merge
- [ ] HF Space sleep status verified after 49 hours of no human traffic (should still be "Running", not "Sleeping")
