# 09 — Phase 8 (Pre-launch) and Phase 9 (Post-launch) Doc Sweep

Two sweeps run during this launch:

- **Pre-launch sweep** (right now, as part of this commit) — updates **every** existing `.md` file to point at the launch plan and the new branch
- **Post-launch sweep** (after phase 7 smoke passes) — flips every doc from "in progress" to "shipped" and adds the new ADRs

## Pre-launch sweep — checklist

The AI agent runs through this BEFORE writing any code. Each file gets either:

1. A "Production launch in progress on `deploy/prod-launch`" notice at the top
2. A link to `launch-plan/README.md`
3. No other changes — the doc's existing content stays accurate for `main`

### Public docs

- [ ] `README.md` — add a banner above the existing content: "Production launch in progress. Plan at `launch-plan/`."
- [ ] `CLAUDE.md` — add "Active branch: `deploy/prod-launch`" + "Read `launch-plan/12-agent-handoff.md` before continuing"
- [ ] `AGENTS.md` — same as CLAUDE.md
- [ ] `RUNBOOK.md` — note that production deploy procedures are drafted in `launch-plan/`; live procedures will land in `RUNBOOK.md` § Production after launch
- [ ] `DECISIONS.md` — no edits yet. ADRs 025–028 are drafted in `launch-plan/draft-adrs/`. They flip to the main `DECISIONS.md` only after the corresponding phase ships.
- [ ] `architecture.md` — add a "Production topology (in progress)" appendix linking to `launch-plan/01-stack-decisions.md`
- [ ] `.env.example` — add commented-out lines for the new BYOK env vars (no values, just names + comments)

### Private docs

- [ ] `private/roadmap.md` — flip P6 status from "next phase, owner-driven" to "**in progress** on `deploy/prod-launch`, plan locked 2026-05-25"
- [ ] `INTERVIEW_DEFENSE.md` — add a new defense entry: "Why HF Spaces?" with the no-CC + Egypt + no-cold-start argument
- [ ] `CV_BLURB.md` — flag for update post-launch; no edit now (we wait for the live URL)

## Post-launch sweep — checklist (run after phase 7 smoke green)

### Public docs

- [ ] `README.md`:
  - Replace the "in progress" banner with the live demo URL + landing URL
  - Add screenshots
  - Add the demo GIF
  - Update the "Quick start" section with two paths: "Try the public demo" and "Run locally"
- [ ] `CLAUDE.md`:
  - Add a "Production deployment" section
  - Update the file/dir map with new files (`Dockerfile.hf`, `interfaces/byok.py`, `retrieval/session_purge.py`, `landing/`, etc.)
  - Bump LOC + test counts
- [ ] `AGENTS.md`:
  - Same updates as CLAUDE.md
  - Add "Production failure modes" section linking to RUNBOOK
- [ ] `RUNBOOK.md`:
  - New § "Production failure modes":
    - HF Space sleeping
    - HF Space build failing
    - Qdrant Cloud cluster down
    - Vercel cold edge function
    - Groq rate limit exceeded
    - Hostinger landing page 500
    - BYOK key leak detection
  - Each with detection, escalation, and recovery
- [ ] `DECISIONS.md`:
  - Add ADR-025 (BYOK demo mode)
  - Add ADR-026 (HF Spaces as backend host)
  - Add ADR-027 (Frontend split to Vercel + Next.js)
  - Add ADR-028 (Qdrant Cloud + session collections)
- [ ] `architecture.md`:
  - Move "Production topology" out of appendix into a top-level section
  - Update sequence diagrams for the BYOK request path
- [ ] `.env.example`:
  - All BYOK + CORS vars uncommented with placeholder values

### Private docs

- [ ] `private/roadmap.md`:
  - Flip P6 to "**DONE**" with final commit hashes
  - Add a post-launch section: "What's next?" (community feedback loop, blog post, HN launch, etc.)
- [ ] `INTERVIEW_DEFENSE.md`:
  - Add the live demo URL
  - Add talking points about the BYOK security model
  - Add the recruiter-targeted argument for Next.js over Streamlit
- [ ] `CV_BLURB.md`:
  - Update with live demo URL
  - Refresh tech stack tags

## Verification

After the post-launch sweep:

```bash
# All .md files must be self-consistent — no stale "in progress" banners
grep -rl "in progress" --include="*.md" .

# Every link in README must resolve
# (manual check)

# No reference to old Streamlit-only architecture in production-relevant docs
grep -ri "streamlit" README.md CLAUDE.md AGENTS.md RUNBOOK.md
# Should only appear in historical/migration context
```

## Files we explicitly do NOT touch

- `audit_logs/*.jsonl` — hash chain, owner-only
- `data/checkpoints.sqlite` — LangGraph state
- `uv.lock` — only via `uv` commands
- `data/checkpoints/reranker-domain-v1/` — gitignored model weights
- `evaluation/calibration.json` — produced by `scripts/calibrate_thresholds.py`
- `evaluation/baseline.json` — produced by the nightly job
- Test fixtures in `tests/test_*` — only edited when behavior under test changes
