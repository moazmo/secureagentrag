# 12 — Agent Handoff Contract

This document is the **operating contract** for any AI agent that continues this launch — Claude Code, Kilo CLI, Hermes Agent, Antigravity Agent, GPT, Gemini, or a fresh Claude session. Read this first.

## Identity rules

- **The repo owner is Moaz Muhammad** (`moazmo27@gmail.com`, GitHub `moazmo`)
- Every commit must be authored by the owner
- **No AI co-author footers in commits or PRs.** Do not add `Co-Authored-By: Claude`, `Co-Authored-By: Hermes`, `Co-Authored-By: <any AI>`, or any equivalent
- No "Generated with [AI Tool]" lines in commit bodies

## Active branch

```
deploy/prod-launch
```

If you find yourself on `main`, switch:

```bash
git checkout deploy/prod-launch
```

`main` is frozen at `56c8c98`. **Do not push to `main` directly.** All work merges via PR after phase 7 smoke green.

## State of the launch on arrival

Run these commands first:

```bash
git status                              # confirm clean working tree
git log --oneline -10                   # confirm last commit
ls launch-plan/                         # confirm plan files present
cat launch-plan/README.md               # read the index
```

Then read in order:

1. `launch-plan/00-context.md`
2. `launch-plan/01-stack-decisions.md`
3. `launch-plan/02-smoke-tests.md`
4. The phase file matching the next pending phase (see `private/roadmap.md` for state)
5. `launch-plan/11-security-checklist.md` if your phase touches keys, audit, or session code

## Where you are in the build

Check `private/roadmap.md` § "Production launch progress" — it has a checkbox grid showing which phases shipped.

If `private/roadmap.md` is unavailable (you do not have read access), assume the launch is at **phase 1 (smoke tests)** and ask the owner to confirm.

## What you may do

- Edit any file under `interfaces/`, `inference/`, `retrieval/`, `core/`, `config/`, `utils/`, `evaluation/`, `tests/`, `scripts/`, `launch-plan/`, `app/` on `deploy/prod-launch`
- Create new files anywhere they logically belong
- Run `uv run pytest` and `uv run ruff` freely
- Read `.env` and `INTERVIEW_DEFENSE.md` and `CV_BLURB.md` and `private/roadmap.md` (the owner has granted explicit access)
- Update **all** `.md` files, public and private, when they become stale
- Commit on `deploy/prod-launch` after acceptance criteria for a phase are met
- Push to `origin/deploy/prod-launch`

## What you may NOT do

- **Push to `main`** — that branch is frozen until launch complete
- **Modify `audit_logs/*.jsonl`** — hash chain integrity
- **Modify `uv.lock` outside `uv` commands** — would break the lockfile
- **Modify `data/checkpoints.sqlite`** — LangGraph state, owner-only
- **Modify `data/checkpoints/reranker-domain-v1/`** — gitignored model weights, owner-only
- **Log raw API keys anywhere** — even for debugging
- **Add a `Co-Authored-By` footer to any commit**
- **Spend the owner's money** — every service in the stack must remain on the free tier. If a phase requires payment, halt and escalate.
- **Skip the smoke tests in phase 1** — those are the gate that proves the free tier still accepts us. Free tiers change; smoke first.

## What to do when blocked

1. Document the blocker in `private/roadmap.md` under "Active blockers"
2. Halt the phase
3. Write a one-paragraph summary of what is blocked, why, and what input is needed from the owner
4. Wait for owner response — do not work around with paid services or different architectures without explicit approval

## Commit message style

The repo follows Conventional Commits with a flat structure:

```
feat(<area>): short description

Longer body explaining why.
```

Examples seen on `main`:

- `feat(retrieval): cache per-tenant QdrantManagers + pin sparse isolation`
- `refactor(ui): slim app/views/chat.py 621 -> 161 LOC`
- `docs: refresh AGENTS / CLAUDE / RUNBOOK after P1 + P2`

For the launch branch, use these areas:

- `deploy:` — anything that touches `Dockerfile.hf`, HF Space config, Vercel config, GitHub Actions
- `feat(byok):` — BYOK mode features in the backend
- `feat(web):` — Next.js frontend (in the sibling repo)
- `feat(landing):` — Hostinger static landing
- `docs(launch):` — updates to `launch-plan/`

## Test discipline

- The repo's existing 487 tests must continue to pass on `deploy/prod-launch`
- Every new feature must add tests
- BYOK + session + key redaction tests are mandatory (see `11-security-checklist.md`)
- Run `uv run pytest -q` before every commit. Halt if any test fails.

## Lint and format

```bash
uv run ruff check .
uv run ruff format .
```

Both must be clean before commit.

## When the launch is done

Phase 9 ends with merging `deploy/prod-launch` into `main`. The merge commit message:

```
feat: production launch (P6) — Next.js + HF Space + Qdrant Cloud + Vercel

Live demo: https://secureagentrag.vercel.app
Backend:   https://LeomordKaly-secureagentrag-api.hf.space
Landing:   https://<hostinger-url>/

Closes P6 from private/roadmap.md.
```

After merge:

- Delete `deploy/prod-launch` from origin and locally
- Update `private/roadmap.md` to mark P6 as DONE with the merge commit hash
- Tag the merge: `git tag v1.0.0-launch && git push --tags`

## Final note

The owner has read every file in this plan. The owner has accepted the stack and the build order. The owner has explicitly granted access to private files for the duration of this launch. **The owner's intent is unambiguous: ship a public, recruiter-quality, $0-cost demo without compromising the security primitives that make this project interesting.** When in doubt, optimize for that intent.
