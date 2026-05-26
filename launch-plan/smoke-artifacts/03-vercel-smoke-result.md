# Phase 1d Smoke — Vercel (PASS)

**Run date:** 2026-05-26
**Run by:** Claude (agent), token supplied by owner

## Outcome

| Check | Result |
|---|---|
| Account signup (no CC required) | ✅ Owner signed up with GitHub OAuth (account `moazmo`) |
| Hobby plan auto-assigned | ✅ `billing.plan: "hobby"`, `billing.status: "active"` |
| Email matches | ✅ `moazmo27@gmail.com` |
| Access token created | ✅ `vcp_5J99…fm1x4OzL1L` saved as `VERCEL_TOKEN` in `.env` |
| `GET /v2/user` from Egypt | ✅ HTTP 200 |
| Deployment created via REST (no CLI, no git push) | ✅ `dpl_HkGUbbL6Wfpb5fTwGKwdwBPQx6BL` (`secureagentrag-smoke`) |
| Build → READY | <5 s (static HTML) |
| Production alias claimed | ✅ `secureagentrag-smoke.vercel.app` |
| Egypt → Vercel TTFB | 0.70 s, HTTP 200 (`curl -w "%{time_starttransfer}"`) |
| TLS handshake from Egypt | 0.37 s |

## Live URL

https://secureagentrag-smoke.vercel.app — static placeholder linking to the HF Space backend (phase 1b) and the GitHub repo.

## API surface used

```
POST /v13/deployments
  Authorization: Bearer <token>
  body: {
    "name": "...",
    "project": "...",
    "target": "production",
    "files": [{"file": "index.html", "data": "<inline HTML>"}],
    "projectSettings": {"framework": null}
  }

GET  /v13/deployments/{deployment_id}     # poll readyState until READY
GET  /v2/user                              # whoami
```

Authorization on all of them: `Authorization: Bearer <vercel_token>`.

The full Next.js frontend (phase 4) will use this same API plus `vercel build` locally + `POST /v13/deployments` with `target=production`, OR the standard Git integration (point a repo at the project for auto-deploy on push).

## What this proves

1. **Vercel Hobby plan in 2026 does not require a credit card.** GitHub OAuth signup landed the owner on `billing.plan: hobby` without any payment prompt.
2. **REST-only deploy works.** No `vercel` CLI install, no local Node build — we ship inline HTML through the API. This is the pattern we will use for the Next.js frontend deploy in phase 4 (pre-build locally, then push the build artifact via API).
3. **Egypt → Vercel is fast.** Sub-second TTFB. Static pages will serve from Vercel's edge network — well within the "no cold start" demo budget.
4. **Project alias claimed.** `secureagentrag-smoke.vercel.app` is now permanently owned by the `moazmo` Vercel account. Phase 4 will reuse the same pattern with project name `secureagentrag`.

## Credentials stored

`.env` (gitignored) now contains:

- `VERCEL_TOKEN` — full-account access token
- `VERCEL_USERNAME` — `moazmo`
- `VERCEL_PROJECT_SMOKE` — `secureagentrag-smoke`

## Cleanup notes

The smoke project `secureagentrag-smoke` is **left intact** so the owner can verify it from any browser and so phase 4 can reuse the working pattern. It will be either:

- renamed to `secureagentrag` during phase 4, or
- deleted via `DELETE /v9/projects/{id}` when phase 4 creates the real Next.js project

No charges are accruing — Hobby plan is free.

## Next phase

Phase 1e: Groq verify — done immediately after this, see `04-groq-smoke-result.md`.
Phase 1f: Hostinger Business hPanel inventory — owner action, needs visual inspection of the panel.
