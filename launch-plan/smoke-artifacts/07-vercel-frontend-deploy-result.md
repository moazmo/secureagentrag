# Phase 4 Deploy — Next.js Frontend on Vercel (PASS)

**Run date:** 2026-05-26
**Run by:** Claude (agent)
**Live URL (temporary):** https://secureagentrag-web.vercel.app
**Target URL (after DNS):** https://app.eilm.live

## Outcome

| Check | Result |
|---|---|
| GitHub sibling repo created | ✅ `github.com/moazmo/secureagentrag-web` |
| Next.js 16 + TypeScript + Tailwind v4 scaffold | ✅ via `create-next-app` |
| BYOK localStorage + persona switcher + chat UI | ✅ single-page `src/app/page.tsx` |
| Edge function `/api/chat` proxy to HF Space | ✅ `src/app/api/chat/route.ts` |
| `npm run build` clean | ✅ 0 TypeScript errors, all routes generated |
| Vercel project created via REST API | ✅ `prj_UbQgkkp0OnWdspplJwdAk2LFyPDG` |
| `vercel --prod` deploy | ✅ `dpl_2CE4v1R2szzHPCJTjVn4ZbqC7ok8`, build 23 s |
| Vercel alias claimed | ✅ `secureagentrag-web.vercel.app` |
| Egypt → Vercel frontend | HTTP 200, TTFB 0.63 s |
| Vercel → HF Space round-trip | HTTP 200, total 4.7 s warm |
| End-to-end smoke | ✅ full chain works: browser → Edge → FastAPI → Qdrant + Groq |
| Custom domain `app.eilm.live` | ⏳ pending CNAME in Hostinger DNS Zone Editor (phase 4.5) |

## Architecture (production end-to-end)

```
Egypt visitor (browser)
   │ HTTPS
   ▼
Vercel Edge (US ingress, free Hobby plan)
   │  static page from edge cache (cold ~0.6s, warm <0.1s)
   │  /api/chat goes to Edge function (cold ~0.5s, warm <0.1s)
   │ HTTPS POST { query, sessionId, persona, byok }
   ▼
src/app/api/chat/route.ts (Edge function)
   │ forwards to https://LeomordKaly-secureagentrag-api.hf.space/byok/chat
   │ headers: X-Session-ID, X-Demo-Persona, X-User-LLM-Key (if BYOK)
   ▼
HF Space FastAPI (CPU Basic, 2 vCPU / 16 GB)
   │ extract_byok dependency
   │ owner-key throttle (no key) or BYOK direct path
   │ run_rag_pipeline -> 9-node LangGraph
   ↓
Qdrant Cloud free 1 GB (sparse + dense) + Groq llama-3.1-8b-instant
```

## Files committed to `secureagentrag-web`

```
src/
  app/
    layout.tsx           # html shell, dark mode, OpenGraph
    page.tsx             # BYOK drawer + persona pills + chat UI (~300 LOC)
    api/chat/route.ts    # Edge function proxy to HF Space
    globals.css          # Tailwind v4 directives (unchanged from scaffold)
  lib/
    byok.ts              # localStorage helpers, sticky session UUID
README.md                # project doc with live URL + dev steps
.env.example             # NEXT_PUBLIC_API_URL override pattern
package.json             # next@16.2.6, react@19.2.4, tailwindcss v4
```

## Features shipped

- **BYOK drawer** — provider radio (groq / openai / anthropic / ollama),
  password-type input, save / clear buttons. localStorage persistence.
- **Persona pills** — engineer / compliance / executive translates to
  the matching `_DEMO_PERSONAS` profile on the backend.
- **Chat messages** — sender labels, message bubbles, per-message
  metadata strip showing `conf 0.xx`, `review`, `byok|owner-key`,
  citation count.
- **Owner-key fallback banner** — appears above chat when BYOK is unset,
  reminding visitors to paste a throwaway key.
- **Sticky session** — `crypto.randomUUID()` slice cached in
  localStorage so a returning visitor's Qdrant collection stays consistent.
- **OpenGraph + Twitter Card** — set in `layout.tsx` for share previews.
- **Dark-first theme** — neutral palette matches the Vercel /
  shadcn aesthetic without pulling shadcn primitives yet (planned for a
  later polish pass).

## Two iteration glitches fixed during deploy

### Glitch 1 — Vercel CLI scope required in non-interactive mode

`vercel --token ... --prod --yes` rejected with `missing_scope` because
the token belongs to a personal team and the CLI refused to auto-pick.

Resolution: added `--scope moazmos-projects` to the command. For future
deploys via CI, set `VERCEL_ORG_ID=team_8pIDB0DVrfmzYXkduBth1BFh` in env
to make this implicit.

### Glitch 2 — `vercel link` token passing on Windows

The shell expansion `--token $(cat .env.token)` did not interpolate the
token in the harness's PowerShell parent. Wrote the token to `/tmp/.vtok`
temporarily, read it back into a shell variable, then deleted the file
after the deploy completed. No token persisted to disk.

## Cost ceiling — unchanged

| Component | Monthly |
|---|---|
| Vercel Hobby plan | $0 |
| Custom domain via Hostinger | already paid |
| Bandwidth | well under 100 GB / month Vercel quota |
| **Total new spend** | **$0** |

## Reproducible deploy command

```bash
# from secureagentrag-web/ on a checkout with .env carrying VERCEL_TOKEN
npx vercel --token "$VERCEL_TOKEN" \
           --prod --yes \
           --scope moazmos-projects
```

## Smoke commands from local machine

```bash
# landing page
curl -s -o /dev/null -w "%{http_code} %{time_starttransfer}s\n" \
     https://secureagentrag-web.vercel.app/

# end-to-end chat through the Edge proxy
curl -s -X POST https://secureagentrag-web.vercel.app/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query":"hi","sessionId":"smoke","persona":"engineer"}'
```

## Next phase

- **4.5 — Custom domain `app.eilm.live`:** add Vercel custom domain via
  REST, write CNAME `app -> cname.vercel-dns.com` in Hostinger DNS Zone
  Editor, wait ~5 min for Let's Encrypt cert provisioning.
- **5 — Hostinger landing page at `eilm.live`** — static HTML linking to
  the live demo + screenshots + YouTube embed (after demo video shoots).
- **6 — GitHub Actions keepalive cron** for the HF Space.
- **7 — End-to-end smoke** with persona switching + RBAC visible.
- **8 — 4-minute demo video** against the live stack.
