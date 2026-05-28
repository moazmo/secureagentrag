# Phase 1f Smoke — Hostinger Business hPanel Inventory (PASS, with architectural upgrade)

**Run date:** 2026-05-26
**Run by:** Owner (visual inspection of hPanel)

## Outcome

| Check | Result |
|---|---|
| Plan tier | **Business Web Hosting** — shared LiteSpeed, NOT VPS, NOT Cloud Startup |
| Hostinger account id | `u956014053` |
| Disk usage | 670.52 MiB / 50 GiB (well under quota) |
| Inodes | 11,134 / 600,000 (well under quota) |
| Domains attached | **`eilm.live` — active** (owner confirmed not needed anymore — can be repurposed) |
| Free subdomains | Available under `eilm.live` (Hostinger free subdomain program) |
| `public_html` writeable | ✅ `domains/eilm.live/public_html/` exists, no Hostinger placeholder visible |
| Other content under domain | `nodejs/` folder (old Node.js project, ~1 month old) — gitignored historical artifact |
| `DO_NOT_UPLOAD_HERE` marker | Present at `domains/eilm.live/DO_NOT_UPLOAD_HERE` — Hostinger reminder that real uploads belong in `public_html/`, not the domain root |
| Python App panel | **Not present** — confirms Business Web Hosting cannot host long-running Python (no Passenger Python configuration option) |
| DNS Zone Editor | ✅ Accessible — CNAME, A, TXT, MX records all manageable from hPanel |
| Old `eilm.live` deploy status | Active + serving the old project's landing page |

## What changed in the launch plan

**The launch plan assumed the owner had no domain.** This turned out to be wrong — `eilm.live` is a real, owned, currently-active domain that the owner has decided is free to repurpose. This is a material upgrade:

| Plan as written | Reality after 1f |
|---|---|
| Landing at `<hostinger-subdomain>.hostingersite.com` (placeholder) | Landing at `https://eilm.live/` (owned domain, free) |
| Frontend at `secureagentrag.vercel.app` (Vercel-default subdomain) | Frontend at `https://app.eilm.live/` via CNAME → `cname.vercel-dns.com` (custom domain, also free) |
| HF Space at `LeomordKaly-secureagentrag-api.hf.space` | Same — HF custom domain requires HF Pro ($9/mo); keep `.hf.space` since the frontend is the recruiter-facing URL anyway |
| Cost: $0 | Cost: $0 — Hostinger plan already paid, Vercel custom domain free on Hobby plan |

Custom domains on Vercel Hobby are free — confirmed in `01-stack-decisions.md` research. No upgrade required.

## What this proves

1. **Business Web Hosting cannot host FastAPI / Streamlit / Docker** — confirms the launch plan's choice to host backend on HF Spaces, not Hostinger. The shared LiteSpeed plan has no Python App configuration option, and shared LiteSpeed cannot run a long-running ASGI server.
2. **Hostinger CAN host the static landing page and own the domain DNS.** This is the only role we ever needed Hostinger for. Plan delivers on its existing role.
3. **The owned `eilm.live` domain unlocks a much better URL story for the demo.** Recruiters land on `https://eilm.live/` (clean) instead of `https://<hostinger-subdomain>.hostingersite.com/` (clearly free-tier).
4. **DNS Zone Editor enables CNAME-to-Vercel.** Standard Vercel custom-domain setup: add CNAME `app.eilm.live → cname.vercel-dns.com`, register the domain in the Vercel project, Vercel auto-provisions a Let's Encrypt cert. Zero cost.

## URL plan after 1f

```
https://eilm.live/                        Hostinger static landing (replaces old project's index)
https://app.eilm.live/                    Vercel — CNAME to cname.vercel-dns.com (custom Vercel domain)
https://LeomordKaly-secureagentrag-api.hf.space/    HF Space backend (unchanged)
https://github.com/moazmo/secureagentrag  Source code
```

Optional later: `https://api.eilm.live/` CNAME → HF Space subdomain. Requires HF Pro for custom domain on the Space, so skipped — visitors hit the backend through the Vercel frontend anyway and never see the `.hf.space` URL.

## Cleanup of old eilm.live content

Owner said they don't need the old project anymore. Two paths:

- **A — Delete old content + ship fresh landing.** Backup `nodejs/` and `public_html/` locally (in case owner ever wants it back), then wipe `public_html/`, upload the new `landing/` from this repo's `landing/` (phase 5).
- **B — Park old at `/legacy` subpath.** Move existing `public_html/*` to `public_html/legacy/`, then put new `index.html` at the root. Slightly riskier — might break old project's relative paths.

**Recommendation: A.** The old project is at `eilm.live` per the owner; it's their judgment whether to back up first. The plan codes path A as default in `06-hostinger-landing.md` (phase 5).

## Next phase

**Phase 1 complete: 6 of 6 smoke tests passed.** No stop conditions hit.

Phase 2: backend BYOK mode on `deploy/prod-launch`. Agent action — no further owner input needed until code is ready for review.
