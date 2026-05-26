# Phase 4.5 + 5 — Custom Domain + Hostinger Landing (Owner Action Required)

**Run date:** 2026-05-26
**Agent work:** done (Vercel domain added, landing page written)
**Owner work:** ~5 minutes in Hostinger hPanel (DNS + File Manager upload)

## What's done already

1. ✅ `app.eilm.live` added to the Vercel project `secureagentrag-web`
   (`prj_UbQgkkp0OnWdspplJwdAk2LFyPDG`) via REST.
2. ✅ Vercel returned `verified: true` and `misconfigured: true` — meaning
   the project owns the name but the DNS CNAME isn't pointed at Vercel yet.
3. ✅ Landing page written at `landing/index.html` + `landing/styles.css`.

## Step 1 — DNS CNAME in Hostinger (you, ~1 min)

1. Log into `https://hpanel.hostinger.com/`.
2. Domains → `eilm.live` → DNS / Nameservers → **DNS Zone Editor**.
3. **Delete any existing record for `app` subdomain** (A or CNAME) if
   there is one — Hostinger refuses to add a duplicate.
4. Click **Add Record** and enter:

   | Field | Value |
   |---|---|
   | Type | `CNAME` |
   | Name | `app` |
   | Target / Points to | `cname.vercel-dns.com` |
   | TTL | `14400` (or "Auto") |

5. Save.
6. Wait 2–10 minutes for DNS propagation + Let's Encrypt cert issuance.
7. Visit `https://app.eilm.live` → should serve the same content as
   `https://secureagentrag-web.vercel.app`.

If Hostinger insists on a trailing `.eilm.live` on the CNAME target,
that's fine — `cname.vercel-dns.com.eilm.live` is **not** what we want.
The CNAME *value* must be exactly `cname.vercel-dns.com` (or
`cname.vercel-dns.com.` with the literal trailing dot).

## Step 2 — Upload landing page (you, ~3 min)

1. Hostinger hPanel → **File Manager** (or open
   `https://hpanel.hostinger.com/file-manager` directly).
2. Navigate to `domains/eilm.live/public_html/`.
3. **Backup first** if you want to keep the old project's content:
   - Right-click `public_html` → **Compress** → download the ZIP.
   - Owner explicitly said "i don't need it anymore" in phase 1f, so
     this is optional.
4. Delete every file currently inside `public_html/` (or move them to
   `public_html/legacy/` if you'd rather hide them than discard).
5. Upload these two files from your local checkout:
   - `F:\CV_project\secureagentrag\landing\index.html`
   - `F:\CV_project\secureagentrag\landing\styles.css`
6. Visit `https://eilm.live` → should render the new landing.

Permissions: Hostinger's File Manager sets file perms to 644 and
directory perms to 755 by default. No manual chmod required.

## Step 3 — Smoke (you or agent in a follow-up turn)

```bash
# DNS check (run from any machine)
nslookup app.eilm.live          # expect CNAME -> cname.vercel-dns.com

# Landing reachable
curl -s -o /dev/null -w "%{http_code} %{time_starttransfer}s\n" https://eilm.live/

# App reachable via custom domain
curl -s -o /dev/null -w "%{http_code} %{time_starttransfer}s\n" https://app.eilm.live/

# Chat still works through the new URL
curl -s -X POST https://app.eilm.live/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query":"hi","sessionId":"smoke","persona":"engineer"}'
```

Each should return HTTP 200. Agent will run these as a smoke after DNS
propagates and write a final follow-up note.

## Stop conditions

If Hostinger refuses the CNAME for any reason:

- Try the alternative target `02bfa34d036a7da2.vercel-dns-017.com` (also
  returned by Vercel's `/v6/domains/{name}/config` endpoint).
- If that fails too, fall back to an A record: `app` → `76.76.21.21`.
- If neither works (Hostinger sometimes locks DNS on certain shared
  plans), skip the custom domain — `secureagentrag-web.vercel.app` is
  still a perfectly valid demo URL.

## Notes on the apex `eilm.live`

The apex domain (`eilm.live` itself) keeps pointing at the Hostinger
shared host so the landing page serves there. No DNS changes needed
at the apex.

If you ever want `eilm.live` (no `app.`) to redirect *to* `app.eilm.live`
instead of serving the static landing, that's a one-line `<meta refresh>`
in `landing/index.html` or a `.htaccess` redirect on Hostinger.

## Once both checks pass (DNS + landing)

Next phase candidates:

- **Phase 8** — 4-minute demo video against the polished URLs
  (`eilm.live` → `app.eilm.live`).
- **Phase 9** — merge `deploy/prod-launch` → `main`, flip ADR-025
  through ADR-028 from draft to **Accepted**, tag `v1.0.0-launch`.
- **Backend 3.5 / 7.5** — fix `/readyz` Ollama check, X-Forwarded-For
  for owner-key throttle, optional SAR_ALLOW_CLOUD_FOR_HIGH.

See `private/roadmap.md` for the full state.
