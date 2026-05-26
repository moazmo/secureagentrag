# Hostinger Landing Page — `eilm.live`

Static HTML/CSS landing page for SecureAgentRAG. Lives at the apex of
`eilm.live`. Single CTA → `https://app.eilm.live` (Vercel frontend).

## Files

```
landing/
├── index.html      # Hero + 4 patterns + try-it steps + arch diagram + footer
├── styles.css      # Dark-first, system fonts, mobile responsive (< 720px)
└── README.md       # This file -- owner reference
```

No JS frameworks, no build step. Pure HTML + inline-SVG favicon. Page
weight: ~12 KB total (no images at this revision).

## Upload to Hostinger (owner action)

1. Log into `https://hpanel.hostinger.com/`.
2. Open File Manager → navigate to `domains/eilm.live/public_html/`.
3. **Backup the existing content first** if you want to keep the old
   project (download as ZIP via File Manager → Compress).
4. Delete everything inside `public_html/` (or move it to
   `public_html/legacy/`).
5. Upload `index.html` and `styles.css` from this directory directly into
   `public_html/`.
6. Visit `https://eilm.live` — should render the new landing page.

## Sanity checks

- `index.html` validates against the W3C HTML5 validator.
- Lighthouse mobile: target Performance ≥ 95, Accessibility ≥ 95.
- No mixed content — all external links use HTTPS.
- No analytics / cookies / trackers.

## Updating

Any future copy edit happens in this folder, then re-upload via
File Manager (or set up SFTP if you want one-command pushes).

## Related

- `app.eilm.live` — Vercel custom domain, served by the
  `secureagentrag-web` GitHub repo. Domain wired in phase 4.5; DNS
  CNAME lives in the Hostinger DNS Zone Editor.
- `LeomordKaly-secureagentrag-api.hf.space` — the backend; visitors
  never see this URL directly, only the Next.js frontend talks to it
  via Vercel Edge function `/api/chat`.
