# 06 — Phase 5: Hostinger Static Landing Page

**Owner of this phase:** AI agent writes the HTML/CSS; owner uploads via Hostinger hPanel File Manager.
**Pre-requisite:** phase 4 (Vercel frontend) live at `https://app.eilm.live`.

## Goal

A single-page static marketing site on the owner's existing Hostinger Business Web Hosting plan, with three sections:

1. Hero — project name, one-line pitch, "Open live demo" CTA
2. Demo embed — YouTube video + animated GIF preview + screenshots
3. Architecture + tech — short paragraph + diagram + GitHub link

No frameworks, no build step. Pure HTML + CSS + a few `<script>` tags. Loads in under 200 ms.

## Why a static landing is needed at all

The Vercel URL `app.eilm.live` is fine for direct sharing but:

- The recruiter-facing URL deserves to look intentional (not a deploy-platform subdomain)
- A landing page is the right place for the screencast embed, README highlights, and a small "About" section that a chat-only UI cannot host
- The owner is paying for Hostinger anyway — use it

## Files to create

```
landing/                                # in the secureagentrag repo
├── index.html
├── styles.css
├── assets/
│   ├── hero-bg.svg
│   ├── demo.gif                        # < 2 MB, autoplaying
│   ├── architecture.svg
│   ├── og-image.png                    # 1200×630 for share previews
│   ├── favicon.ico
│   └── screenshots/
│       ├── chat.png
│       ├── audit.png
│       └── persona-switch.png
├── robots.txt
└── sitemap.xml
```

These ship in the `launch-plan/landing-assets/` directory of this branch first, then the owner copies them to Hostinger.

## Skeleton

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SecureAgentRAG — privacy-first multi-agent RAG platform</title>
<meta name="description" content="A production-grade multi-agent RAG platform with RBAC at the vector DB layer, sensitivity-based inference routing, faithfulness gate, and hash-chained audit log." />
<meta property="og:title" content="SecureAgentRAG" />
<meta property="og:description" content="Privacy-first multi-agent RAG with RBAC, faithfulness gate, and audit chain." />
<meta property="og:image" content="https://eilm.live/assets/og-image.png" />
<meta property="og:url" content="https://eilm.live/" />
<meta name="twitter:card" content="summary_large_image" />
<link rel="icon" href="/assets/favicon.ico" />
<link rel="stylesheet" href="/styles.css" />
</head>
<body>

<header class="hero">
  <nav>
    <a href="#" class="logo">SecureAgentRAG</a>
    <a href="https://github.com/moazmo/secureagentrag" target="_blank" rel="noopener">GitHub</a>
  </nav>
  <div class="hero-content">
    <h1>Privacy-first multi-agent RAG</h1>
    <p class="lede">
      Four production patterns most RAG demos skip:
      RBAC at the vector DB layer, sensitivity-based inference routing,
      NLI per-sentence faithfulness gate, SHA-256 hash-chained audit log.
    </p>
    <div class="cta-row">
      <a class="cta-primary" href="https://app.eilm.live">Open live demo →</a>
      <a class="cta-secondary" href="https://github.com/moazmo/secureagentrag">View source</a>
    </div>
  </div>
</header>

<section id="demo" class="demo">
  <h2>Watch it in action</h2>
  <div class="video-wrap">
    <iframe
      src="https://www.youtube.com/embed/<VIDEO_ID>?rel=0&modestbranding=1"
      title="SecureAgentRAG live walkthrough"
      allowfullscreen
      loading="lazy"
    ></iframe>
  </div>
  <p class="caption">
    Four minutes. Persona switching, PDF upload, streaming, audit chain verification,
    prompt-injection block.
  </p>
</section>

<section id="patterns" class="patterns">
  <h2>What makes it different</h2>
  <div class="grid">
    <div class="card">
      <h3>RBAC at vector DB</h3>
      <p>Org + clearance + role filter enforced as a Qdrant must-filter on every search. Structurally impossible to bypass from the application layer.</p>
    </div>
    <div class="card">
      <h3>Sensitivity routing</h3>
      <p>HIGH-sensitivity documents force local Ollama inference. LOW can opt in to Groq / OpenAI / Anthropic. The router decides per-request.</p>
    </div>
    <div class="card">
      <h3>NLI faithfulness gate</h3>
      <p>Every cited sentence must be entailed by its source chunk. Citation marker presence is not enough.</p>
    </div>
    <div class="card">
      <h3>Hash-chained audit</h3>
      <p>SHA-256 chain over every request. PII redacted before persist. Tamper-evident, verifiable in-browser.</p>
    </div>
  </div>
</section>

<section id="architecture" class="architecture">
  <h2>Architecture</h2>
  <img src="/assets/architecture.svg" alt="System architecture diagram" />
  <p>
    LangGraph 1.x state machine with 9 nodes. Qdrant native sparse vectors.
    Streamlit / FastAPI / MCP interfaces. 487 tests, 24 ADRs.
  </p>
</section>

<footer>
  <p>Built by <a href="https://github.com/moazmo">Moaz Muhammad</a> · MIT licensed</p>
</footer>

</body>
</html>
```

## Styling notes

- Dark mode default (`#0a0a0a` background, `#fafafa` text, `#2563eb` accent) — matches the Vercel app
- Hero uses a subtle SVG gradient backdrop
- Cards have a thin `1px` border + `8px` border-radius
- All fonts: `system-ui, -apple-system, "Segoe UI", sans-serif` — zero font load
- Mobile: stack cards vertically below 768 px
- Total page weight target: < 100 KB excluding video iframe

## Hostinger upload steps (owner action)

1. Log into hPanel
2. File Manager → `public_html/`
3. Delete any existing placeholder `index.html`
4. Upload all files from `launch-plan/landing-assets/` to `public_html/`
5. Visit the domain or free subdomain → confirm the landing page renders
6. If Hostinger shows a 403/404, check file permissions (`644` for files, `755` for `assets/`)

## DNS

If the owner buys a domain or already has one:

- A record: `@` → Hostinger shared hosting IP (auto)
- CNAME: `www` → `@`

No external DNS needed since the landing page lives on Hostinger.

## Acceptance criteria

- [ ] `landing/index.html` validates against W3C HTML5 validator
- [ ] Lighthouse on mobile: Performance ≥ 95, Accessibility ≥ 95
- [ ] Page weight < 100 KB without video iframe load
- [ ] CTA button click takes user to `https://app.eilm.live` (verified after phase 4)
- [ ] Owner uploaded to Hostinger and the public URL serves the page

## Out of scope

- No CMS
- No forms / contact submission
- No analytics
- No newsletter signup
- No multi-page navigation
