# 05 — Phase 4: Next.js BYOK Frontend

**Owner of this phase:** AI agent.
**Pre-requisite:** phase 3 (HF Space) live, `https://LeomordKaly-secureagentrag-api.hf.space/health` returns 200 from Egypt.

## Goal

A modern, dark-first, single-page Next.js 15 app deployed to Vercel that:

1. Accepts a visitor BYOK key + provider choice (localStorage, never sent to telemetry)
2. Lets the visitor pick one of three preset personas (engineer / compliance / executive)
3. Streams the corrective RAG loop in real time via Vercel AI SDK
4. Shows the audit chain inline (collapsible)
5. Shows the faithfulness annotations inline (citation markers with supported/unsupported badges)
6. Mobile responsive
7. Cold-start free (Vercel Edge for static + SSR)

## Tech stack

| Layer | Choice |
|---|---|
| Framework | Next.js 15 App Router |
| Lang | TypeScript |
| UI kit | shadcn/ui + Tailwind v4 |
| Streaming | Vercel AI SDK (`useChat`, `useCompletion`) |
| State | React Server Components for static, `useState` for interactive |
| Theme | Dark-first neutral palette (`#0a0a0a` background, `#fafafa` text, `#2563eb` accent) |
| Fonts | Geist Sans + Geist Mono (Vercel default) |
| Icons | Lucide |
| Auth | none — BYOK is the auth model |
| Analytics | none initially (privacy-first signal) |

## Project location

The Next.js app lives in a sibling repo: `secureagentrag-web`. Reason: keeping it inside the Python repo confuses Vercel auto-detection and forces a monorepo config. Sibling repo with a CI badge in the main repo's README is cleaner.

**To be created:** `https://github.com/moazmo/secureagentrag-web`

This launch plan's `05-nextjs-frontend.md` is the single spec for that sibling repo; do not create it inside the Python repo.

## Pages

| Route | What it does |
|---|---|
| `/` | Landing hero + persona selector + BYOK key input + chat UI |
| `/audit` | Modal/drawer overlay — shows the SHA-256 hash chain for current session |
| `/about` | One-pager — describes the project's four "production patterns most demos skip" |

No `/login`. No `/dashboard`. BYOK is the model.

## BYOK panel

Top-right corner: small "🔑 Set API Key" pill. Click → drawer opens with:

- **Provider** radio: Groq / OpenAI / Anthropic / Ollama (URL)
- **API key** textbox (`type="password"`, no autofill)
- **Save** button — writes to `localStorage`, never to a backend
- **Clear** button — wipes localStorage entry

Banner at the top: **"Public demo. Use throwaway keys only. Do not paste production credentials."**

If localStorage has no key, the chat UI shows a small "Using owner-key (limited to 3 queries/hour per IP)" notice with a "Use my own key" link.

## Chat UI

shadcn-style chat with:

- Sender labels (You / SecureAgentRAG)
- Token-by-token streaming
- Per-message metadata strip:
  - Confidence: `0.82` (green/amber/red dot)
  - Faithfulness: `unsupported sentences = 0/3` (with click-to-highlight)
  - Citations: `[1] NIST AI RMF, p.12` (click to expand chunk)
  - Provider: `groq:llama-3.1-8b-instant` (small tag)
  - Latency: `1.2s`
- Persona switcher above the input (Engineer / Compliance / Executive)
- Upload PDF button (queues to session-scoped Qdrant collection)

## Audit-chain viewer

Drawer on the right. Shows each request as a card:

```
┌────────────────────────────────────────────────────┐
│ #003 · 2026-05-26 14:32:18 UTC                     │
│ query: "What's the NIST AI RMF approach to..."     │
│ outcome: ALLOW · confidence 0.82                   │
│ hash: 0xa3...e8f                                   │
│ prev:  0x2c...b7d                                  │
└────────────────────────────────────────────────────┘
```

Verify button at the bottom recomputes the chain and shows ✓ / ✗.

## Persona selector

Three preset RBAC profiles baked into the frontend. When selected, the frontend sends `X-Demo-Persona: engineer|compliance|executive` header. Backend translates to the appropriate `org_id`, `clearance`, and `roles` for the request.

| Persona | Clearance | Roles | Sample query |
|---|---|---|---|
| Engineer | 2 (low) | `["engineering"]` | "What's the OAuth grant type for our service?" |
| Compliance | 4 (high) | `["compliance", "legal"]` | "Show me ISO 27001 mapping for incident response." |
| Executive | 5 (top) | `["executive", "compliance"]` | "Summarize Q4 risk posture across business units." |

The same backend RBAC filter applies — the demo shows that switching personas shows different chunks for the same query (live evidence of the RBAC layer).

## Streaming wire format

Vercel AI SDK's `useChat` expects SSE events in a known format. Our FastAPI backend already streams via `graph.astream(stream_mode=["updates","custom"])`. The bridge is a thin adapter:

```typescript
// app/api/chat/route.ts (Next.js side, Vercel Edge Function)
import { StreamingTextResponse } from 'ai';

export const runtime = 'edge';

export async function POST(req: Request) {
  const { messages, persona, sessionId } = await req.json();
  const userKey = req.headers.get('x-user-llm-key');
  const provider = req.headers.get('x-user-provider');

  const backend = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-User-LLM-Key': userKey || '',
      'X-User-Provider': provider || 'groq',
      'X-Session-ID': sessionId,
      'X-Demo-Persona': persona,
    },
    body: JSON.stringify({ messages }),
  });

  return new StreamingTextResponse(backend.body!);
}
```

Setting `runtime = 'edge'` puts this function at Cloudflare Edge → zero cold start.

## Environment variables (Vercel project settings)

| Var | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://LeomordKaly-secureagentrag-api.hf.space` |
| `NEXT_PUBLIC_OWNER_KEY_QUOTA` | `3` (queries per IP per hour) |
| `NEXT_PUBLIC_DEMO_URL` | The Vercel deploy URL (for OG share preview) |

## SEO + share

- Open Graph image generated via Next.js `opengraph-image.tsx` (free, no external service)
- Twitter Card meta tags
- Sitemap at `/sitemap.xml`
- Robots.txt allowing all (this is a portfolio demo, we want indexing)

## Files to create in the sibling repo

```
secureagentrag-web/
├── app/
│   ├── layout.tsx
│   ├── page.tsx                    # main chat UI
│   ├── about/page.tsx
│   ├── api/chat/route.ts           # streaming adapter
│   ├── opengraph-image.tsx
│   └── globals.css
├── components/
│   ├── byok-drawer.tsx
│   ├── chat-message.tsx
│   ├── persona-switcher.tsx
│   ├── audit-viewer.tsx
│   ├── citation-popover.tsx
│   └── ui/                          # shadcn primitives
├── lib/
│   ├── api.ts                       # backend client
│   ├── session.ts                   # session UUID management
│   └── localstorage.ts              # BYOK key persistence
├── public/
│   ├── favicon.ico
│   └── og.png
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── next.config.mjs
└── README.md
```

## Acceptance criteria

- [ ] Sibling repo `secureagentrag-web` created on GitHub
- [ ] `npm run dev` boots without error
- [ ] Visiting `localhost:3000` shows persona selector + BYOK drawer + chat input
- [ ] BYOK key saves to localStorage and is sent on next request
- [ ] Streaming works end-to-end against `localhost:8000` (local FastAPI) — tokens appear progressively
- [ ] Confidence + faithfulness + citation badges render correctly
- [ ] Mobile responsive (Chrome devtools iPhone 14 Pro emulation)
- [ ] Deploy to Vercel succeeds via `vercel --prod`
- [ ] `https://secureagentrag.vercel.app` reachable from Egypt
- [ ] Lighthouse score ≥ 90 on Performance, Accessibility, Best Practices, SEO
- [ ] Backend CORS allowlist updated to include final Vercel URL

## Reference UIs

Aesthetic targets to study (clone the feel, not the code):

- `chat.vercel.ai` — token streaming UX
- `t3.chat` — model picker pattern
- `morphic.sh` — citation popover
- `liner.com` — answer chain visualization

## Out of scope

- No multi-user accounts
- No conversation history beyond the current session (no DB writes from frontend)
- No image upload (PDF only for v1)
- No voice input
- No mobile app
