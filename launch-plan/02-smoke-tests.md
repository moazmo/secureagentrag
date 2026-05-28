# 02 — Phase 1: Smoke Tests (no code yet)

Run all five sections before writing a single line of production code. If any one fails, halt and consult the fallback in `01-stack-decisions.md` § Stop conditions.

**Time budget:** 30-60 minutes total (mostly waiting for signup emails).

**Who runs this:** the owner (Moaz). AI agents cannot sign up for accounts on the owner's behalf — they will create the resources but the owner clicks the email confirmation links and pastes credentials back.

## Why this phase exists

Free-tier policies change. "No credit card" today may be "credit card after first project" tomorrow. Egypt may be silently added to a blocklist. Sparse vector support on Qdrant Cloud free may be deprecated. We refuse to write a single line of deployment code until all five providers verifiably accept us without payment friction.

---

## 1. Hugging Face Spaces

**Goal:** account created, free Docker Space deployed, hello-world reachable from Egypt.

### Steps

1. Open `https://huggingface.co/join` in an incognito window
2. Sign up with `moazmo27@gmail.com` (or owner's preferred email)
3. **Verify: no credit card field appears at signup.** If it does, abort and switch to Northflank backup
4. Confirm email
5. Generate an access token at `https://huggingface.co/settings/tokens` with `write` scope. Save as `HF_TOKEN` in `.env` (added below in agent step)
6. Create a new Space at `https://huggingface.co/new-space`:
   - **Space name:** `secureagentrag-api`
   - **License:** MIT
   - **SDK:** Docker
   - **Hardware:** CPU basic (2 vCPU, 16 GB) — should be the default
   - **Visibility:** Public
7. Wait for the Space to initialize (~30 seconds)
8. Clone the Space repo locally:
   ```bash
   git clone https://huggingface.co/spaces/LeomordKaly/secureagentrag-api hf-smoke
   cd hf-smoke
   ```
9. Add a minimal `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN pip install fastapi uvicorn
   RUN echo 'from fastapi import FastAPI\napp = FastAPI()\n@app.get("/")\ndef root(): return {"ok": True, "service": "smoke"}' > main.py
   EXPOSE 7860
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
10. Commit and push:
    ```bash
    git add Dockerfile
    git commit -m "smoke: hello-world FastAPI"
    git push
    ```
11. Wait 1-2 minutes for HF to build and start
12. From Egypt, `curl https://LeomordKaly-secureagentrag-api.hf.space/` — expect `{"ok": true, "service": "smoke"}`
13. Note the URL — record below

### Pass criteria

- [ ] Signup completed without credit card
- [ ] Public URL serves the smoke endpoint from Egypt
- [ ] HF token saved as `HF_TOKEN` in `.env` (file is gitignored)
- [ ] Space URL recorded: `https://________________________.hf.space`

---

## 2. Qdrant Cloud

**Goal:** account created, free 1 GB cluster running, Python `qdrant-client` connects from Egypt.

### Steps

1. Open `https://cloud.qdrant.io/` in incognito
2. Sign up with email or GitHub OAuth
3. **Verify: no credit card field at signup.** If it does, abort and switch to self-hosted Qdrant in HF Space
4. Create a new cluster:
   - **Cluster name:** `secureagentrag-demo`
   - **Tier:** Free (1 GB, 0.5 vCPU, 4 GB disk)
   - **Cloud provider:** any (AWS us-east is fine, Egypt has low latency to both AWS and GCP US-East)
5. Wait for cluster to provision (~2 minutes)
6. Copy the cluster URL (looks like `https://<uuid>.us-east.aws.cloud.qdrant.io:6333`)
7. Generate an API key from the cluster's "API Keys" tab
8. Save in `.env` as:
   ```
   SAR_QDRANT_CLOUD_URL=https://<uuid>.us-east.aws.cloud.qdrant.io
   SAR_QDRANT_CLOUD_API_KEY=<key>
   ```
9. Smoke from local Python:
   ```python
   from qdrant_client import QdrantClient
   c = QdrantClient(url="<URL>", api_key="<KEY>")
   print(c.get_collections())
   # expect: collections=[]
   ```
10. Verify sparse vector support:
    ```python
    from qdrant_client.http.models import (
        VectorParams, Distance, SparseVectorParams
    )
    c.create_collection(
        "smoke",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    print(c.get_collection("smoke"))
    c.delete_collection("smoke")
    ```

### Pass criteria

- [ ] Signup completed without credit card
- [ ] Free 1 GB cluster provisioned
- [ ] `get_collections()` returns from Egypt latency < 1s
- [ ] Sparse vector collection creation succeeds
- [ ] URL + API key saved in `.env`

---

## 3. Vercel

**Goal:** account created, Hobby plan, Next.js hello-world deployed.

### Steps

1. Open `https://vercel.com/signup` in incognito
2. Sign up with GitHub OAuth (use `moazmo` GitHub account)
3. **Verify: Hobby plan auto-selected, no credit card prompt.** If CC asked, abort and use Cloudflare Pages or fall back to static export on Hostinger
4. From local terminal:
   ```bash
   npm i -g vercel
   npx create-next-app@latest secureagentrag-smoke --typescript --tailwind --app --use-npm
   cd secureagentrag-smoke
   vercel  # follow prompts, link to new project
   vercel --prod
   ```
5. Wait for deploy (~1 minute)
6. Vercel prints URL, e.g. `https://secureagentrag-smoke-<hash>.vercel.app`
7. From Egypt browser, visit URL — expect Next.js welcome page
8. Note URL below

### Pass criteria

- [ ] Signup completed without credit card
- [ ] Hobby plan active
- [ ] Public URL serves Next.js app from Egypt
- [ ] URL recorded: `https://________________________.vercel.app`

---

## 4. Groq (verify existing key still works)

**Goal:** confirm `SAR_GROQ_API_KEY` in `.env` still authenticates from Egypt and rate limits have not changed under our feet.

### Steps

1. From local Python in the secureagentrag repo:
   ```python
   import httpx
   import os
   from dotenv import load_dotenv
   load_dotenv()
   r = httpx.post(
       "https://api.groq.com/openai/v1/chat/completions",
       headers={"Authorization": f"Bearer {os.environ['SAR_GROQ_API_KEY']}"},
       json={
           "model": "llama-3.1-8b-instant",
           "messages": [{"role": "user", "content": "Reply with the word PONG only."}],
           "max_tokens": 5,
       },
   )
   print(r.status_code, r.json())
   ```
2. Expect HTTP 200 and a response containing "PONG"
3. Inspect headers: `x-ratelimit-remaining-requests` should be present and reasonable

### Pass criteria

- [ ] HTTP 200 response from Egypt
- [ ] Rate-limit headers indicate quota available

---

## 5. Hostinger Business hPanel

**Goal:** confirm what owner's existing plan can host.

### Steps

1. Log into `https://hpanel.hostinger.com/`
2. Identify the active plan name (Premium Web Hosting, Business Web Hosting, Cloud Startup, VPS, etc.) — record in `private/roadmap.md`
3. Open File Manager → confirm a `public_html` directory exists and is writeable
4. Open Domains panel — list any domains attached to the account. If `eilm.live` is still listed, note whether it is still pointed at the old project
5. Check if any **free subdomain** is available under Hostinger's free domains program (some plans include `<name>.hostingersite.com` or similar)
6. Open the "Python App" panel (if present) — record what Python version is available, and whether ASGI / WSGI options are listed (if Python App offers Uvicorn, we could host backend here too — unlikely but worth checking)
7. Open DNS Zone Editor — confirm CNAME records can be created
8. Note any "Coming soon" placeholder pages

### Pass criteria

- [ ] Plan tier identified and recorded
- [ ] `public_html` writeable
- [ ] At least one domain or free subdomain available for the landing page (or owner explicitly accepts no landing page)
- [ ] DNS Zone Editor accessible
- [ ] Confirmed Python App support level (informational only — we are NOT using Hostinger for compute)

---

## Aggregate gate

All five sections must show all checkboxes ticked before phase 2 (backend BYOK code) begins. If any single checkbox cannot be ticked, halt and escalate to owner with the specific failure mode.

## What the agent does next

Once owner reports "all smoke green":

1. Update `.env` with `HF_TOKEN`, `SAR_QDRANT_CLOUD_URL`, `SAR_QDRANT_CLOUD_API_KEY`
2. Mark task #25 (`P6.3 Verify stack live`) as completed
3. Proceed to `03-backend-byok.md`

## What if owner gets stuck

Most common failure modes and the agent's response:

- **HF Space build hangs** — check Space logs via `huggingface.co/spaces/LeomordKaly/secureagentrag-api/logs`. Usually Dockerfile syntax error or missing `EXPOSE 7860`
- **Qdrant Cloud cluster stuck in "Provisioning"** — wait full 5 minutes; if still stuck, delete and re-create in a different region
- **Vercel deploy fails** — usually a Node version mismatch; pin Node 20 in `package.json` engines field
- **Groq 401 from Egypt** — key may have been rotated; check `https://console.groq.com/keys`; if revoked, generate a new key and update `.env`
- **Hostinger panel won't show plan** — refresh, try a different browser; if still not visible, owner may need to log in with the original signup email
