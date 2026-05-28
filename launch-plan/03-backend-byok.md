# 03 — Phase 2: Backend BYOK Mode

**Owner of this phase:** AI agent (on `deploy/prod-launch` branch).
**Pre-requisite:** phase 1 smoke tests all green.

## Goal

Add a "Bring Your Own Key" mode to the FastAPI backend. Visitors paste their own LLM API key into the Next.js frontend; the frontend forwards it as an HTTP header on every request. The backend extracts it per-request, uses it instead of the owner's env-stored key, and never persists it anywhere.

Each visitor also gets a session-scoped Qdrant collection so visitor A's uploaded PDFs are not visible to visitor B.

## Scope

This phase only modifies the **backend** (`interfaces/api.py`, `inference/`, `retrieval/`, `config/`). The Next.js frontend is phase 4. The HF Space Dockerfile is phase 3.

## File-by-file changes

### `config/settings.py`

Add three new env-controlled settings:

```python
class Settings(BaseSettings):
    ...
    # ── BYOK demo mode (production launch) ───────────────────────────────
    byok_mode: bool = False                                # SAR_BYOK_MODE
    byok_owner_key_quota_per_hour: int = 3                 # SAR_BYOK_OWNER_QUOTA
    session_collection_ttl_hours: int = 24                 # SAR_SESSION_TTL_HOURS
    cors_allow_origins: list[str] = []                     # SAR_CORS_ALLOW_ORIGINS (JSON array)
```

When `byok_mode=True`:
- Phoenix instrumentation is disabled
- Audit log is wiped at process start
- `multi_tenant_collections` is forced to `True`
- The default `org_id` per request is the session UUID, not `"default"`

### `interfaces/api.py`

Add a dependency that extracts BYOK credentials per request:

```python
# interfaces/byok.py (new file)
from fastapi import Header, Request
from pydantic import BaseModel

class ByokCreds(BaseModel):
    user_key: str | None = None
    provider: str | None = None           # "groq" | "openai" | "anthropic" | "ollama"
    ollama_url: str | None = None
    session_id: str

def extract_byok(
    request: Request,
    x_user_llm_key: str | None = Header(None),
    x_user_provider: str | None = Header(None),
    x_user_ollama_url: str | None = Header(None),
    x_session_id: str | None = Header(None),
) -> ByokCreds:
    # Generate a server-side session ID if client did not provide one;
    # client-provided is fine but never trusted as authentication.
    session_id = x_session_id or _generate_session_id(request)
    return ByokCreds(
        user_key=x_user_llm_key,
        provider=x_user_provider,
        ollama_url=x_user_ollama_url,
        session_id=session_id,
    )
```

Then wire every existing endpoint that already takes an LLM call to receive `creds: ByokCreds = Depends(extract_byok)` and pass it down into the pipeline.

### `inference/cloud_clients.py`

Today, the cloud client reads `settings.groq_api_key` at module import. Change it to accept an optional per-request override:

```python
class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str | None = None):
        self._api_key = api_key or settings.groq_api_key
        self._base_url = base_url
    
    @classmethod
    def for_request(cls, provider: str | None, user_key: str | None):
        """
        Per-request factory used in BYOK mode. Falls back to owner key
        only if the per-IP throttle allows it.
        """
        if user_key:
            # Visitor BYOK path. No throttling.
            return _build_for_provider(provider, user_key)
        # Owner-key fallback — caller must have already checked throttle.
        return _build_owner_default()
```

### `inference/ollama_client.py`

Same shape: add `for_request(ollama_url: str | None)` that overrides the default URL when the visitor brought their own Ollama instance.

### `retrieval/multitenancy.py`

Today `get_collection_name(org_id)` returns the per-org collection. Extend it to accept a session_id when in BYOK mode:

```python
def get_collection_name(org_id: str | None, *, session_id: str | None = None) -> str:
    """
    Single-tenant: returns settings.qdrant_collection
    Multi-tenant: returns "{base}_{sanitized_org}"
    BYOK mode: returns "{base}_sess_{sanitized_session}"
    """
    if settings.byok_mode and session_id:
        return _sanitize(f"{settings.qdrant_collection}_sess_{session_id}")
    if settings.multi_tenant_collections and org_id:
        return _sanitize(f"{settings.qdrant_collection}_{org_id}")
    return settings.qdrant_collection
```

### `utils/rate_limiter.py`

Today this exists as a generic in-process rate limiter. Add a new `per_ip_per_hour` configuration that the BYOK middleware consults before allowing an owner-key fallback request:

```python
class OwnerKeyThrottle:
    def __init__(self, quota_per_hour: int):
        self._buckets: dict[str, list[float]] = {}
        self._quota = quota_per_hour
    
    def allow(self, ip: str) -> bool:
        now = time.monotonic()
        bucket = [t for t in self._buckets.get(ip, []) if now - t < 3600]
        if len(bucket) >= self._quota:
            return False
        bucket.append(now)
        self._buckets[ip] = bucket
        return True
```

Wire this into the BYOK dependency: if `user_key` is None, consult the throttle. If denied, return HTTP 429 with a body suggesting the visitor paste their own key.

### Session purge cron

HF Spaces support "Scheduled Jobs" via the standard Python `apscheduler` library running inside the FastAPI lifespan. Add a daily task that:

1. Lists all collections in the Qdrant Cloud cluster
2. Filters those matching `documents_sess_*`
3. Inspects each collection's `info.creation_date` (recorded in payload metadata when the collection is first created)
4. Deletes collections older than `settings.session_collection_ttl_hours`

```python
# retrieval/session_purge.py (new file)
import asyncio
from datetime import datetime, timedelta, UTC

async def purge_expired_sessions(client, ttl_hours: int) -> int:
    cutoff = datetime.now(UTC) - timedelta(hours=ttl_hours)
    deleted = 0
    for c in client.get_collections().collections:
        if not c.name.startswith(f"{settings.qdrant_collection}_sess_"):
            continue
        meta = client.get_collection(c.name).config.params.metadata or {}
        created = datetime.fromisoformat(meta.get("created_at", "2000-01-01"))
        if created < cutoff:
            client.delete_collection(c.name)
            deleted += 1
    return deleted
```

Schedule from FastAPI lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler()
    scheduler.add_job(purge_expired_sessions, "interval", hours=6, args=[qdrant_client, settings.session_collection_ttl_hours])
    scheduler.start()
    yield
    scheduler.shutdown()
```

### Audit redaction regression test

`utils/pii.redact` already redacts API keys via a regex for `[A-Z0-9_]{20,}` patterns. Add an explicit test that proves a Groq, OpenAI, and Anthropic key never survives a round trip through the audit log:

```python
# tests/test_security/test_byok_key_redaction.py
import pytest
from utils.pii import redact

@pytest.mark.parametrize("key", [
    "gsk_" + "A" * 52,                                           # Groq shape (synthetic)
    "sk-proj-" + "a" * 48,                                       # OpenAI shape (synthetic)
    "sk-ant-api03-" + "a" * 40,                                  # Anthropic shape (synthetic)
])
def test_api_keys_are_redacted_from_audit(key):
    body = f"User provided key: {key} for inference"
    redacted = redact(body)
    assert key not in redacted, "API key survived redaction — security regression"
```

### CORS

In BYOK mode, enable CORS for the Vercel frontend's URL only. Read from `settings.cors_allow_origins`.

```python
if settings.byok_mode:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,                          # BYOK does not use cookies
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
```

## Tests to add

| Test file | What it covers |
|---|---|
| `tests/test_interfaces/test_byok.py` | header extraction, session ID generation, throttle integration |
| `tests/test_security/test_byok_key_redaction.py` | API keys redacted from audit log |
| `tests/test_retrieval/test_session_collections.py` | session UUIDs route to `documents_sess_<id>` |
| `tests/test_retrieval/test_session_purge.py` | purge cron deletes only collections older than TTL |
| `tests/test_inference/test_per_request_keys.py` | per-request key override does not leak into shared client state |

## Acceptance criteria

- [ ] All existing 487 tests still pass
- [ ] At least 5 new tests for BYOK mode, all pass
- [ ] `SAR_BYOK_MODE=true uvicorn interfaces.api:app` boots without error
- [ ] Manual smoke: `curl -H "X-User-LLM-Key: $TEST_KEY" -H "X-User-Provider: groq" -H "X-Session-ID: abc" http://localhost:8000/chat ...` returns a streamed response
- [ ] Owner key request without `X-User-LLM-Key`: first 3 attempts from one IP succeed, 4th returns HTTP 429
- [ ] `redact()` removes a real Groq key from a sample request body
- [ ] Lint clean, format clean

## Out of scope for this phase

- Streamlit changes (we keep Streamlit on `main` only; on `deploy/prod-launch` we leave `app/` untouched but unused)
- Frontend code (phase 4)
- HF Space Dockerfile (phase 3)
- Hostinger landing page (phase 5)
