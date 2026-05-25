# 11 — Security Checklist

The BYOK demo mode introduces a new threat surface: visitor API keys travel through the owner's backend. A compromised backend can exfiltrate them. This checklist is mandatory before going live and before every subsequent change to the BYOK code path.

## Threat model

| Threat | Where it surfaces | Severity |
|---|---|---|
| Visitor's API key logged to disk | `audit_logs/*.jsonl` write path | **CRITICAL** |
| Visitor's API key sent to Phoenix / OTel | Observability middleware | **CRITICAL** |
| Visitor A's PDFs visible to visitor B | Qdrant collection routing | HIGH |
| Visitor A's prompt visible in visitor B's audit | Audit log scoping | HIGH |
| Cross-origin abuse (CSRF, key theft via JS) | CORS misconfig | HIGH |
| Owner-key abuse (recruiter blasts 1000 queries) | Per-IP throttle | MEDIUM |
| Prompt injection bypasses RBAC | Guardrails layer | MEDIUM (already mitigated) |
| Session collection grows unbounded | Purge cron failure | MEDIUM |
| Reflected XSS via citation rendering | Frontend escaping | LOW (Next.js escapes by default) |

## Pre-launch mandatory checks

### 1. Key never lands on disk

```python
# tests/test_security/test_byok_key_redaction.py
@pytest.mark.parametrize("key", [
    "gsk_" + "A" * 52,                                           # Groq shape
    "sk-proj-" + "a" * 48,                                       # OpenAI shape
    "sk-ant-api03-" + "a" * 40,                                  # Anthropic shape
])
def test_api_keys_are_redacted_from_audit(key):
    body = f"User provided key: {key} for inference"
    redacted = redact(body)
    assert key not in redacted

def test_audit_log_never_writes_raw_authorization_header():
    """End-to-end: simulate a BYOK request and grep audit JSONL for the key."""
    key = "gsk_" + "Z" * 50                                      # synthetic test fixture
    # ... make request ...
    log_content = Path("audit_logs/test.jsonl").read_text()
    assert key not in log_content, "raw API key found in audit log"
```

Both tests must pass before merge.

### 2. Phoenix disabled in BYOK mode

```python
# config/observability.py
def setup_phoenix():
    if settings.byok_mode:
        return                              # never instrument in BYOK
    ...
```

Add a test:

```python
def test_phoenix_is_not_initialized_in_byok_mode(monkeypatch):
    monkeypatch.setenv("SAR_BYOK_MODE", "true")
    # ... import path forcing settings reload ...
    assert not _phoenix_initialized()
```

### 3. CORS allowlist

`SAR_CORS_ALLOW_ORIGINS` must be set to the exact Vercel URL, no wildcards.

```python
# Smoke test
def test_cors_rejects_unknown_origin(client):
    r = client.options("/chat", headers={"Origin": "https://evil.example.com"})
    assert "evil.example.com" not in r.headers.get("Access-Control-Allow-Origin", "")
```

### 4. Per-IP throttle on owner-key

Owner-key fallback must consult the throttle before reaching the LLM client. Bypass test:

```python
def test_owner_key_throttle_returns_429_after_quota(client):
    for _ in range(3):
        r = client.post("/chat", json={"query": "..."})  # no BYOK header
        assert r.status_code == 200
    r4 = client.post("/chat", json={"query": "..."})
    assert r4.status_code == 429
    assert "use your own key" in r4.json()["detail"].lower()
```

### 5. Session isolation

Two parallel session IDs must not see each other's data:

```python
def test_session_a_cannot_read_session_b_uploads(client):
    # session A uploads
    client.post("/ingest", files=..., headers={"X-Session-ID": "a"})
    # session B queries
    r = client.post("/chat", json={"query": "..."}, headers={"X-Session-ID": "b"})
    chunks = r.json()["citations"]
    # B must see zero chunks from A's upload
    assert all(c["source"] != "a-only.pdf" for c in chunks)
```

### 6. Frontend never logs the BYOK key

In the Next.js app:

```typescript
// lib/api.ts
export async function callChat(...) {
  const userKey = localStorage.getItem('byok-key');
  // BAD: console.log({ userKey })
  // BAD: telemetry.track({ userKey })
  // GOOD: just forward as a header
  return fetch(...);
}
```

Lint rule: no `console.log` calls anywhere in the production bundle (enabled in `next.config.mjs`).

### 7. Browser key persistence is localStorage-only

Do NOT use cookies for the BYOK key:

- Cookies travel automatically on every request (CSRF surface)
- localStorage requires explicit JS read (controlled forwarding)

```typescript
// CORRECT
localStorage.setItem('byok-key', key);

// INCORRECT — do not do this
document.cookie = `byok-key=${key}`;
```

### 8. Banner UX

The BYOK drawer must show a warning above the input:

> **Public demo. Use throwaway API keys only. Do not paste production credentials.**

This is a UX safety net for visitors who skim.

## Runtime monitoring

Once live, the agent or owner should periodically:

- [ ] Spot-check audit log entries — grep for `sk-` and `gsk_` patterns
- [ ] Review HF Space logs for any unexpected exception traces containing key-like strings
- [ ] Confirm the keepalive cron is hitting `/health`, not `/chat` (cron requests must never carry a real key)

## Incident response

If a visitor reports key leakage:

1. Take the HF Space offline (Pause button in settings)
2. Acknowledge in writing
3. Wipe the audit logs (`huggingface-cli space-files ...delete /tmp/audit_logs --recursive`)
4. Patch the underlying bug, add a regression test
5. Re-enable
6. Notify the visitor of resolution

See `10-rollback-plan.md` § Scenario 4 for details.

## Notable: this list grows

Every new feature on `deploy/prod-launch` that touches the BYOK or session code path must add a corresponding test to `tests/test_security/test_byok_*.py`. The checklist is not "complete" — it is the minimum bar.
