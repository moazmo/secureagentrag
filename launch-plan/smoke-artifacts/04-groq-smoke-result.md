# Phase 1e Smoke — Groq Key Verify (PASS)

**Run date:** 2026-05-26
**Run by:** Claude (agent), key already in `.env` from prior work

## Outcome

| Check | Result |
|---|---|
| Existing `SAR_GROQ_API_KEY` still valid | ✅ |
| HTTP status from Egypt | 200 |
| Model availability (`llama-3.1-8b-instant`) | ✅ |
| Rate limit headers present | ✅ |
| Rate limit — daily request quota | `x-ratelimit-limit-requests: 14400` (= free tier 14.4k req/day) |
| Rate limit — remaining requests | 14399 (used 1 for this smoke) |
| Rate limit — tokens-per-minute quota | `x-ratelimit-remaining-tokens: 5952` of 6000 |
| Rate limit reset window | 6 s |
| Response correctness | model replied `"PONG."` to the test prompt |
| End-to-end latency | `total_time: 9.7 ms` (Groq-side); from Egypt total RTT ~300-400 ms |

## Test prompt

```python
{
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "Reply with exactly the word PONG."}],
    "max_tokens": 5,
}
```

## What this proves

1. **Free tier still no-CC in 2026.** The key was issued under the free tier and continues to authenticate without any payment method on file.
2. **14.4k requests / 6000 TPM is the current free quota.** Matches what the launch plan budgeted.
3. **Egypt → Groq is fast.** Single-digit-millisecond Groq-side latency, sub-second wall clock including TLS. Acceptable for streaming UX.
4. **`x-ratelimit-*` headers are exposed.** The BYOK throttle (phase 2) can use these to detect when the owner-key fallback is approaching exhaustion and respond with HTTP 429 + "use your own key" copy.

## Security flag

This exact `SAR_GROQ_API_KEY` value appeared in chat transcripts twice (once in the original `.env`, once as a literal in the security-checklist draft). Per the launch plan's standard hygiene, the owner will **rotate** this key after all five phase-1 smokes complete. The new key will replace the current one in `.env`; the `git` repo is unaffected since `.env` is gitignored.

## No artifacts changed

Phase 1e is a pure read-side smoke against an existing credential. No deployment was created, no Groq-side resource was provisioned, no env-var was added.

## Next phase

Phase 1f: Hostinger Business hPanel inventory. Owner-only — needs hPanel login.
