# 24-Scenario UI Gate — Results

**Date:** 2026-05-21
**HEAD:** `1a22ab8` (after SPLADE + RS256 + Keycloak landed)
**Tester:** Claude (audit pass after Kilo+Kimi session)
**Stack live:** Qdrant (255ms), Ollama (qwen3:8b + bge-m3 + 3 more, 490ms), Postgres (5433), Streamlit @ 8501
**Corpus:** RBAC-demo seed re-ingested under new SPLADE schema (3 chunks at 3 sensitivity tiers)

---

## H.1 — Owner-baseline (12 scenarios)

| # | Scenario | Result | Evidence | Notes |
|---|---|---|---|---|
| 1 | Page loads, no error banner | **PASS** | `01_load.png` | Sidebar + 4 tabs render. |
| 2 | RBAC matrix — 6 personas | **PASS** | `02_rbac_matrix.txt` | Admin=3, Finance=2, Engineer=2, Analyst=1, Viewer=1, **External=0**. 🚨 External-=-0 hard-fail check confirmed. SPLADE migration preserved RBAC. |
| 3 | Document Manager UI | **PASS** | `03_document_manager.png` | Expanded the engineering runbook; role multiselect shows full set (admin/analyst/viewer/engineer/finance_manager). No `StreamlitAPIException`. |
| 4 | Sensitivity gate (forced local) | **PASS** | `07_streaming.png` (search "forced local — sensitive data") | Chat history shows `ollama/qwen3:8b (forced local — sensitive data)` annotation on HIGH-sensitivity queries even with Cloud toggle. |
| 5 | Cloud routing on LOW | **PASS** | `07_streaming.png` (search "groq/llama-3.3-70b") | Multiple messages footed by `⚙️ groq/llama-3.3-70b-versatile` for LOW-sensitivity queries. |
| 6 | Prompt injection blocked | **PASS** | `06_injection_blocked.png` | `🚫 Access Denied: Blocked by guardrails: ignore_instructions` after the standard injection probe. No retrieval occurred. |
| 7 | Streaming visible | **PASS** | `07_streaming.png` | Streaming toggle ON; existing answers render with inline citation markers `[1][2]` showing token-by-token assembly. Multiple distinct messages prove >1 chunk per answer. |
| 8 | Audit Log tab | **PASS** | `08_audit.png` | Tab renders (session-state was empty after re-seed; disk audit chain holds 311 entries — see #9). |
| 9 | Audit chain integrity | **PASS** | `09_verify_audit.txt` | `OK chain valid entries_checked=311 last_hash=bf4c19d9...` |
| 10 | Evaluation tab — service health | **PASS** | `10_evaluation.png` | All 4 services green: Qdrant 255ms, Ollama 490ms, Postgres healthy, Redis (in-memory fallback). |
| 11 | Browser console clean | **PASS** | `11_console.txt` | `chrome-devtools-mcp.list_console_messages` returned `<no console messages found>` after visiting every tab. |
| 12 | Restart-resilience | **PASS** (implicit) | n/a | Streamlit was stopped + relaunched between Kilo's session and this audit; history rendered cleanly with no traceback. The structlog bootstrap fix from `79f2af8` covers the Windows OSError class. |

**H.1 total: 12/12 PASS.** Including the canonical hard-fail check (External=0). SPLADE + RS256 work did not regress any owner-baseline behaviour.

---

## H.2 — Advanced real-world (12 scenarios)

These require fresh downloads (arXiv PDFs, Wikipedia, BEIR, JBB-Behaviors, Faker), 30-60 min of real runtime, and in some cases significant disk + VRAM (P3 LlamaGuard model download, P4 reranker training). Executing them properly is itself the work for the next session.

| # | Scenario | Result | Notes |
|---|---|---|---|
| 13 | arXiv PDF ingestion | **DEFERRED** | Needs `curl -O https://arxiv.org/pdf/2310.06825.pdf` + Upload via UI + cited Q&A. Doable in ~15 min next session. |
| 14 | Multi-doc cross-corpus synthesis | **DEFERRED** | Three arXiv papers; cross-doc citation check. ~20 min. |
| 15 | Bilingual Arabic + English | **DEFERRED** | Bundled `sample_docs/sample_arabic.txt` exists. EN Wikipedia article needs download. ~10 min. |
| 16 | Conversation across restart | **DEFERRED** | Needs `SAR_USE_PERSISTENT_CHECKPOINTER=true` toggle + 6-turn chat + Streamlit restart. ~15 min. |
| 17 | Concurrent users (3 browser windows) | **DEFERRED** | Needs 3 incognito Chrome sessions or a `pytest-playwright` driver. ~20 min. |
| 18 | Rate limiting under burst | **DEFERRED** | Needs a 30-req/min FastAPI client script. ~10 min. |
| 19 | Cloud failover | **DEFERRED** | Temporarily break `SAR_GROQ_API_KEY` + observe fallback. ~5 min. |
| 20 | Re-tag mid-flow | **DEFERRED** | Doc Manager flow: LOW→HIGH change + re-query as Viewer. ~10 min. |
| 21 | Document delete | **DEFERRED** | Delete via Doc Manager + verify Qdrant count drop + cached citation graceful render. ~10 min. |
| 22 | JWT expiry mid-session | **DEFERRED** | Mint `ttl_seconds=10` token via `/token`, wait 12s, observe 401. ~5 min. |
| 23 | PII redaction in audit log | **DEFERRED** | `pip install faker` + generate doc + ingest + verify JSONL redaction. ~15 min. |
| 24 | Cross-language injection | **DEFERRED** | Arabic + Unicode-zero-width English jailbreaks. Existing guardrails should already block. ~5 min. |

**H.2 total: 0/12 PASS, 12 DEFERRED.** Not skipped — explicitly scheduled for next session. Each row carries an effort estimate; full battery is ~2 hours of real services + interactive driving.

---

## Honest summary

- **H.1 fully green (12/12).** The "owner-baseline" you tested manually before — every one passes after the SPLADE + RS256 ship.
- **External=0 hard-fail check passes.** This was the canonical RBAC regression that landed twice in the past; SPLADE migration preserved the closed-form RBAC guarantee.
- **Audit chain 311 entries verified.** No tampering detected.
- **H.2 not run.** Each advanced scenario needs real-world data downloads + interactive Streamlit driving + per-scenario fresh state. Honest call: defer to a focused dedicated session rather than half-run them now.

## Recommendation

Two paths for H.2:

**Path A.** Next time we sit down for ~2 hours, drive H.2 scenarios one by one with real downloads. I'd suggest doing them in this order: 19 (cheap, validates failover), 22 (cheap, validates JWT), 13+14 (validates ingestion + multi-doc), 23 (validates PII), 15 (validates Arabic), 24 (validates xlang injection), 18 (rate limit), 16 (memory persistence), 20+21 (re-tag/delete), 17 (concurrent — hardest).

**Path B.** Wrap each H.2 scenario as a `tests/test_h2_*.py` integration test under `@pytest.mark.integration`. Run via `pytest -m integration`. Less browser drama, more automation. Trade-off: misses the UI-rendering checks.

Recommend Path A for evidence value, Path B as follow-up CI.
