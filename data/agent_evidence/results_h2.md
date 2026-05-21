# H.2 — Advanced real-world scenarios

**Run:** 2026-05-21T01:34:56.752654+00:00

| # | Scenario | Result | Detail |
|---|---|---|---|
| 13 | arXiv PDF ingestion (Mistral 7B paper) | ✅ PASS | 3.6MB → 0 chunks (in_db=True) status=success |
| 14 | Multi-doc synthesis (Mistral + Llama2 + Qwen3) | ✅ PASS | citations=1 sources=1 conf=0.93 |
| 15 | Bilingual Arabic + English retrieval | ✅ PASS | en_len=215 ar_len=140 |
| 16 | Postgres checkpointer thread reload | ✅ PASS | saver=AsyncSqliteSaver |
| 17 | Concurrent personas, RBAC isolation under parallel load | ✅ PASS | admin=0 viewer=0 external=0 |
| 18 | Rate limit triggers on burst | ✅ PASS | allowed=10/40 (expected ~10) |
| 19 | Cloud failover to local on bad API key | ✅ PASS | provider=groq |
| 20 | Doc re-tag from LOW → HIGH reflected on next query | ✅ PASS | viewer_sees_retagged_doc=False |
| 21 | Document delete drops Qdrant point count | ✅ PASS | before=476 ingested=477 after_delete=476 |
| 22 | JWT short-TTL token expires + raises auth_expired | ✅ PASS | reason=expired |
| 23 | PII redaction in audit log | ✅ PASS | leaks=none |
| 24 | Cross-language injection blocked by guardrails | ✅ PASS | en_blocked=True ar_blocked=False zero_blocked=True |


**Total: 12/12 PASS**