# AGENTS.md — Operating Manual for AI Coding Agents

This file tells AI agents (Hermes / Kimi / Claude Code / Cursor / Aider) how to work on **SecureAgentRAG** without breaking it.

**Prerequisite:** read `CLAUDE.md` first.

---

## 0. Identity and authorship

- The repo owner is `moazmo` (`moazmo27@gmail.com`).
- **Never** add AI attribution to commits, PRs, or code (no "Co-Authored-By: Claude / Hermes / Kimi", no "Generated with X" footers).
- Commit author must be the owner. If you are running locally as that user, git will get it right automatically.

---

## 1. Loop: before / during / after every task

### Before
1. `git status` — must be clean. Don't start work on a dirty tree.
2. `git pull origin main` — make sure you're on the latest.
3. `uv run pytest -q` — must be **484 passed, 3 skipped** (or higher). If it's not, stop and fix the baseline first.
4. `uv run ruff check . && uv run ruff format --check .` — both clean.
5. Read the relevant module top-to-bottom before touching it. Read its test file.
6. Have a working `qdrant` (`docker compose up -d qdrant`) and `ollama serve` running. If you need cloud, ensure `SAR_GROQ_API_KEY` is in `.env`.

### During
1. **One concern per commit.** Don't bundle refactor + feature + fix.
2. **Write the test first** when adding a security or correctness feature.
3. **Run tests after every meaningful change.** `uv run pytest tests/<path> -q` for fast feedback.
4. **Delete code aggressively.** If you find dead branches, kill them in a separate commit.
5. **Watch out for these foot-guns:**
   - LangGraph 1.x binds a stream writer in every node context — *do not* use writer presence to detect streaming. Use the `_stream` flag in `GraphState`.
   - On Windows, psycopg async requires `WindowsSelectorEventLoopPolicy`. `core/graph.py` pins it at import.
   - structlog must be configured **at import time** (already done in `utils/logging.py`), otherwise module-level loggers crash under Streamlit's stdout capture.
   - Qdrant payload roles need integer sensitivity (`sensitivity_level_int`). Don't accidentally write the string version into the filter.
   - The audit chain breaks on any in-place edit. If you must alter audit semantics, change the schema and re-genesis.

### After
1. `uv run pytest -q` — green.
2. `uv run ruff check . && uv run ruff format --check .` — green.
3. Run the relevant smoke script:
   - Retrieval / RBAC / faithfulness change → `uv run python -m scripts.interview_demo`
   - Auth change → `uv run python -m scripts.e2e_smoke`
   - Audit change → `uv run python -m scripts.verify_audit_chain`
   - UI change → `uv run streamlit run app/main.py` and visually inspect.
4. Stage in logical groups. Write a Conventional Commits message. Push.
5. If you opened a PR: title is a one-line summary, body is **Summary** (3 bullets) + **Test plan** (checklist) + **Files** (terse list).

---

## 2. Definition of done

A task is **not** done until **all** of these are true:

- [ ] Tests pass (`uv run pytest -q`) — 0 failures.
- [ ] Lint passes (`uv run ruff check .`) — 0 errors.
- [ ] Format passes (`uv run ruff format --check .`) — 0 reformats needed.
- [ ] **New code is covered by tests.** New behavior → new test that would have failed without the change.
- [ ] **For security features**: a regression test that demonstrates the failure mode being closed. (See `tests/test_retrieval/test_hybrid_search.py::test_bm25_drops_unauthorised_when_dense_returns_zero` for the canonical pattern.)
- [ ] **For retrieval / eval features**: tested on **real data**, not toy `["foo", "bar"]` fixtures. See section 4 below.
- [ ] **Audit trail and provenance preserved.** Every new node writes an audit entry. Provider/model/jti land in the log.
- [ ] **No new TODO comments** that aren't tracked as GitHub issues.
- [ ] **README, RUNBOOK, architecture.md updated** if the change is user-visible.
- [ ] **DECISIONS.md updated** if the change is architectural (new ADR number).
- [ ] **CLAUDE.md "state of the codebase" section updated** with new commits.

---

## 3. Quality bar: less code, simplicity

The repo has been deliberately trimmed (`-80 LOC` in the streaming refactor alone). Future work must maintain this.

**Rules:**

- **Prefer deletion.** Every PR that adds >100 LOC of net code needs justification in the commit body.
- **No new dependencies without justification.** Adding a package is a permanent cost. If `httpx + asyncio` covers it, don't pull in a framework.
- **No re-implementing what's already there.** If you need rate limiting, use `utils/rate_limiter.py`. If you need a logger, `utils/logging.get_logger`. If you need to call an LLM, `core/agents/router::call_llm_with_decision` or `call_llm_stream`.
- **No premature abstractions.** Don't add an interface for one implementation.
- **Inline anything used once.** Helpers that are called from a single place don't earn their own function unless they're >20 LOC and have a clear name.
- **Comments explain *why*, not *what*.** The code says what. Tell future-you why.

**Counter-example — what NOT to do:**
> "I added a new `AbstractRetrievalStrategy` base class with five concrete implementations to support future extensibility."

We have one retrieval strategy that works. Add the second only when you actually need it.

---

## 4. Testing: REAL data is mandatory

Toy fixtures are fine for parser-level unit tests. **For anything that touches retrieval quality, eval, training, or the integration boundary, use real data.**

### Datasets to download when you need them

| Use case | Dataset | Source |
|---|---|---|
| Reranker training | MS-MARCO Passage Ranking (small) | https://huggingface.co/datasets/microsoft/ms_marco |
| Retrieval eval | BEIR benchmark (TREC-COVID, FiQA, SciFact) | https://huggingface.co/BeIR |
| Faithfulness / hallucination eval | TruthfulQA | https://huggingface.co/datasets/truthful_qa |
| OCR (Arabic, scanned PDFs) | KITAB-Bench, Arabic-OCR-Benchmark | https://huggingface.co/datasets (search) |
| LlamaGuard | meta-llama/Llama-Guard-3-8B | https://huggingface.co/meta-llama/Llama-Guard-3-8B |
| SPLADE | naver/splade-v3, naver/splade-cocondenser-ensembledistil | https://huggingface.co/naver |
| Prompt-injection corpus | jailbreakbench/JBB-Behaviors | https://huggingface.co/datasets/jailbreakbench/JBB-Behaviors |
| OIDC test IdP | Keycloak | `quay.io/keycloak/keycloak:latest` (docker) |
| Bundled real PDF | NIST AI RMF 1.0 (147 chunks, already ingested) | `sample_docs/real/NIST_AI_RMF.pdf` |

### Test scenarios that must be exercised before declaring a feature done

Each feature has its own bar. Use the most demanding subset that applies.

#### A. Retrieval / RBAC / ranking changes
1. **Cross-org isolation:** External(org=partner_inc) gets 0 docs. *Must fail closed on every retrieval path.*
2. **Role mismatch:** Analyst gets 0 engineering-runbook docs.
3. **Clearance underflow:** Viewer (clearance=1) gets 0 HIGH-sensitivity docs.
4. **Permission spill:** Run 50 queries across 4 personas, count docs by `sensitivity_level`. Tabulate. No persona above their clearance.
5. **BM25-only branch:** Disable embeddings (mock failure) and re-run (1)–(4). RBAC must still hold.
6. **Real corpus:** Ingest the NIST AI RMF PDF (147 chunks), confirm 5 standard queries return cited answers.

#### B. Faithfulness / NLI gate changes
1. **Synthetic injection:** Insert a sentence the LLM hallucinates (use a doc that doesn't support a known claim). Confirm `*[unsupported]*` annotation in flag mode, removal in drop mode.
2. **Ratio drives review:** Faithfulness ratio < threshold flips `needs_human_review=True`.
3. **Latency budget:** 5 cited sentences should add <3 s on local Qwen3-8B with `max_concurrent=4`.
4. **Multilingual:** Run on the bundled Arabic sample (`sample_docs/sample_arabic.txt`). Gate must not mis-classify due to script.
5. **Real entailment benchmark:** Score against a 50-sentence labeled subset of TruthfulQA. Report precision/recall.

#### C. Auth / JWT / OIDC changes
1. **Round trip:** mint → verify → assert claims match.
2. **Expired token:** rejected with `reason=expired`.
3. **Bad signature:** rejected with `reason=bad_signature`.
4. **Wrong audience / issuer:** rejected with `reason=bad_claims`.
5. **jti in audit:** every authenticated query carries the token's `jti` in `thread_id`.
6. **RS256 + Keycloak (if implementing #4):** spin up Keycloak in docker, configure realm, mint via Keycloak, verify via JWKS endpoint. Test with rotated keys.

#### D. Guardrails / LlamaGuard changes
1. **JBB-Behaviors corpus:** Score the regex gate vs the LLM escalation vs LlamaGuard on the full set. Compare detection rates.
2. **False positive set:** Normal queries that contain trigger words ("how do I drop a database column" — should NOT block). Confirm low FP rate.
3. **Latency:** Per-query overhead under 500 ms median.
4. **Strict mode escalation path:** Confirm the regex hit triggers the LLM/LlamaGuard escalation, not the other way around.

#### E. Sparse vector / SPLADE migration
1. **Recall parity:** SPLADE vs BM25 on TREC-COVID — SPLADE recall@10 must be ≥ BM25 recall@10 + 2pp.
2. **Per-tenant isolation:** Multi-tenant collection with two orgs — query in org A returns zero org B docs even with SPLADE-only retrieval.
3. **Re-index time:** 147 NIST chunks → SPLADE indexed in < 2 min on RTX 3060.
4. **Storage:** SPLADE index size vs BM25 pickle size — report.
5. **End-to-end:** All RBAC tests (A1-A6 above) pass with SPLADE swapped for BM25.

#### F. Reranker fine-tuning
1. **Train set:** MS-MARCO small triplets (~500K). Hold out 5K for eval.
2. **Baseline:** off-the-shelf BGE-Reranker-v2-M3 on hold-out.
3. **Fine-tuned:** train on MS-MARCO + NIST corpus pairs, eval on hold-out.
4. **Acceptance:** fine-tuned NDCG@10 ≥ baseline + 1pp on hold-out.
5. **In-domain:** Both checkpoints evaluated on a hand-labeled 20-query NIST subset.

#### G. Anything that changes the streaming or graph topology
1. **Streaming contract test** in `tests/test_agents/test_graph.py::test_streaming_emits_token_events_via_writer` still passes.
2. **Non-streaming path** still calls `call_llm_with_decision` (not `call_llm_stream`).
3. **Deadline:** `SAR_REQUEST_TIMEOUT_S=0.05` produces a timeout audit entry.
4. **All four interfaces still work:** Streamlit + FastAPI + MCP + `run_rag_pipeline()` direct.

#### H. UI changes (Streamlit)
1. **Browser smoke:** Launch Streamlit, click every tab, run one query. **Use chrome-devtools-mcp or Playwright if available**, otherwise document a manual checklist.
2. **No console errors.** Capture browser console after each tab.
3. **All four personas (Admin / Analyst / Viewer / External) still differentiate.**
4. **Streaming visible:** tokens render incrementally, not in one chunk.

---

## 5. Commit style

Conventional Commits. Subject ≤72 chars. Imperative mood. No AI attribution.

```
<type>(<scope>): <subject>

<body — wrap at 80 cols, explain WHY, not what>

<optional footer: e.g. Closes #123>
```

Types: `feat | fix | docs | style | refactor | test | chore | perf | build | ci`.

**Good examples** (from this repo):

- `feat(faithfulness): NLI-based citation faithfulness gate + evaluator integration`
- `fix(security): close RBAC bypass in HybridSearcher when dense returns zero`
- `refactor(streaming): unify with graph.astream(stream_mode=['updates','custom'])`
- `chore(scripts): merge cloud_bench_quick + seed_demo_rbac into siblings with flags`

**Bad examples** (do not do this):

- `update files` (no scope, no info)
- `fix bug` (which one?)
- `feat: implement comprehensive solution for advanced retrieval architecture` (waffle)
- Any commit with `Co-Authored-By: Claude` or similar.

---

## 6. PR style

```
## Summary
- One-line bullet describing the user-visible change.
- One-line bullet describing the mechanism.
- One-line bullet for any breaking change or migration step.

## Test plan
- [ ] uv run pytest -q  → 484+ passed
- [ ] uv run ruff check .  → All checks passed!
- [ ] uv run ruff format --check .  → clean
- [ ] (Feature-specific real-data test from section 4)
- [ ] (Smoke / UI / integration tests as applicable)

## Files
- core/foo.py — <one line>
- tests/test_core/test_foo.py — <one line>
```

No AI attribution. No "🤖 Generated with..." footer.

---

## 7. When to stop and ask the human

Stop and ask before:

- Adding a new top-level dependency.
- Changing the public schema of `QueryResponse` or `UserContext`.
- Breaking the audit chain format (would invalidate existing log files).
- Removing or renaming a public CLI script.
- Adding any network call that bypasses the inference router.
- Changing the RBAC filter shape (`build_rbac_filter` signature).

When in doubt: open an issue with a one-paragraph proposal and the trade-offs.

---

## 8. Survival kit — what to do if you broke something

1. **Tests broke.** Read the failing test. Don't update assertions to match new behavior unless behavior is intentionally changing — that's the test's job to catch.
2. **Streamlit blank with `OSError [Errno 22]`.** structlog wasn't bootstrapped. Confirm `utils/logging.py` ends with `with contextlib.suppress(Exception): setup_logging()`.
3. **Streamlit blank with `engineer not in options`.** Role multiselect dropdown narrower than payload. See `app/views/upload.py`.
4. **Postgres checkpointer fails on Windows.** Confirm `core/graph.py` pins selector loop at import. `SAR_USE_PERSISTENT_CHECKPOINTER=false` is the safe fallback.
5. **External user sees ACME docs.** RBAC re-check guard regressed. Read `retrieval/hybrid_search.py` ~line 411 — the `allowed_doc_ids` must always be initialised when RBAC is on. Run `tests/test_retrieval/test_hybrid_search.py::test_bm25_drops_unauthorised_when_dense_returns_zero`.
6. **Audit chain verifier reports `broken=1`.** Someone edited a past entry. `audit_logs/*.jsonl` is append-only. Revert the file or accept that the chain is broken from that index forward and re-genesis.

---

## 9. Where to commit changes

- All work goes to `main` via small focused commits (single-developer flow).
- For larger sequences (a multi-day feature), use a feature branch `feat/<short-name>` and rebase onto `main` before merge.
- Push to `origin/main` only after the full test + lint + format gate is green.
- Tag releases as `v0.X.Y` once we hit a milestone. Currently pre-1.0.

---

## 10. Final reminder

The four hero stories are the project's identity. **Do not compromise them.**

1. RBAC at the vector layer + multi-tenant collections.
2. Sensitivity-based inference routing (HIGH = local, always).
3. NLI faithfulness gate over corrective RAG.
4. Tamper-evident audit chain + SLO deadline.

If a refactor or new feature weakens any of these even briefly, stop, surface it in the PR description, and propose mitigation.

**Less code. Real data. Tests green. Audit clean.**

That's the bar.
