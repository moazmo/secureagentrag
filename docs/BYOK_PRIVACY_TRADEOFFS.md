# BYOK Demo: Privacy Trade-offs (read this before trusting the live demo with sensitive data)

SecureAgentRAG's headline privacy claim is:

> **HIGH-sensitivity content never leaves local infrastructure** — the sensitivity router forces it onto a local Ollama model, regardless of the caller's `prefer_cloud` flag.

That claim is **true in self-hosted mode** (the default, `SAR_BYOK_MODE=false`). It is **deliberately relaxed on the public demo**. This document states exactly what changes, why, and how to get the strict guarantee back. Honesty about a limitation is worth more than a guarantee you can't keep.

---

## What the public demo actually does

The live demo backend runs on a **Hugging Face Space, CPU Basic** tier. That box has **no GPU and no Ollama** — there is no local model to route HIGH content to. So the demo image (`Dockerfile.hf`) sets:

```
SAR_ALLOW_CLOUD_FOR_HIGH=true
```

With that flag on, the inference router is allowed to synthesize **HIGH-classified** content on the configured cloud provider (Groq `llama-3.1-8b-instant`) instead of refusing. Concretely:

| | Self-hosted (default) | Public demo |
|---|---|---|
| `SAR_BYOK_MODE` | `false` | `true` |
| Local Ollama present | Yes | **No** |
| `SAR_ALLOW_CLOUD_FOR_HIGH` | `false` | **`true`** |
| HIGH-sensitivity query | Synthesized on **local Ollama** | Synthesized on **Groq cloud** |
| MEDIUM / LOW query | Local by default, cloud opt-in | Groq cloud |

### How the demo discloses this to the visitor

- Every answer carries the classified **`sensitivity:` badge** in the UI. When a HIGH answer was produced on cloud, the visitor sees `sensitivity: high` next to a cloud-routed answer.
- The audit row records the truth for that turn: `synth_provider=groq`, `forced_local=false`. The provenance is never hidden — it is written to the SHA-256-chained audit log and is exportable from `/byok/audit`.

So the demo does not *pretend* HIGH stayed local. It shows you that, for this deployment, it didn't.

---

## What is still protected on the demo

The cloud-for-HIGH relaxation is the **only** privacy invariant that changes. Everything else still holds:

- **RBAC at the vector-DB layer** — Qdrant payload filters (`org_id` + clearance + roles) run on every search; an unauthorized document is never retrieved no matter how similar.
- **Per-session isolation** — visitor uploads land in `documents_sess_<sid>` collections; cross-session retrieval is structurally impossible and the collection is purged after 24 h.
- **No key persistence** — a visitor's pasted BYOK key (`X-User-LLM-Key`) is used for that request and never written to disk or audit.
- **PII redaction** — emails, SSNs, Luhn-validated cards, IBANs, and seven provider API-key shapes are scrubbed before anything is written to the audit log.
- **Tamper-evident audit** — the SHA-256 hash chain is intact and exportable.

---

## Getting the strict guarantee back (self-hosting)

If you need "HIGH never leaves local", run the platform yourself with a local Ollama:

1. Leave `SAR_BYOK_MODE` unset / `false`.
2. Leave `SAR_ALLOW_CLOUD_FOR_HIGH` unset / `false` (the default).
3. Provide a reachable Ollama (`SAR_OLLAMA_URL`) with `qwen3:8b` pulled.

Now a HIGH-classified query is synthesized on Ollama, or — if Ollama is unavailable — **refused**, never silently routed to cloud. This is the intended posture for any real deployment that handles confidential data. The public demo trades it away on purpose so visitors can exercise the full pipeline at $0 with no GPU.

See [`configuration.md`](./configuration.md) for the full variable reference and [`DECISIONS.md`](../DECISIONS.md) ADR-010 (sensitivity routing) and ADR-025/026/030 (the BYOK demo) for the rationale.
