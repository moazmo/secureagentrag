"""Seed a 3-doc RBAC demo corpus.

Creates three documents at three sensitivity tiers, each tagged for a
distinct role set, so the RBAC user-switcher demo in the Streamlit
sidebar produces visibly different result sets for the same query.

Usage:
    uv run python -m scripts.seed_demo_rbac

The same query "What is our policy on data sharing?" returns:
- viewer: 1 doc (public handbook)
- engineer: 2 docs (public + engineering runbook)
- finance_manager: 2 docs (public + finance revenue)
- admin: 3 docs (all)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingestion.metadata import IngestRequest, SensitivityLevel  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embeddings import EmbeddingService  # noqa: E402
from retrieval.hybrid_search import BM25Index  # noqa: E402
from retrieval.qdrant_client import QdrantManager  # noqa: E402

PUBLIC_HANDBOOK = """ACME Corp Employee Handbook (Public)

Our policy on data sharing applies to all employees and contractors.
External sharing of any company document requires written approval from
your manager. Personal data of customers must never be transmitted via
unencrypted channels. All employees complete annual privacy training.

Data classification levels at ACME are:
- Public: marketing materials, press releases, this handbook
- Internal: engineering runbooks, project plans
- Confidential: financial records, customer PII, M&A discussions

When in doubt about whether a document can be shared, escalate to legal.
"""

ENGINEERING_RUNBOOK = """ACME Engineering Runbook — Service Reliability (Internal)

Our policy on data sharing across services follows zero-trust principles.
All inter-service traffic is mTLS-authenticated. Production database
credentials live in HashiCorp Vault and rotate every 90 days. Logs
exported to the central SIEM are pre-scrubbed of customer PII via the
data redaction pipeline.

Incident response: page the on-call engineer via PagerDuty for any
Sev-1 or Sev-2 event. Customer data exfiltration suspicion is always
Sev-1 regardless of confirmed impact. Engineering managers must run
post-incident reviews within 48 hours.
"""

FINANCE_REPORT = """ACME Finance — Q3 Revenue Brief (Confidential)

Our policy on data sharing of unpublished financial figures is strict:
Q3 revenue numbers must not leave the finance team until the earnings
call. Q3 GAAP revenue came in at $487.2M, beating consensus of $462M by
5.4%. Operating margin held at 22.1% versus 21.3% Q2. Free cash flow
$94M.

Material non-public information must not be discussed on Slack, in
unsecured email, or with any external party including spouses. Insider
trading exposure is taken seriously — Compliance reviews every trade by
designated officers within 24h.
"""

DOCS = [
    {
        "name": "demo_public_handbook.txt",
        "content": PUBLIC_HANDBOOK,
        "sensitivity": SensitivityLevel.LOW,
        "roles": ["viewer", "engineer", "finance_manager", "analyst", "admin"],
    },
    {
        "name": "demo_engineering_runbook.txt",
        "content": ENGINEERING_RUNBOOK,
        "sensitivity": SensitivityLevel.MEDIUM,
        "roles": ["engineer", "admin"],
    },
    {
        "name": "demo_finance_q3.txt",
        "content": FINANCE_REPORT,
        "sensitivity": SensitivityLevel.HIGH,
        "roles": ["finance_manager", "admin"],
    },
]


async def main() -> int:
    qdrant = QdrantManager()
    embeddings = EmbeddingService()
    bm25 = BM25Index()
    pipeline = IngestionPipeline(qdrant, embeddings, bm25_index=bm25)

    print(f"Dropping collection '{qdrant.collection_name}'...")
    qdrant.delete_collection()
    qdrant.ensure_collection()

    target_dir = _ROOT / "sample_docs" / "demo_rbac"
    target_dir.mkdir(parents=True, exist_ok=True)

    for doc in DOCS:
        target = target_dir / doc["name"]
        target.write_text(doc["content"], encoding="utf-8")
        req = IngestRequest(
            file_path=str(target),
            user_id="seed_demo_rbac",
            org_id="acme_corp",
            sensitivity_level=doc["sensitivity"],
            roles=doc["roles"],
        )
        print(
            f"Ingesting {doc['name']:35} "
            f"sensitivity={doc['sensitivity'].value:8} "
            f"roles={doc['roles']}"
        )
        result = await pipeline.ingest_document(req)
        print(f"  -> {result.num_chunks} chunks, status={result.status}")

    print()
    print("Demo seed complete. Try the same query as different users to see RBAC in action:")
    print('  - viewer          (1 doc visible) — "What is our policy on data sharing?"')
    print("  - engineer        (2 docs visible)")
    print("  - finance_manager (2 docs visible)")
    print("  - admin           (3 docs visible)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
