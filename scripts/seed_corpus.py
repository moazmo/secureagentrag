"""Bulk-ingest a seed corpus into Qdrant for live demos.

Two modes:

- ``--mode bundled`` (default) ingests the bundled ``sample_docs/`` files
  with sensible per-document RBAC metadata.
- ``--mode rbac`` writes three synthetic ACME-Corp docs at the three
  classification tiers so the Streamlit RBAC user-switcher shows
  visibly different result sets for the same query.

Usage::

    uv run python -m scripts.seed_corpus              # bundled samples
    uv run python -m scripts.seed_corpus --mode rbac  # RBAC demo

The collection is dropped before re-ingestion so repeated runs stay clean.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ingestion.metadata import IngestRequest, SensitivityLevel  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embeddings import EmbeddingService  # noqa: E402
from retrieval.qdrant_client import QdrantManager  # noqa: E402
from retrieval.sparse_embeddings import SparseEmbeddingService  # noqa: E402

# Bundled sample-docs mode — (path, sensitivity, allowed_roles).
BUNDLED_SEEDS: list[tuple[str, SensitivityLevel, list[str]]] = [
    ("sample_docs/sample_english.txt", SensitivityLevel.HIGH, ["admin", "analyst"]),
    (
        "sample_docs/sample_internal_report.pdf",
        SensitivityLevel.MEDIUM,
        ["analyst", "viewer", "admin"],
    ),
    ("sample_docs/sample_arabic.txt", SensitivityLevel.LOW, ["viewer", "analyst", "admin"]),
    ("sample_docs/sample_mixed.txt", SensitivityLevel.LOW, ["viewer", "analyst", "admin"]),
]

# RBAC demo mode — three synthetic docs covering three sensitivity tiers
# and three role groups. Same query "What is our policy on data sharing?"
# returns 1 / 2 / 2 / 3 docs for viewer / engineer / finance_manager / admin.
_PUBLIC_HANDBOOK = """ACME Corp Employee Handbook (Public)

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

_ENGINEERING_RUNBOOK = """ACME Engineering Runbook — Service Reliability (Internal)

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

_FINANCE_REPORT = """ACME Finance — Q3 Revenue Brief (Confidential)

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

RBAC_DOCS = [
    {
        "name": "demo_public_handbook.txt",
        "content": _PUBLIC_HANDBOOK,
        "sensitivity": SensitivityLevel.LOW,
        "roles": ["viewer", "engineer", "finance_manager", "analyst", "admin"],
    },
    {
        "name": "demo_engineering_runbook.txt",
        "content": _ENGINEERING_RUNBOOK,
        "sensitivity": SensitivityLevel.MEDIUM,
        "roles": ["engineer", "admin"],
    },
    {
        "name": "demo_finance_q3.txt",
        "content": _FINANCE_REPORT,
        "sensitivity": SensitivityLevel.HIGH,
        "roles": ["finance_manager", "admin"],
    },
]


async def _seed_bundled(pipeline: IngestionPipeline) -> int:
    total_chunks = 0
    for path, sensitivity, roles in BUNDLED_SEEDS:
        if not Path(path).exists():
            print(f"  skip (not found): {path}")
            continue
        req = IngestRequest(
            file_path=path,
            user_id="seed_script",
            org_id="acme_corp",
            sensitivity_level=sensitivity,
            roles=roles,
        )
        print(f"Ingesting {path}  sensitivity={sensitivity.value}  roles={roles}")
        result = await pipeline.ingest_document(req)
        print(f"  -> {result.num_chunks} chunks, status={result.status}")
        for err in result.errors:
            print(f"     ! {err}")
        total_chunks += result.num_chunks
    return total_chunks


async def _seed_rbac(pipeline: IngestionPipeline) -> int:
    target_dir = _ROOT / "sample_docs" / "demo_rbac"
    target_dir.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    for doc in RBAC_DOCS:
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
        total_chunks += result.num_chunks
    print()
    print("Demo seed complete. Try the same query as different users to see RBAC in action:")
    print('  - viewer          (1 doc)  "What is our policy on data sharing?"')
    print("  - engineer        (2 docs)")
    print("  - finance_manager (2 docs)")
    print("  - admin           (3 docs)")
    return total_chunks


async def _amain(mode: str) -> int:
    qdrant = QdrantManager()
    embeddings = EmbeddingService()
    sparse = SparseEmbeddingService()
    pipeline = IngestionPipeline(qdrant, embeddings, sparse_service=sparse)

    print(f"Dropping collection '{qdrant.collection_name}'...")
    qdrant.delete_collection()
    qdrant.ensure_collection()

    if mode == "rbac":
        total_chunks = await _seed_rbac(pipeline)
    else:
        total_chunks = await _seed_bundled(pipeline)

    print()
    info = qdrant.get_collection_info()
    points = info["points_count"] if info else total_chunks
    print(f"Done. {total_chunks} chunks ingested. Collection points: {points}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("bundled", "rbac"),
        default="bundled",
        help="bundled = sample_docs/*; rbac = three synthetic ACME docs for RBAC demo",
    )
    args = parser.parse_args()
    return asyncio.run(_amain(args.mode))


if __name__ == "__main__":
    sys.exit(main())
