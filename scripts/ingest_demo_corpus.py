"""Ingest the demo corpus into the production Qdrant Cloud cluster.

The launch demo needs a small, RBAC-diverse corpus so the public
``/byok/chat`` endpoint returns interesting answers and the persona
switcher visibly changes the visible chunks.

Ingests four documents from ``sample_docs/`` into the base ``documents``
collection on Qdrant Cloud, with metadata tuned so each demo persona
sees a different slice:

| File                              | Sensitivity | Roles                              |
|-----------------------------------|-------------|------------------------------------|
| demo_public_handbook.txt          | PUBLIC (0)  | engineering, compliance, executive |
| demo_engineering_runbook.txt      | INTERNAL (2)| engineering                        |
| demo_finance_q3.txt               | CONFIDENTIAL (3) | compliance, executive          |
| real/NIST_AI_RMF.pdf              | PUBLIC (0)  | engineering, compliance, executive |

Personas after ingestion will see:

- engineer (clearance 2, roles=[engineering]) → handbook + runbook + NIST
- compliance (clearance 4, roles=[compliance,legal]) → handbook + finance + NIST
- executive (clearance 5, roles=[executive,compliance]) → all four

RBAC filter is enforced at Qdrant query time -- chunks the persona is
not authorised to see are physically not returned, regardless of
similarity score.

Run::

    SAR_QDRANT_URL=$SAR_QDRANT_CLOUD_URL \\
    SAR_QDRANT_API_KEY=$SAR_QDRANT_CLOUD_API_KEY \\
    SAR_MULTI_TENANT_COLLECTIONS=false \\
    SAR_EMBEDDING_BACKEND=local \\
    uv run python scripts/ingest_demo_corpus.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Force env BEFORE importing settings so the SAR_* values land in the
# pydantic model on first read. We override (not setdefault) because the
# local .env carries SAR_QDRANT_URL=http://localhost:6333 from phase 1.
load_dotenv()
cloud_url = os.environ.get("SAR_QDRANT_CLOUD_URL", "").strip()
cloud_key = os.environ.get("SAR_QDRANT_CLOUD_API_KEY", "").strip()
if not cloud_url:
    sys.exit("SAR_QDRANT_CLOUD_URL missing in .env -- complete phase 1c first")
os.environ["SAR_QDRANT_URL"] = cloud_url
os.environ["SAR_QDRANT_API_KEY"] = cloud_key
os.environ["SAR_MULTI_TENANT_COLLECTIONS"] = "false"
os.environ["SAR_BYOK_MODE"] = "false"  # ingestion is server-side, not BYOK
os.environ["SAR_EMBEDDING_BACKEND"] = "local"
os.environ.setdefault("SAR_LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")

from config.settings import settings  # noqa: E402
from ingestion.metadata import IngestRequest, SensitivityLevel  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embeddings import EmbeddingService  # noqa: E402
from retrieval.qdrant_client import QdrantManager  # noqa: E402
from retrieval.sparse_embeddings import SparseEmbeddingService  # noqa: E402

DEMO_ORG_ID = "demo"
SAMPLE_ROOT = Path(__file__).resolve().parent.parent / "sample_docs"


@dataclass
class CorpusItem:
    """One demo document and its RBAC metadata."""

    file: Path
    sensitivity: SensitivityLevel
    roles: list[str]
    label: str


CORPUS: list[CorpusItem] = [
    # SensitivityLevel maps LOW=1, MEDIUM=2, HIGH=3 (see ingestion/metadata.py).
    # All personas have clearance>=2 so LOW docs are universally visible.
    CorpusItem(
        file=SAMPLE_ROOT / "demo_rbac" / "demo_public_handbook.txt",
        sensitivity=SensitivityLevel.LOW,
        roles=["engineering", "compliance", "legal", "executive", "viewer"],
        label="public handbook",
    ),
    CorpusItem(
        file=SAMPLE_ROOT / "demo_rbac" / "demo_engineering_runbook.txt",
        sensitivity=SensitivityLevel.MEDIUM,
        roles=["engineering"],
        label="engineering runbook",
    ),
    # NOTE: kept at MEDIUM (not HIGH) so cloud synthesis is allowed. HIGH
    # sensitivity forces local Ollama in ``inference.router`` and the HF
    # Space CPU Basic image has no Ollama. RBAC differentiation against
    # engineer is preserved via the ``roles`` allowlist.
    CorpusItem(
        file=SAMPLE_ROOT / "demo_rbac" / "demo_finance_q3.txt",
        sensitivity=SensitivityLevel.MEDIUM,
        roles=["compliance", "executive"],
        label="finance Q3",
    ),
    CorpusItem(
        file=SAMPLE_ROOT / "real" / "NIST_AI_RMF.pdf",
        sensitivity=SensitivityLevel.LOW,
        roles=["engineering", "compliance", "legal", "executive", "viewer"],
        label="NIST AI RMF",
    ),
]


async def main() -> int:
    print(f"Qdrant target: {settings.qdrant_url}")
    print(f"Collection:    {settings.qdrant_collection}")
    print(f"Multi-tenant:  {settings.multi_tenant_collections}")
    print(f"Embedding:     {settings.embedding_backend} / {settings.local_embedding_model}")
    print()

    qdrant = QdrantManager(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )
    qdrant.ensure_collection()

    embeddings = EmbeddingService(
        ollama_url=settings.ollama_url,
        model=settings.embedding_model,
    )
    sparse = SparseEmbeddingService()

    pipeline = IngestionPipeline(
        qdrant_manager=qdrant,
        embedding_service=embeddings,
        sparse_service=sparse,
    )

    grand_total_chunks = 0
    errors: list[str] = []

    for item in CORPUS:
        if not item.file.exists():
            print(f"  SKIP  {item.label}  ({item.file}) -- file missing")
            errors.append(f"missing: {item.file}")
            continue
        print(f"  -> {item.label}: {item.file.name}")
        req = IngestRequest(
            file_path=str(item.file),
            user_id="demo-ingestor",
            org_id=DEMO_ORG_ID,
            sensitivity_level=item.sensitivity,
            roles=item.roles,
        )
        result = await pipeline.ingest_document(req)
        print(
            f"     status={result.status} chunks={result.num_chunks}"
            f" time={result.processing_time_seconds:.1f}s"
        )
        if result.errors:
            print(f"     errors: {result.errors}")
            errors.extend(result.errors)
        grand_total_chunks += result.num_chunks

    print()
    print(f"== summary ==")
    print(f"  total chunks ingested: {grand_total_chunks}")
    print(f"  errors:                {len(errors)}")
    if errors:
        for e in errors:
            print(f"    - {e}")
        return 1

    # Smoke a search call against the cluster to confirm the chunks landed.
    print()
    print("== smoke: count points in collection ==")
    info = qdrant._client.get_collection(qdrant.collection_name)
    print(f"  collection: {qdrant.collection_name}")
    print(f"  vectors_count: {info.vectors_count}")
    print(f"  points_count:  {info.points_count}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
