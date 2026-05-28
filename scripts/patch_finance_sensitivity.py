"""One-shot: re-promote demo_finance_q3.txt chunks to HIGH sensitivity.

Earlier ingest demoted them to MEDIUM as a workaround for the cloud-only HF
Space (HIGH used to force local Ollama). Now that ``SAR_ALLOW_CLOUD_FOR_HIGH``
unlocks cloud synthesis for HIGH, we want the finance doc to demonstrate the
sensitivity router decision in the trace UI. Patches the existing chunks
in place rather than deleting + re-ingesting (cheap; no embedding recompute).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()
cloud_url = os.environ.get("SAR_QDRANT_CLOUD_URL", "").strip()
cloud_key = os.environ.get("SAR_QDRANT_CLOUD_API_KEY", "").strip()
if not cloud_url:
    sys.exit("SAR_QDRANT_CLOUD_URL missing")
os.environ["SAR_QDRANT_URL"] = cloud_url
os.environ["SAR_QDRANT_API_KEY"] = cloud_key

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

client = QdrantClient(url=cloud_url, api_key=cloud_key, timeout=30)
collection = "documents"

# Payload stores the full absolute path the document was ingested from.
# Match the absolute path used when the doc was originally upserted.
source_match = Filter(
    must=[
        FieldCondition(
            key="source_file",
            match=MatchValue(
                value=r"F:\CV_project\secureagentrag\sample_docs\demo_rbac\demo_finance_q3.txt"
            ),
        )
    ]
)

# Update both the categorical and integer sensitivity fields so RBAC
# range filter sees HIGH (3) for this doc.
client.set_payload(
    collection_name=collection,
    payload={
        "sensitivity_level": "high",
        "sensitivity_level_int": 3,
    },
    points=source_match,
)
print("Patched demo_finance_q3.txt -> sensitivity=high (3)")

count, _ = (
    client.count(
        collection_name=collection,
        count_filter=source_match,
        exact=True,
    ),
    None,
)
print(f"Affected chunks: {count.count}")
