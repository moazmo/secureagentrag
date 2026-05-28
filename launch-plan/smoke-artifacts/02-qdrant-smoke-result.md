# Phase 1c Smoke — Qdrant Cloud (PASS)

**Run date:** 2026-05-26
**Run by:** Claude (agent), management API key supplied by owner

## Outcome

| Check | Result |
|---|---|
| Account signup (no CC required) | ✅ Owner signed up `cloud.qdrant.io`; management API key generated; no payment method on file |
| Management API key validation | ✅ `apikey 12b9…2bPgkyhw7jAWpQ` (truncated) — confirmed via `GET /api/account/v1/accounts` returning account `f88e3546-…7b757` |
| Free-tier package available on AWS us-east-1 | ✅ `package_id=39b48a76-2a60-4ee0-9266-6d1e0f91ea14` (name `free2`, type `free`) |
| Free-tier package available on GCP us-central1 | ❌ 0 packages returned — GCP free tier appears deprecated on owner's account, AWS used instead |
| Cluster creation via management API | ✅ `db2d4134-4e7d-48e9-a771-ccf4bd1d1d2c` (`secureagentrag-demo`) |
| Cluster provisioning time | CREATING → NOT_READY → HEALTHY in ~70 seconds |
| Database API key creation via management API | ✅ `a47501bc-a92d-4103-868e-82aafb8b3e10` (JWT-format key) |
| Egypt → AWS us-east-1 latency | `get_collections`: 484 ms; `create_collection` with sparse field: 266 ms |
| Sparse vector field provisioning | ✅ `SparseVectorParams(index=None, modifier=None)` returned by `get_collection` |
| `qdrant-client` Python SDK connection | ✅ Works with stored URL + JWT key from `.env` |

## Cluster details

```
cluster_id          : db2d4134-4e7d-48e9-a771-ccf4bd1d1d2c
account_id          : f88e3546-2132-4dda-a75f-7ec7ed32b757
url                 : https://db2d4134-4e7d-48e9-a771-ccf4bd1d1d2c.us-east-1-1.aws.cloud.qdrant.io
rest_port           : 6333
grpc_port           : 6334
package             : free2 (type=free)
hardware            : 0.5 vCPU, 1 GB RAM, 4 GB disk
qdrant_version      : v1.18.1
nodes               : 1 (us-east-1a)
jwt_rbac            : true
backup_supported    : false (free tier limitation)
auto_suspend        : 1 week of inactivity (per Qdrant Cloud free-tier policy)
auto_delete         : 4 weeks of inactivity (per Qdrant Cloud free-tier policy)
```

## Credentials stored

`.env` (gitignored) now contains:

- `SAR_QDRANT_CLOUD_URL` — cluster REST endpoint
- `SAR_QDRANT_CLOUD_API_KEY` — JWT cluster API key (database-api-key v2)
- `SAR_QDRANT_CLOUD_ACCOUNT_ID` — account UUID (for management API calls)
- `SAR_QDRANT_CLOUD_CLUSTER_ID` — cluster UUID (for management API calls)

The HF Space production deploy (phase 3) will read `SAR_QDRANT_URL` and `SAR_QDRANT_API_KEY` — phase 2 backend code aliases the existing env names to the new `_CLOUD_` names so local dev keeps pointing at `localhost:6333`.

## API surface discovered

These are the Qdrant Cloud Management REST endpoints we use, with the exact paths:

| Action | Method + Path |
|---|---|
| List accounts | `GET /api/account/v1/accounts` |
| List clusters | `GET /api/cluster/v1/accounts/{account_id}/clusters` |
| Get cluster | `GET /api/cluster/v1/accounts/{account_id}/clusters/{cluster_id}` |
| Create cluster | `POST /api/cluster/v1/accounts/{account_id}/clusters` |
| Delete cluster | `DELETE /api/cluster/v1/accounts/{account_id}/clusters/{cluster_id}` |
| List packages (account-scoped) | `GET /api/booking/v1/accounts/{account_id}/packages?cloud_provider_id=aws&cloud_provider_region_id=us-east-1` |
| Create database API key | `POST /api/cluster/auth/v2/accounts/{account_id}/database-api-keys` |
| Delete database API key | `DELETE /api/cluster/auth/v2/accounts/{account_id}/database-api-keys/{key_id}` |

Authorization on all of them: `Authorization: apikey <management_key>`.

Confirmed in `qdrant-cloud-public-api` proto definitions:
- `proto/qdrant/cloud/cluster/v1/cluster.proto`
- `proto/qdrant/cloud/booking/v1/booking.proto`
- `proto/qdrant/cloud/cluster/auth/v2/database_api_key.proto`

## What this proves

1. **Qdrant Cloud free tier in 2026 does not require a credit card.** Account was created and a cluster spun up with zero payment friction.
2. **Sparse vector support is on free tier.** This was a known stop condition. Confirmed via `SparseVectorParams` round-trip on `smoke_sparse` collection — same shape we use in `retrieval/sparse_embeddings.py`.
3. **AWS us-east-1 is the working free-tier region.** GCP us-central1 returned 0 packages on this account. If we ever need GCP, owner would need to escalate or switch region.
4. **End-to-end provisioning is API-driven.** Owner does not need to click through the dashboard — agent can create / list / delete clusters and rotate API keys autonomously given the management key.
5. **Egypt latency is well within budget.** Single-digit-hundred-ms operations against AWS US-East from Egypt. The HF Space (also AWS US-East per typical HF infra) will see sub-50ms intra-region calls in production.

## Capacity considerations (track post-launch)

The free `free2` package gives 1 GB RAM and 4 GB disk on a 0.5 vCPU node. For our use case:

- BGE-M3 dense vectors at 1024 floats × 4 bytes = ~4 KB/vector → ~250k dense vectors fit in 1 GB RAM
- SPLADE sparse vectors are roughly 30-150 active tokens per chunk → ~5-20 KB/vector on average → ~50-200k sparse vectors per 1 GB RAM
- Realistic demo corpus: 5-20k chunks across all per-session collections
- 4 GB disk fits multiple snapshots of the working set
- **Capacity guardrail:** the session purge cron (phase 2) deletes collections older than 24h; without it, traffic would exhaust the 1 GB cluster within days

## Reproducibility script

For anyone re-creating this cluster (e.g. after auto-suspend wipes it):

```python
import httpx, os
key = os.environ['QDRANT_CLOUD_MGMT_KEY']   # the management key, not the cluster API key
ACC = os.environ['SAR_QDRANT_CLOUD_ACCOUNT_ID']
H = {'Authorization': f'apikey {key}', 'Content-Type': 'application/json'}

# create free cluster on AWS us-east-1
r = httpx.post(
    f'https://api.cloud.qdrant.io/api/cluster/v1/accounts/{ACC}/clusters',
    headers=H,
    json={'cluster': {
        'account_id': ACC,
        'name': 'secureagentrag-demo',
        'cloud_provider_id': 'aws',
        'cloud_provider_region_id': 'us-east-1',
        'configuration': {
            'number_of_nodes': 1,
            'package_id': '39b48a76-2a60-4ee0-9266-6d1e0f91ea14',  # free2
        },
    }},
    timeout=60,
)
print(r.json())
```

The management key remains the long-lived credential; the cluster API key can be rotated freely via the `database-api-keys` endpoints whenever the JWT in `.env` is suspected of leakage.

## Next phase

Phase 1d: Vercel Hobby signup + Next.js hello-world deploy.

Phase 1c done; phase 1d pending owner signup at `vercel.com`.
