# Reranker Fine-tune Benchmark

- **Baseline:**  `BAAI/bge-reranker-v2-m3`
- **Candidate:** `data/checkpoints/reranker-domain-v1`
- **Eval:** MS-MARCO 500 pairs, NIST gold 20 pairs

| Dataset       | Baseline NDCG@10 | Candidate NDCG@10 | Delta (pp) |
|---------------|-----------------:|------------------:|-----------:|
| MS-MARCO      | 0.7744        | 0.7904         | +1.60    |
| NIST (in-dom) | 0.9162      | 0.9215       | +0.54  |

Acceptance bar (per ADR-022): candidate must beat baseline by ≥1pp on MS-MARCO and win on NIST.
