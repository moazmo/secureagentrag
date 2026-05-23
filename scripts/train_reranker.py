"""Fine-tune a cross-encoder reranker on MS-MARCO + hard negatives.

What this script does
---------------------
1. Downloads the MS-MARCO Passage Ranking small split from HuggingFace
   (``microsoft/ms_marco``). About 500k triplets ``(query, positive,
   negative)``. Held-out 500 rows form the evaluation set.
2. Optionally mines hard negatives from the in-domain NIST AI RMF
   corpus stored in Qdrant. For each training query we retrieve the top-K
   dense hits and treat the lowest-scored non-positive as the hard
   negative. Falls back to the random MS-MARCO negative when Qdrant is
   not reachable.
3. Loads the off-the-shelf BGE-Reranker-v2-M3 checkpoint
   (``BAAI/bge-reranker-v2-m3``) and fine-tunes it with
   ``CrossEncoder.fit`` (sentence-transformers).
4. Saves the resulting checkpoint to
   ``data/checkpoints/reranker-domain-v1/`` so the existing
   ``retrieval.reranker`` factory can pick it up by setting
   ``SAR_RERANKER_TYPE=fine_tuned``.

Why a script, not a training service
------------------------------------
A real training run takes 1-2 GPU hours on an RTX 3060. Wiring that
into CI is heavy and irrelevant for most contributors. The script is
the deliverable; running it is left to whoever owns the GPU box. The
companion ``scripts/bench_reranker.py`` measures NDCG@10 hold-out
performance of any saved checkpoint against the baseline.

Usage
-----
::

    # Quick smoke (1000 train rows, 100 hold-out)
    uv run python -m scripts.train_reranker --smoke

    # Full run
    uv run python -m scripts.train_reranker \\
        --train-size 100000 \\
        --eval-size 500 \\
        --epochs 1 \\
        --output data/checkpoints/reranker-domain-v1

Dependencies (install with ``uv sync --extra embeddings-local``):
``sentence-transformers``, ``datasets``, ``torch``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _load_triplets(train_size: int, eval_size: int) -> tuple[list, list]:
    """Pull MS-MARCO triplets from HuggingFace.

    Returns:
        (train_examples, eval_pairs). Each train example is a
        ``InputExample`` (query, passage, label) with label=1.0 for
        positives and 0.0 for negatives. Eval pairs are tuples of
        ``(query, [positives], [negatives])`` for NDCG scoring.
    """
    try:
        from datasets import load_dataset
        from sentence_transformers import InputExample
    except ImportError as exc:
        raise RuntimeError(
            "MS-MARCO loader requires the [embeddings-local] extra. "
            "Install with: uv sync --extra embeddings-local"
        ) from exc

    logger.info("loading_ms_marco", train_size=train_size, eval_size=eval_size)
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)

    train_examples = []
    eval_pairs = []
    count = 0
    for row in ds:
        query = row["query"]
        passages = row.get("passages", {})
        is_selected = passages.get("is_selected", [])
        passage_texts = passages.get("passage_text", [])

        positives = [t for t, sel in zip(passage_texts, is_selected, strict=False) if sel == 1]
        negatives = [t for t, sel in zip(passage_texts, is_selected, strict=False) if sel == 0]
        if not positives or not negatives:
            continue

        if count < train_size:
            train_examples.append(InputExample(texts=[query, positives[0]], label=1.0))
            train_examples.append(InputExample(texts=[query, negatives[0]], label=0.0))
            count += 1
        elif count < train_size + eval_size:
            eval_pairs.append((query, positives, negatives))
            count += 1
        else:
            break

    logger.info(
        "ms_marco_loaded",
        train_examples=len(train_examples),
        eval_pairs=len(eval_pairs),
    )
    return train_examples, eval_pairs


def _mine_hard_negative(query: str, fallback: str) -> str:
    """Return a hard negative from Qdrant if reachable, else the fallback.

    The hard negative is the lowest-scored doc among the dense top-10 for
    this query. Falls back to the MS-MARCO random negative on any error
    so the training run still completes without Qdrant.
    """
    try:
        from ingestion.metadata import UserContext
        from retrieval.embeddings import EmbeddingService
        from retrieval.qdrant_client import QdrantManager

        qm = QdrantManager()
        es = EmbeddingService()
        emb = es.embed_text_sync(query)
        ctx = UserContext(
            user_id="reranker_trainer",
            org_id="acme_corp",
            roles=["admin"],
            clearance_level=3,
        )
        hits = qm.search_with_rbac(query_embedding=emb, user_context=ctx, top_k=10)
        if not hits:
            return fallback
        # Lowest-scored doc among the top-10 is the canonical hard negative.
        worst = min(hits, key=lambda p: p.score)
        text = (worst.payload or {}).get("text", "")
        return text or fallback
    except Exception as exc:
        logger.debug("hard_negative_mine_failed", error=str(exc))
        return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--base-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder checkpoint to fine-tune from.",
    )
    parser.add_argument(
        "--output",
        default="data/checkpoints/reranker-domain-v1",
        help="Where to write the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--mine-hard-negatives",
        action="store_true",
        help="Replace MS-MARCO random negatives with hard negatives mined "
        "from the local Qdrant index. Requires the corpus to be ingested.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke run: 1000 train rows, 100 eval rows, 1 epoch.",
    )
    args = parser.parse_args()

    if args.smoke:
        args.train_size = 1_000
        args.eval_size = 100
        args.epochs = 1

    try:
        import torch
        from sentence_transformers import CrossEncoder
        from torch.utils.data import DataLoader
    except ImportError as exc:
        print(
            "[train_reranker] sentence-transformers + torch not installed. "
            "Run: uv sync --extra embeddings-local",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    train_examples, _eval_pairs = _load_triplets(args.train_size, args.eval_size)

    if args.mine_hard_negatives:
        logger.info("mining_hard_negatives", count=len(train_examples) // 2)
        for ex in train_examples:
            if ex.label == 0.0:
                ex.texts[1] = _mine_hard_negative(ex.texts[0], ex.texts[1])

    logger.info("fine_tuning", base_model=args.base_model, epochs=args.epochs)
    model = CrossEncoder(args.base_model, num_labels=1)
    loader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_dataloader=loader,
        epochs=args.epochs,
        warmup_steps=max(1, len(train_examples) // 20),
        use_amp=torch.cuda.is_available(),
        output_path=args.output,
        show_progress_bar=True,
    )

    # ST 5.x deprecated `fit()` calls the underlying CrossEncoderTrainer with
    # `save_strategy="no"` hard-coded, and `save_best_model` only fires when
    # an evaluator was supplied. Persist the final weights explicitly so the
    # bench script (and the SAR_RERANKER_TYPE=fine_tuned factory) can load
    # the checkpoint. Without this the output dir holds only train_meta.json.
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    logger.info("model_saved", path=args.output)

    # Persist a small companion JSON with reproducibility metadata so we
    # can trace any future bench number back to the right training run.
    meta = {
        "base_model": args.base_model,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mined_hard_negatives": args.mine_hard_negatives,
    }
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Checkpoint saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
