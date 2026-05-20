"""Sparse embedding generation for Qdrant native sparse vectors.

Backends
--------
* ``bm25`` — whitespace tokenization + term-frequency vectors.
  Zero external dependencies; quality is baseline BM25.
* ``splade`` — SPLADE++ (``naver/splade-cocondenser-ensembledistil``)
  via ``transformers`` AutoModelForMaskedLM. Requires the
  ``[embeddings-local]`` extra (installs ``transformers`` + ``torch``).
  Falls back to ``bm25`` on import or runtime errors.

Both backends return :class:`qdrant_client.http.models.SparseVector`
objects that can be stored in Qdrant 1.10+ sparse vector fields and
queried with the same RBAC filters as dense vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.settings import settings
from utils.logging import get_logger

if TYPE_CHECKING:
    from qdrant_client.http.models import SparseVector

logger = get_logger(__name__)

try:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    _SPLADE_DEPS = True
except ImportError:
    _SPLADE_DEPS = False


class SparseEmbeddingService:
    """Generates sparse embedding vectors for Qdrant native sparse storage.

    Args:
        backend: ``"bm25"`` or ``"splade"``. Defaults to
            ``settings.sparse_backend``.
        model_name: HuggingFace model id for SPLADE. Defaults to
            ``settings.sparse_model``.
    """

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._backend = (backend or getattr(settings, "sparse_backend", "bm25")).lower()
        self._model_name = model_name or getattr(
            settings, "sparse_model", "naver/splade-cocondenser-ensembledistil"
        )
        self._tokenizer: object | None = None
        self._model: object | None = None

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend

    def embed_texts(self, texts: list[str]) -> list[SparseVector]:
        """Generate a sparse vector for every text in *texts*.

        Returns:
            List of :class:`SparseVector` instances aligned with *texts*.
        """
        if self._backend == "splade":
            try:
                return self._embed_splade(texts)
            except Exception as exc:
                logger.warning("splade_failed_falling_back_to_bm25", error=str(exc))
                return self._embed_bm25(texts)
        return self._embed_bm25(texts)

    def embed_text(self, text: str) -> SparseVector:
        """Generate a single sparse vector."""
        return self.embed_texts([text])[0]

    # ------------------------------------------------------------------ #
    # bm25 backend — pure Python, no external deps
    # ------------------------------------------------------------------ #

    @staticmethod
    def _embed_bm25(texts: list[str]) -> list[SparseVector]:
        import zlib

        from qdrant_client.http.models import SparseVector

        results: list[SparseVector] = []
        for text in texts:
            tokens = text.lower().split()
            tf: dict[int, float] = {}
            for token in tokens:
                # Deterministic positive integer hash for each token.
                # zlib.crc32 is stable across process restarts (unlike hash()).
                idx = zlib.crc32(token.encode("utf-8")) & 0x7FFF_FFFF
                tf[idx] = tf.get(idx, 0.0) + 1.0

            if tf:
                max_tf = max(tf.values())
                indices = sorted(tf.keys())
                values = [tf[i] / max_tf for i in indices]
            else:
                indices = []
                values = []

            results.append(SparseVector(indices=indices, values=values))
        return results

    # ------------------------------------------------------------------ #
    # splade backend — transformers AutoModelForMaskedLM
    # ------------------------------------------------------------------ #

    def _get_splade_model(self) -> AutoModelForMaskedLM:
        if self._model is None:
            if not _SPLADE_DEPS:
                raise RuntimeError(
                    "SPLADE dependencies missing. Install with: uv sync --extra embeddings-local"
                )
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self._model_name)
            self._model.eval()
            logger.info("splade_model_loaded", model=self._model_name)
        return self._model  # type: ignore[return-value]

    def _embed_splade(self, texts: list[str]) -> list[SparseVector]:
        from qdrant_client.http.models import SparseVector

        model = self._get_splade_model()
        tokenizer = self._tokenizer

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        # SPLADE++ activation: log(1 + ReLU(x))
        activations = torch.log(1 + torch.relu(logits))

        # Max-pool over sequence dimension → vocab-sized sparse vector
        max_activations = activations.max(dim=1).values

        results: list[SparseVector] = []
        for vec in max_activations:
            # Keep only non-zero entries (sparse representation)
            nonzero = vec.nonzero(as_tuple=True)[0]
            indices = nonzero.tolist()
            values = vec[nonzero].tolist()
            results.append(SparseVector(indices=indices, values=values))

        return results
