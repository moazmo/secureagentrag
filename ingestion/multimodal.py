"""Multi-modal image understanding for RAG.

Uses a vision-language model (Qwen-VL, LLaVA, etc.) via Ollama to generate
rich text descriptions of images. These descriptions are embedded as chunks
alongside OCR text, enabling retrieval for queries like "what does the
diagram show?" or "describe the chart on page 5".

The approach translates visual content into text space so standard dense
embeddings (BGE-M3) can retrieve it without requiring CLIP or other
multi-modal embedding models.
"""

from __future__ import annotations

import base64
from pathlib import Path

from config.settings import settings
from utils.async_helpers import run_async
from utils.logging import get_logger

logger = get_logger(__name__)

_IMAGE_DESCRIPTION_PROMPT = (
    "Describe this image in detail for a document retrieval system. "
    "Include:\n"
    "1. What type of image it is (diagram, chart, photo, screenshot, etc.)\n"
    "2. All visible text, labels, and annotations\n"
    "3. Relationships and structures shown (flows, hierarchies, comparisons)\n"
    "4. Any numbers, percentages, or data points visible\n"
    "5. Colors, layouts, or visual patterns that convey meaning\n\n"
    "Be comprehensive but concise. The description will be embedded for search."
)

_IMAGE_DESCRIPTION_SYSTEM = (
    "You are an image describer for a RAG system. Your descriptions must be "
    "detailed enough that someone searching for visual content can find this "
    "image based on your text alone."
)


class ImageDescriptor:
    """Generates text descriptions of images using a vision-language model.

    Args:
        model: VLM model name on Ollama. Defaults to settings.vlm_ocr_model.
        base_url: Ollama server URL. Defaults to settings.ollama_url.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._available = False
        self.model = model or getattr(settings, "vlm_ocr_model", "qwen2.5-vl")
        self.base_url = (base_url or settings.ollama_url).rstrip("/")
        self._client = None

        try:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0),
            )
            self._available = True
            logger.info("image_descriptor_initialized", model=self.model)
        except ImportError:
            logger.warning("image_descriptor_init_failed", reason="httpx not installed")

    def is_available(self) -> bool:
        """Return True if the image descriptor is ready to use."""
        return self._available and self._client is not None

    async def describe_image_async(self, image_path: str | Path) -> str:
        """Generate a rich text description of an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Text description, or empty string on failure.
        """
        if not self.is_available():
            return ""

        path = Path(image_path)
        if not path.exists():
            logger.warning("image_descriptor_file_missing", file=str(path))
            return ""

        try:
            image_bytes = path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            payload = {
                "model": self.model,
                "prompt": _IMAGE_DESCRIPTION_PROMPT,
                "system": _IMAGE_DESCRIPTION_SYSTEM,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 2048,
                },
                "keep_alive": settings.ollama_keep_alive,
            }

            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            description = data.get("response", "").strip()

            logger.info(
                "image_described",
                file=str(path),
                chars=len(description),
                model=self.model,
            )
            return description
        except Exception as exc:
            logger.warning("image_description_failed", file=str(path), error=str(exc))
            return ""

    def describe_image(self, image_path: str | Path) -> str:
        """Synchronous wrapper for ``describe_image_async``."""
        return run_async(self.describe_image_async(image_path))
