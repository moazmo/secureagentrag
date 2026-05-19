"""VLM-based OCR using Ollama vision models (Qwen-VL, LLaVA, etc.).

Primary OCR path for scanned documents, images, and complex layouts.
Falls back to PaddleOCR when the VLM is unavailable or fails.

The VLM is prompted with a base64-encoded image and asked to transcribe
all visible text faithfully, preserving line breaks and paragraph structure.
"""

from __future__ import annotations

import base64
from pathlib import Path

from config.settings import settings
from utils.async_helpers import run_async
from utils.logging import get_logger

logger = get_logger(__name__)

_VLM_OCR_PROMPT = (
    "Transcribe ALL visible text in this image faithfully. "
    "Preserve line breaks and paragraph structure exactly as they appear. "
    "Do NOT summarise, interpret, or add commentary — only output the raw text. "
    "If the image contains tables, transcribe them as markdown tables. "
    "If no text is visible, respond with exactly: NO_TEXT_FOUND"
)

_VLM_SYSTEM_PROMPT = (
    "You are an OCR engine. Your only job is to transcribe text from images. "
    "Be precise and do not hallucinate content that is not visible."
)


class VLMOCRProcessor:
    """OCR processor backed by a vision-language model via Ollama.

    Args:
        model: VLM model name on the Ollama server. Defaults to
            ``settings.vlm_ocr_model``.
        base_url: Ollama server URL. Defaults to ``settings.ollama_url``.
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
            logger.info("vlm_ocr_initialized", model=self.model)
        except ImportError:
            logger.warning("vlm_ocr_init_failed", reason="httpx not installed")

    def is_available(self) -> bool:
        """Return True if the VLM OCR processor is ready to use."""
        return self._available and self._client is not None

    async def _call_vlm(self, image_b64: str) -> str:
        """Send the image to the VLM and return the transcribed text."""

        payload = {
            "model": self.model,
            "prompt": _VLM_OCR_PROMPT,
            "system": _VLM_SYSTEM_PROMPT,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 4096,
            },
            "keep_alive": settings.ollama_keep_alive,
        }

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "").strip()

        # Normalise the "no text" sentinel
        if text == "NO_TEXT_FOUND":
            return ""
        return text

    async def extract_text_from_image_async(self, image_path: str | Path) -> str:
        """Async version — extract text from an image via VLM.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text, or empty string on failure.
        """
        if not self.is_available():
            return ""

        path = Path(image_path)
        if not path.exists():
            logger.warning("vlm_ocr_file_missing", file=str(path))
            return ""

        try:
            image_bytes = path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("ascii")

            text = await self._call_vlm(image_b64)
            logger.info(
                "vlm_ocr_extracted",
                file=str(path),
                chars=len(text),
                model=self.model,
            )
            return text
        except Exception as exc:
            logger.warning("vlm_ocr_extraction_failed", file=str(path), error=str(exc))
            return ""

    def extract_text_from_image(self, image_path: str | Path) -> str:
        """Synchronous wrapper for ``extract_text_from_image_async``."""
        return run_async(self.extract_text_from_image_async(image_path))

    async def extract_text_from_pdf_page_async(
        self,
        pdf_path: str | Path,
        page_number: int,
    ) -> str:
        """Async version — render a PDF page to image and OCR via VLM.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Zero-indexed page number.

        Returns:
            Extracted text, or empty string on failure.
        """
        if not self.is_available():
            return ""

        try:
            import fitz

            path = Path(pdf_path)
            with fitz.open(str(path)) as doc:
                if page_number >= len(doc):
                    logger.warning(
                        "vlm_ocr_page_out_of_range",
                        file=str(path),
                        page=page_number,
                        total=len(doc),
                    )
                    return ""

                page = doc[page_number]
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
                image_b64 = base64.b64encode(image_bytes).decode("ascii")

                text = await self._call_vlm(image_b64)
                logger.info(
                    "vlm_ocr_pdf_page_extracted",
                    file=str(path),
                    page=page_number,
                    chars=len(text),
                )
                return text
        except ImportError:
            logger.warning("vlm_ocr_fitz_missing", msg="PyMuPDF not installed")
            return ""
        except Exception as exc:
            logger.warning(
                "vlm_ocr_pdf_page_failed",
                file=str(pdf_path),
                page=page_number,
                error=str(exc),
            )
            return ""

    def extract_text_from_pdf_page(
        self,
        pdf_path: str | Path,
        page_number: int,
    ) -> str:
        """Synchronous wrapper for ``extract_text_from_pdf_page_async``."""
        return run_async(self.extract_text_from_pdf_page_async(pdf_path, page_number))
