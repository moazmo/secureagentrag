"""OCR integration using PaddleOCR for Arabic and English text extraction."""

from __future__ import annotations

from pathlib import Path

from ingestion.loaders import LoadedDocument
from utils.logging import get_logger

logger = get_logger(__name__)

# Conditional PaddleOCR import
try:
    from paddleocr import PaddleOCR

    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False
    logger.warning(
        "paddleocr_not_installed", msg="PaddleOCR is not available. OCR features disabled."
    )


class OCRProcessor:
    """OCR processor using PaddleOCR for multilingual text extraction.

    Supports English and Arabic by default. Gracefully degrades if PaddleOCR
    is not installed in the environment.

    Args:
        languages: List of language codes for OCR. Defaults to ["en", "ar"].
    """

    def __init__(self, languages: list[str] | None = None) -> None:
        """Initialize the OCR processor.

        Args:
            languages: Language codes for PaddleOCR initialization.
                Defaults to ["en", "ar"].
        """
        self._available = False
        self._ocr = None
        self._languages = languages or ["en", "ar"]

        if not _PADDLEOCR_AVAILABLE:
            logger.warning("ocr_init_skipped", reason="PaddleOCR not installed")
            return

        try:
            # PaddleOCR accepts 'lang' as a single string; use first language
            # For multilingual support, 'en' handles Latin scripts and 'ar' for Arabic
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                use_gpu=True,
                lang=self._languages[0] if self._languages else "en",
                show_log=False,
            )
            self._available = True
            logger.info("ocr_initialized", languages=self._languages)
        except Exception as exc:
            logger.warning(
                "ocr_init_failed",
                error=str(exc),
                msg="Falling back to CPU or disabling OCR",
            )
            # Retry with GPU disabled
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    use_gpu=False,
                    lang=self._languages[0] if self._languages else "en",
                    show_log=False,
                )
                self._available = True
                logger.info("ocr_initialized_cpu_fallback", languages=self._languages)
            except Exception as fallback_exc:
                logger.error("ocr_init_completely_failed", error=str(fallback_exc))
                self._available = False

    def is_available(self) -> bool:
        """Check if OCR processing is available.

        Returns:
            True if PaddleOCR is initialized and ready.
        """
        return self._available

    def extract_text_from_image(self, image_path: str | Path) -> str:
        """Extract text from an image file using PaddleOCR.

        Args:
            image_path: Path to the image file.

        Returns:
            Extracted text concatenated from all detected regions.
            Empty string on failure or if OCR is unavailable.
        """
        if not self._available or self._ocr is None:
            logger.warning("ocr_unavailable", action="extract_text_from_image")
            return ""

        try:
            path_str = str(Path(image_path))
            result = self._ocr.ocr(path_str, cls=True)

            if not result or not result[0]:
                return ""

            lines: list[str] = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    lines.append(text)

            extracted = "\n".join(lines)
            logger.info(
                "ocr_text_extracted",
                file=path_str,
                chars=len(extracted),
            )
            return extracted

        except Exception as exc:
            logger.error("ocr_extraction_failed", file=str(image_path), error=str(exc))
            return ""

    def extract_text_from_pdf_page(self, pdf_path: str | Path, page_number: int) -> str:
        """Extract text from a specific PDF page by rendering to image and running OCR.

        Args:
            pdf_path: Path to the PDF file.
            page_number: Zero-indexed page number to process.

        Returns:
            Extracted text from the page. Empty string on failure.
        """
        if not self._available or self._ocr is None:
            logger.warning("ocr_unavailable", action="extract_text_from_pdf_page")
            return ""

        try:
            import fitz

            with fitz.open(str(pdf_path)) as doc:
                if page_number >= len(doc):
                    logger.warning(
                        "ocr_page_out_of_range",
                        file=str(pdf_path),
                        page=page_number,
                        total=len(doc),
                    )
                    return ""

                page = doc[page_number]
                # Render page to a high-resolution pixmap
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR accuracy
                pix = page.get_pixmap(matrix=mat)

                # Convert to numpy array for PaddleOCR
                import numpy as np
                from PIL import Image

                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                img_array = np.array(img)

                result = self._ocr.ocr(img_array, cls=True)

                if not result or not result[0]:
                    return ""

                lines: list[str] = []
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                        lines.append(text)

                return "\n".join(lines)

        except Exception as exc:
            logger.error(
                "ocr_pdf_page_failed",
                file=str(pdf_path),
                page=page_number,
                error=str(exc),
            )
            return ""

    def process_document(self, file_path: str | Path) -> list[LoadedDocument]:
        """Process a document with OCR, handling both images and scanned PDFs.

        For images: Run OCR directly on the file.
        For PDFs: Check each page — if standard text extraction yields very little
        text (< 50 characters), fall back to OCR for that page.

        Args:
            file_path: Path to the document file.

        Returns:
            List of LoadedDocument instances with OCR-extracted text.
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        documents: list[LoadedDocument] = []

        if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            # Direct image OCR
            text = self.extract_text_from_image(path)
            documents.append(
                LoadedDocument(
                    text=text,
                    page_number=0,
                    source_file=str(path),
                    file_type="image",
                    metadata={"ocr_processed": True},
                )
            )

        elif ext == ".pdf":
            try:
                import fitz

                with fitz.open(str(path)) as doc:
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        text = page.get_text("text").strip()

                        # If text extraction yields very little, try OCR
                        if len(text) < 50:
                            logger.info(
                                "ocr_fallback_triggered",
                                file=str(path),
                                page=page_num,
                                text_len=len(text),
                            )
                            ocr_text = self.extract_text_from_pdf_page(path, page_num)
                            if ocr_text:
                                text = ocr_text

                        documents.append(
                            LoadedDocument(
                                text=text,
                                page_number=page_num,
                                source_file=str(path),
                                file_type="pdf",
                                metadata={
                                    "ocr_processed": len(page.get_text("text").strip()) < 50,
                                    "total_pages": len(doc),
                                },
                            )
                        )
            except Exception as exc:
                logger.error("ocr_process_pdf_failed", file=str(path), error=str(exc))

        else:
            logger.warning("ocr_unsupported_format", file=str(path), extension=ext)

        return documents
