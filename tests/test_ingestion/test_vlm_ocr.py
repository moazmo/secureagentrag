"""Tests for VLM OCR module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from ingestion.vlm_ocr import VLMOCRProcessor


class TestVLMOCRProcessorInit:
    """Tests for VLMOCRProcessor initialization."""

    def test_default_model_from_settings(self):
        """Default model comes from settings.vlm_ocr_model."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.vlm_ocr_model = "test-vlm"
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            assert proc.model == "test-vlm"
            assert proc.is_available() is True

    def test_custom_model_override(self):
        """Constructor model argument overrides settings."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor(model="custom-vlm")
            assert proc.model == "custom-vlm"

    def test_unavailable_when_httpx_missing(self):
        """If httpx is not installed, is_available() returns False."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            with patch.dict("sys.modules", {"httpx": None}):
                proc = VLMOCRProcessor()
                assert proc.is_available() is False


class TestExtractTextFromImageAsync:
    """Tests for extract_text_from_image_async."""

    @patch("ingestion.vlm_ocr.VLMOCRProcessor._call_vlm", new_callable=AsyncMock)
    async def test_success(self, mock_vlm):
        """A successful VLM call returns the transcribed text."""
        mock_vlm.return_value = "Transcribed text from image."

        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with (
                patch.object(proc, "is_available", return_value=True),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_bytes", return_value=b"fake_image"),
            ):
                result = await proc.extract_text_from_image_async("/tmp/test.png")

        assert result == "Transcribed text from image."
        mock_vlm.assert_awaited_once()
        # Verify base64 image was passed
        assert mock_vlm.call_args[0][0] == "ZmFrZV9pbWFnZQ=="

    async def test_unavailable_returns_empty(self):
        """If processor is not available, return empty string immediately."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with patch.object(proc, "is_available", return_value=False):
                result = await proc.extract_text_from_image_async("/tmp/test.png")

        assert result == ""

    @patch("ingestion.vlm_ocr.VLMOCRProcessor._call_vlm", new_callable=AsyncMock)
    async def test_file_missing_returns_empty(self, mock_vlm):
        """Non-existent file returns empty string."""
        mock_vlm.return_value = "should not be called"

        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with (
                patch.object(proc, "is_available", return_value=True),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = await proc.extract_text_from_image_async("/tmp/missing.png")

        assert result == ""
        mock_vlm.assert_not_awaited()

    @patch("ingestion.vlm_ocr.VLMOCRProcessor._call_vlm", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_vlm):
        """Any failure returns empty string so ingestion continues."""
        mock_vlm.side_effect = RuntimeError("ollama timeout")

        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with (
                patch.object(proc, "is_available", return_value=True),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_bytes", return_value=b"fake"),
            ):
                result = await proc.extract_text_from_image_async("/tmp/test.png")

        assert result == ""

    async def test_no_text_sentinel_normalised_in_call_vlm(self):
        """The NO_TEXT_FOUND sentinel is normalised to an empty string inside _call_vlm."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()

            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "NO_TEXT_FOUND"}
            proc._client = MagicMock()
            proc._client.post = AsyncMock(return_value=mock_response)

            result = await proc._call_vlm("b64")
            assert result == ""


class TestExtractTextFromPdfPageAsync:
    """Tests for extract_text_from_pdf_page_async."""

    @patch("ingestion.vlm_ocr.VLMOCRProcessor._call_vlm", new_callable=AsyncMock)
    async def test_success(self, mock_vlm):
        """A successful PDF page render + VLM call returns text."""
        mock_vlm.return_value = "Page text."

        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with patch.object(proc, "is_available", return_value=True):
                mock_doc = MagicMock()
                mock_doc.__len__ = MagicMock(return_value=3)
                mock_page = MagicMock()
                mock_pix = MagicMock()
                mock_pix.width = 100
                mock_pix.height = 100
                mock_pix.tobytes.return_value = b"png_data"
                mock_page.get_pixmap.return_value = mock_pix
                mock_doc.__getitem__ = MagicMock(return_value=mock_page)
                mock_doc.__enter__ = MagicMock(return_value=mock_doc)
                mock_doc.__exit__ = MagicMock(return_value=False)

                with patch("fitz.open", return_value=mock_doc):
                    result = await proc.extract_text_from_pdf_page_async("/tmp/doc.pdf", 1)

        assert result == "Page text."
        mock_vlm.assert_awaited_once()

    async def test_unavailable_returns_empty(self):
        """If processor is unavailable, return empty string."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with patch.object(proc, "is_available", return_value=False):
                result = await proc.extract_text_from_pdf_page_async("/tmp/doc.pdf", 0)

        assert result == ""

    @patch("ingestion.vlm_ocr.VLMOCRProcessor._call_vlm", new_callable=AsyncMock)
    async def test_page_out_of_range(self, mock_vlm):
        """Page number beyond document length returns empty string."""
        mock_vlm.return_value = "should not be called"

        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()
            with patch.object(proc, "is_available", return_value=True):
                mock_doc = MagicMock()
                mock_doc.__len__ = MagicMock(return_value=2)

                with patch("fitz.open", return_value=mock_doc):
                    result = await proc.extract_text_from_pdf_page_async("/tmp/doc.pdf", 5)

        assert result == ""
        mock_vlm.assert_not_awaited()


class TestCallVlm:
    """Tests for the internal _call_vlm method."""

    async def test_payload_structure(self):
        """The Ollama payload contains the expected fields."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor(model="test-vlm")

            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "hello"}
            proc._client = MagicMock()
            proc._client.post = AsyncMock(return_value=mock_response)

            result = await proc._call_vlm("b64image")

            assert result == "hello"
            call_args = proc._client.post.call_args
            payload = call_args.kwargs["json"]
            assert payload["model"] == "test-vlm"
            assert payload["images"] == ["b64image"]
            assert payload["stream"] is False
            assert payload["options"]["temperature"] == 0.1

    async def test_strips_response(self):
        """Whitespace around the VLM response is stripped."""
        with patch("ingestion.vlm_ocr.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            proc = VLMOCRProcessor()

            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "  text with spaces  \n"}
            proc._client = MagicMock()
            proc._client.post = AsyncMock(return_value=mock_response)

            result = await proc._call_vlm("b64")

            assert result == "text with spaces"
