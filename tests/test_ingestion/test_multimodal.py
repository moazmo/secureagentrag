"""Tests for multi-modal image descriptor."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from ingestion.multimodal import ImageDescriptor


class TestImageDescriptorInit:
    """Tests for ImageDescriptor initialization."""

    def test_default_model_from_settings(self):
        """Default model comes from settings.vlm_ocr_model."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.vlm_ocr_model = "test-vlm"
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor()
            assert desc.model == "test-vlm"
            assert desc.is_available() is True

    def test_custom_model_override(self):
        """Constructor model argument overrides settings."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor(model="custom-vlm")
            assert desc.model == "custom-vlm"

    def test_unavailable_when_httpx_missing(self):
        """If httpx is not installed, is_available() returns False."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            with patch.dict("sys.modules", {"httpx": None}):
                desc = ImageDescriptor()
                assert desc.is_available() is False


class TestDescribeImageAsync:
    """Tests for describe_image_async."""

    async def test_success(self):
        """A successful VLM call returns the description."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.vlm_ocr_model = "test-vlm"
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor()
            # Override the model that was set from the MagicMock settings
            desc.model = "test-vlm"

            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "A diagram showing four boxes."}
            desc._client = MagicMock()
            desc._client.post = AsyncMock(return_value=mock_response)

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_bytes", return_value=b"fake_image"),
            ):
                result = await desc.describe_image_async("/tmp/test.png")

        assert result == "A diagram showing four boxes."
        desc._client.post.assert_awaited_once()
        payload = desc._client.post.call_args.kwargs["json"]
        assert payload["model"] == "test-vlm"
        assert payload["images"] == ["ZmFrZV9pbWFnZQ=="]

    async def test_unavailable_returns_empty(self):
        """If descriptor is unavailable, return empty string immediately."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor()
            with patch.object(desc, "is_available", return_value=False):
                result = await desc.describe_image_async("/tmp/test.png")

        assert result == ""

    async def test_file_missing_returns_empty(self):
        """Non-existent file returns empty string."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor()
            with (
                patch.object(desc, "is_available", return_value=True),
                patch("pathlib.Path.exists", return_value=False),
            ):
                result = await desc.describe_image_async("/tmp/missing.png")

        assert result == ""

    async def test_exception_returns_empty(self):
        """Any failure returns empty string so ingestion continues."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor()
            desc._client = MagicMock()
            desc._client.post = AsyncMock(side_effect=RuntimeError("timeout"))

            with (
                patch.object(desc, "is_available", return_value=True),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_bytes", return_value=b"fake"),
            ):
                result = await desc.describe_image_async("/tmp/test.png")

        assert result == ""

    async def test_strips_response(self):
        """Whitespace around the VLM response is stripped."""
        with patch("ingestion.multimodal.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.ollama_keep_alive = "30m"
            desc = ImageDescriptor()

            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "  description with spaces  \n"}
            desc._client = MagicMock()
            desc._client.post = AsyncMock(return_value=mock_response)

            with (
                patch.object(desc, "is_available", return_value=True),
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_bytes", return_value=b"fake"),
            ):
                result = await desc.describe_image_async("/tmp/test.png")

        assert result == "description with spaces"
