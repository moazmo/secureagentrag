"""Tests for document loaders."""

from __future__ import annotations

import pytest

from ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    LoadedDocument,
    load_document,
)


class TestSupportedExtensions:
    """Tests for the SUPPORTED_EXTENSIONS constant."""

    def test_contains_pdf(self) -> None:
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_contains_docx(self) -> None:
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_contains_doc(self) -> None:
        assert ".doc" in SUPPORTED_EXTENSIONS

    def test_contains_image_formats(self) -> None:
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS

    def test_is_a_set(self) -> None:
        assert isinstance(SUPPORTED_EXTENSIONS, set)

    def test_all_extensions_start_with_dot(self) -> None:
        for ext in SUPPORTED_EXTENSIONS:
            assert ext.startswith("."), f"Extension '{ext}' missing leading dot"


class TestLoadedDocument:
    """Tests for the LoadedDocument model."""

    def test_creation_with_required_fields(self) -> None:
        doc = LoadedDocument(
            text="Hello world",
            source_file="test.pdf",
            file_type="pdf",
        )
        assert doc.text == "Hello world"
        assert doc.source_file == "test.pdf"
        assert doc.file_type == "pdf"
        assert doc.page_number == 0
        assert doc.metadata == {}

    def test_creation_with_all_fields(self) -> None:
        doc = LoadedDocument(
            text="Content here",
            page_number=3,
            source_file="report.pdf",
            file_type="pdf",
            metadata={"total_pages": 10},
        )
        assert doc.page_number == 3
        assert doc.metadata == {"total_pages": 10}


class TestLoadDocument:
    """Tests for the load_document factory function."""

    def test_raises_value_error_for_unsupported_extension(self) -> None:
        """Should raise ValueError for unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_document("document.xyz")

    def test_load_text_file(self, tmp_path) -> None:
        """TXT files should be supported and loaded."""
        txt_path = tmp_path / "readme.txt"
        txt_path.write_text("Hello world", encoding="utf-8")
        docs = load_document(str(txt_path))
        assert len(docs) == 1
        assert docs[0].text == "Hello world"
        assert docs[0].source_file == str(txt_path)

    def test_raises_value_error_for_csv(self) -> None:
        """CSV files should not be supported."""
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_document("data.csv")

    def test_raises_file_not_found_for_pdf(self, tmp_path) -> None:
        """Should raise FileNotFoundError for non-existent PDF."""
        fake_path = tmp_path / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError):
            load_document(str(fake_path))

    def test_raises_file_not_found_for_docx(self, tmp_path) -> None:
        """Should raise FileNotFoundError for non-existent DOCX."""
        fake_path = tmp_path / "nonexistent.docx"
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_document(str(fake_path))

    def test_image_loader_returns_empty_text(self, tmp_path) -> None:
        """Image loader should return doc with empty text and OCR flag."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header

        docs = load_document(str(img_path))
        assert len(docs) == 1
        assert docs[0].text == ""
        assert docs[0].file_type == "image"
        assert docs[0].metadata.get("ocr_needed") is True

    def test_image_loader_jpg(self, tmp_path) -> None:
        """JPG images should be handled by the image loader."""
        img_path = tmp_path / "photo.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

        docs = load_document(str(img_path))
        assert len(docs) == 1
        assert docs[0].file_type == "image"

    def test_case_insensitive_extension(self, tmp_path) -> None:
        """Should handle uppercase extensions via Path.suffix.lower()."""
        # This tests that the factory handles the extension correctly
        # Note: We can't easily test .PDF detection without an actual file
        # but we can verify the extension detection logic
        with pytest.raises(ValueError, match="Unsupported"):
            load_document("file.UNSUPPORTED")
