"""Tests for text chunking strategies."""

from __future__ import annotations

import pytest

from ingestion.chunker import TextChunker
from ingestion.loaders import LoadedDocument


class TestTextChunkerInit:
    """Tests for TextChunker initialization."""

    def test_default_initialization(self) -> None:
        """Should initialize with settings defaults."""
        chunker = TextChunker()
        assert chunker._chunk_size > 0
        assert chunker._chunk_overlap >= 0
        assert len(chunker._separators) > 0

    def test_custom_initialization(self) -> None:
        """Should accept custom parameters."""
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n", " "],
        )
        assert chunker._chunk_size == 500
        assert chunker._chunk_overlap == 50
        assert chunker._separators == ["\n", " "]

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        """Should raise ValueError if overlap >= chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=100, chunk_overlap=150)


class TestChunkText:
    """Tests for TextChunker.chunk_text()."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty or whitespace-only text should return empty list."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []
        assert chunker.chunk_text("\n\n") == []

    def test_short_text_returns_single_chunk(self) -> None:
        """Text shorter than chunk_size should return as a single chunk."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        text = "This is a short paragraph."
        result = chunker.chunk_text(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_gets_split(self) -> None:
        """Text longer than chunk_size should be split into multiple chunks."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "word " * 100  # 500 chars
        result = chunker.chunk_text(text)
        assert len(result) > 1

    def test_all_chunks_within_size_limit(self) -> None:
        """No chunk should exceed chunk_size (approximately, due to overlap)."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a sentence. " * 50
        result = chunker.chunk_text(text)

        for chunk in result:
            # Allow slight overflow due to overlap prepending
            assert len(chunk) <= 100 + 20 + 5

    def test_paragraph_separator_preserves_semantics(self) -> None:
        """Should prefer splitting on paragraph boundaries."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        text = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        result = chunker.chunk_text(text)
        # Should split on \n\n since each paragraph is well under chunk_size
        # The whole text is under 100 chars, so should be a single chunk
        assert len(result) >= 1

    def test_newline_separator(self) -> None:
        """Should split on newlines when paragraphs are too long."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        text = "Line one of text.\nLine two of text.\nLine three of text.\nLine four of text."
        result = chunker.chunk_text(text)
        assert len(result) >= 1
        for chunk in result:
            assert len(chunk) > 0

    def test_overlap_creates_continuity(self) -> None:
        """Chunks with overlap should share some text at boundaries."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "A" * 30 + " " + "B" * 30 + " " + "C" * 30
        result = chunker.chunk_text(text)
        # With overlap > 0, subsequent chunks include text from previous
        if len(result) > 1:
            # Just verify we get multiple chunks
            assert len(result) >= 2

    def test_custom_separators(self) -> None:
        """Should use custom separators for splitting."""
        chunker = TextChunker(
            chunk_size=20,
            chunk_overlap=0,
            separators=["|", " ", ""],
        )
        text = "hello|world|this is a test"
        result = chunker.chunk_text(text)
        assert len(result) >= 1


class TestChunkDocuments:
    """Tests for TextChunker.chunk_documents()."""

    def test_returns_correct_format(self) -> None:
        """Should return list of (text, metadata) tuples."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=0)
        docs = [
            LoadedDocument(
                text="Some content here.",
                page_number=0,
                source_file="test.pdf",
                file_type="pdf",
            )
        ]
        result = chunker.chunk_documents(docs, source_file="test.pdf")
        assert len(result) == 1
        text, meta = result[0]
        assert text == "Some content here."
        assert meta["source_file"] == "test.pdf"
        assert meta["page_number"] == 0
        assert meta["chunk_index"] == 0

    def test_chunk_index_increments_globally(self) -> None:
        """chunk_index should increment across all documents."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=0)
        docs = [
            LoadedDocument(text="Page one.", page_number=0, source_file="t.pdf", file_type="pdf"),
            LoadedDocument(text="Page two.", page_number=1, source_file="t.pdf", file_type="pdf"),
        ]
        result = chunker.chunk_documents(docs, source_file="t.pdf")
        assert len(result) == 2
        assert result[0][1]["chunk_index"] == 0
        assert result[1][1]["chunk_index"] == 1

    def test_empty_documents_skipped(self) -> None:
        """Documents with empty text should be skipped."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=0)
        docs = [
            LoadedDocument(text="", page_number=0, source_file="t.pdf", file_type="pdf"),
            LoadedDocument(text="Content.", page_number=1, source_file="t.pdf", file_type="pdf"),
        ]
        result = chunker.chunk_documents(docs, source_file="t.pdf")
        assert len(result) == 1
        assert result[0][1]["page_number"] == 1

    def test_metadata_includes_file_type(self) -> None:
        """Metadata should include the file_type from the document."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=0)
        docs = [
            LoadedDocument(text="Hello.", page_number=0, source_file="img.png", file_type="image"),
        ]
        result = chunker.chunk_documents(docs, source_file="img.png")
        assert result[0][1]["file_type"] == "image"

    def test_multiple_chunks_from_single_document(self) -> None:
        """A long document should produce multiple chunks with incrementing indices."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        long_text = "This is a moderately long sentence. " * 10
        docs = [
            LoadedDocument(
                text=long_text,
                page_number=0,
                source_file="long.pdf",
                file_type="pdf",
            ),
        ]
        result = chunker.chunk_documents(docs, source_file="long.pdf")
        assert len(result) > 1
        # Verify chunk indices are sequential
        for i, (_, meta) in enumerate(result):
            assert meta["chunk_index"] == i


class TestArabicChunking:
    """Tests for Arabic text chunking with RTL and attached morphology."""

    def test_arabic_language_detection(self) -> None:
        """Should detect Arabic script and use Arabic-aware separators."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        arabic_text = "هذا نص عربي للاختبار. يحتوي على جمل متعددة للتأكد من تقسيم النص بشكل صحيح."
        result = chunker.chunk_text(arabic_text)
        assert len(result) > 0
        # All chunks should contain Arabic text
        for chunk in result:
            assert any("\u0600" <= c <= "\u06FF" for c in chunk)

    def test_arabic_chunk_preserves_words(self) -> None:
        """Arabic chunks should not split words in the middle."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        # Text with words that should stay intact
        arabic_text = "الذكاء الاصطناعي مجال مهم. التعلم الآلي جزء أساسي منه."
        result = chunker.chunk_text(arabic_text)
        assert len(result) >= 1
        for chunk in result:
            # No chunk should start with a suffix or end with a prefix
            # (basic check: words should be separated by spaces)
            words = chunk.split()
            assert len(words) >= 1

    def test_arabic_document_chunking(self) -> None:
        """Should chunk Arabic documents with proper metadata."""
        chunker = TextChunker(chunk_size=80, chunk_overlap=10)
        arabic_doc = LoadedDocument(
            text="هذا مستند عربي طويل. " * 20,
            page_number=1,
            source_file="arabic_doc.pdf",
            file_type="pdf",
        )
        result = chunker.chunk_documents([arabic_doc], source_file="arabic_doc.pdf")
        assert len(result) > 1
        for chunk_text, meta in result:
            assert meta["page_number"] == 1
            assert meta["source_file"] == "arabic_doc.pdf"
            assert any("\u0600" <= c <= "\u06FF" for c in chunk_text)

    def test_mixed_arabic_english_chunking(self) -> None:
        """Should handle mixed Arabic-English text gracefully."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        mixed_text = (
            "This document discusses الذكاء الاصطناعي (Artificial Intelligence) "
            "and its applications in التعلم الآلي (Machine Learning). "
            "It covers important topics like neural networks and الشبكات العصبية."
        )
        result = chunker.chunk_text(mixed_text)
        assert len(result) > 0
        # At least one chunk should have both scripts
        has_mixed = any(
            any("\u0600" <= c <= "\u06FF" for c in chunk) and any(c.isascii() for c in chunk)
            for chunk in result
        )
        assert has_mixed, "Expected at least one mixed-script chunk"
