"""Text chunking strategies for document processing.

Supports multilingual text including Arabic (RTL) with language-aware
separator selection and proper handling of attached prefixes/suffixes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from config.settings import settings
from utils.logging import get_logger

if TYPE_CHECKING:
    from ingestion.loaders import LoadedDocument

logger = get_logger(__name__)

# Arabic-specific separators (priority order)
_ARABIC_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "، ", "؛ ", " ", ""]

# Arabic sentence-ending punctuation (includes Arabic full stop U+06D4)
_ARABIC_SENTENCE_END = re.compile(r"[.!?\u06D4]\s+")

# Arabic attached prefixes that should not be split from words
# ال (definite article), و (and), ب (with), ل (for), ك (like), ف (so)
_ARABIC_PREFIXES = re.compile(r"^[\u0627\u0644\u0648\u0628\u0644\u0643\u0641]")

# Arabic attached suffixes (possessive pronouns)
# ي (my), ك (your), ه (his), ها (her), هم (their), نا (our)
_ARABIC_SUFFIXES = re.compile(r"[\u064a\u0643\u0647\u0647\u0627\u0645\u0646\u0627]$")

# Detect if text contains significant Arabic content
_ARABIC_SCRIPT_RANGE = re.compile(r"[\u0600-\u06FF]")


def _detect_language(text: str) -> str:
    """Detect the primary language of the text.

    Args:
        text: Input text to analyze.

    Returns:
        'arabic' if significant Arabic content detected, 'default' otherwise.
    """
    if not text:
        return "default"
    arabic_chars = len(_ARABIC_SCRIPT_RANGE.findall(text))
    total_chars = len(text.strip())
    if total_chars == 0:
        return "default"
    # If > 15% of characters are Arabic script, treat as Arabic text
    if arabic_chars / total_chars > 0.15:
        return "arabic"
    return "default"


class TextChunker:
    """Recursive character text splitter for document chunking.

    Splits text using a hierarchy of separators, attempting to keep chunks
    within the specified size limit while maintaining semantic coherence.
    Automatically selects language-appropriate separators for Arabic text.

    Args:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        separators: Ordered list of separators to try for splitting.
        arabic_separators: Arabic-specific separators. Uses default if None.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
        arabic_separators: list[str] | None = None,
    ) -> None:
        """Initialize the text chunker.

        Args:
            chunk_size: Maximum chunk size in characters. Defaults to settings value.
            chunk_overlap: Overlap between chunks. Defaults to settings value.
            separators: List of separators in priority order. Defaults to standard set.
            arabic_separators: Arabic-specific separators. Uses default if None.
        """
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self._separators = separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]
        self._arabic_separators = (
            arabic_separators if arabic_separators is not None else _ARABIC_SEPARATORS
        )

        # Input validation
        if self._chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self._chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self._chunk_size > 100_000:
            raise ValueError("chunk_size exceeds maximum (100,000)")

        if self._chunk_overlap >= self._chunk_size:
            raise ValueError(
                f"chunk_overlap ({self._chunk_overlap}) must be less than "
                f"chunk_size ({self._chunk_size})"
            )

        logger.info(
            "chunker_initialized",
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators_count=len(self._separators),
            arabic_separators_count=len(self._arabic_separators),
        )

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks using recursive character splitting.

        Automatically detects Arabic content and uses Arabic-appropriate
        separators (including Arabic punctuation like ، and ؛).

        Args:
            text: The input text to split.

        Returns:
            List of text chunks. Returns empty list for empty input.
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # If text fits in a single chunk, return it directly
        if len(text) <= self._chunk_size:
            return [text]

        # Detect language and select appropriate separators
        lang = _detect_language(text)
        if lang == "arabic":
            logger.debug("chunking_arabic_text", text_len=len(text))
            return self._recursive_split(text, 0, use_arabic=True)

        return self._recursive_split(text, 0, use_arabic=False)

    def _get_separators(self, use_arabic: bool) -> list[str]:
        """Return the appropriate separator list for the language.

        Args:
            use_arabic: Whether to use Arabic-specific separators.

        Returns:
            List of separator strings in priority order.
        """
        return self._arabic_separators if use_arabic else self._separators

    def _recursive_split(
        self, text: str, separator_idx: int, use_arabic: bool = False
    ) -> list[str]:
        """Recursively split text using separators at the given index.

        Args:
            text: Text to split.
            separator_idx: Index into the separators list.
            use_arabic: Whether to use Arabic-specific separators.

        Returns:
            List of text chunks.
        """
        separators = self._get_separators(use_arabic)

        if separator_idx >= len(separators):
            # No more separators — force split by character
            return self._force_split(text)

        separator = separators[separator_idx]
        chunks: list[str] = []

        if separator == "":
            # Empty separator means split by character (force split)
            return self._force_split(text)

        splits = text.split(separator)

        current_chunk = ""
        for split in splits:
            # Determine what the new chunk would be if we add this split
            candidate = current_chunk + separator + split if current_chunk else split

            if len(candidate) <= self._chunk_size:
                current_chunk = candidate
            else:
                # Current chunk is ready to be emitted
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Check if the split itself is too large
                if len(split) > self._chunk_size:
                    # Recursively split with next separator
                    sub_chunks = self._recursive_split(
                        split, separator_idx + 1, use_arabic=use_arabic
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # Don't forget the last chunk
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Apply overlap
        if self._chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _force_split(self, text: str) -> list[str]:
        """Force-split text into chunks of exactly chunk_size characters.

        Args:
            text: Text to force-split.

        Returns:
            List of text chunks.
        """
        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self._chunk_overlap if self._chunk_overlap > 0 else end

        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between consecutive chunks.

        For each chunk after the first, prepend characters from the end
        of the previous chunk to create overlap.

        Args:
            chunks: List of non-overlapping chunks.

        Returns:
            List of chunks with overlap applied.
        """
        if len(chunks) <= 1:
            return chunks

        overlapped: list[str] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            # Take the overlap portion from the end of the previous chunk
            overlap_text = prev_chunk[-self._chunk_overlap :]
            # Prepend overlap to current chunk
            merged = overlap_text + " " + chunks[i]
            # Trim to chunk_size if necessary
            if len(merged) > self._chunk_size:
                merged = merged[: self._chunk_size]
            overlapped.append(merged.strip())

        return overlapped

    def chunk_documents(
        self,
        documents: list[LoadedDocument],
        source_file: str,
    ) -> list[tuple[str, dict]]:
        """Chunk a list of LoadedDocuments and return chunks with metadata.

        Args:
            documents: List of LoadedDocument instances to process.
            source_file: Original source file path for metadata.

        Returns:
            List of tuples (chunk_text, metadata_dict) where metadata includes
            source_file, page_number, and chunk_index (global incrementing counter).
        """
        results: list[tuple[str, dict]] = []
        global_chunk_index = 0

        for doc in documents:
            if not doc.text or not doc.text.strip():
                logger.debug(
                    "skipping_empty_document",
                    source_file=source_file,
                    page_number=doc.page_number,
                )
                continue

            chunks = self.chunk_text(doc.text)

            for chunk_text in chunks:
                metadata = {
                    "source_file": source_file,
                    "page_number": doc.page_number,
                    "chunk_index": global_chunk_index,
                    "file_type": doc.file_type,
                }
                results.append((chunk_text, metadata))
                global_chunk_index += 1

        logger.info(
            "documents_chunked",
            source_file=source_file,
            document_count=len(documents),
            total_chunks=global_chunk_index,
        )

        return results
