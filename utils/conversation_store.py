"""Persistent conversation history storage using JSONL files.

Provides durable, file-backed conversation storage that survives server restarts.
Conversations are keyed by thread_id and stored in daily-organized JSONL files
for easy rotation and backup.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Legacy citation markers used by earlier synthesizer prompts. Rendered as
# literal "[[N]]" in the UI, which looked like a bug. The current prompt
# emits `[N]`. We rewrite legacy strings at load time so historic threads
# display the new format without re-running the LLM.
_LEGACY_CITATION_RE = re.compile(r"\[\[(\d+)\]\]")


def _normalize_citations(text: str) -> str:
    """Convert legacy ``[[N]]`` markers to ``[N]`` in stored content."""
    if not text:
        return text
    return _LEGACY_CITATION_RE.sub(r"[\1]", text)


# Default retention: 90 days for conversation threads
DEFAULT_RETENTION_DAYS: int = 90


@dataclass
class ConversationMessage:
    """A single message in a conversation thread.

    Attributes:
        role: Message role — "user", "assistant", "blocked", or "system".
        content: Message text content.
        timestamp: UTC timestamp of the message.
        metadata: Optional metadata (citations, confidence, routing info, etc.).
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationThread:
    """A complete conversation thread with metadata.

    Attributes:
        thread_id: Unique thread identifier.
        user_id: Owner of the conversation.
        org_id: Organization identifier.
        messages: List of messages in chronological order.
        created_at: Thread creation timestamp.
        updated_at: Last activity timestamp.
        metadata: Thread-level metadata (model, mode, etc.).
    """

    thread_id: str
    user_id: str
    org_id: str
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationStore:
    """Persistent conversation storage backed by JSONL files.

    Stores conversations in a directory structure organized by date:
        conversations/YYYY-MM-DD/<thread_id>.jsonl

    Each file contains the full thread as a single JSON line for atomic writes.

    Args:
        base_dir: Base directory for conversation storage.
            Defaults to "conversations" in the project root.
    """

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialize the conversation store.

        Args:
            base_dir: Base directory for conversation files.
        """
        self._base_dir = Path(base_dir or settings.conversation_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("conversation_store_initialized", path=str(self._base_dir))

    def _thread_path(self, thread_id: str) -> Path:
        """Compute the file path for a given thread_id.

        Uses today's date for new files; existing files are found via search.

        Args:
            thread_id: The thread identifier.

        Returns:
            Path to the thread's JSONL file.
        """
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        date_dir = self._base_dir / today
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{thread_id}.jsonl"

    def _find_thread_path(self, thread_id: str) -> Path | None:
        """Search for an existing thread file across all date directories.

        Args:
            thread_id: The thread identifier to search for.

        Returns:
            Path to the existing file, or None if not found.
        """
        for date_dir in self._base_dir.iterdir():
            if date_dir.is_dir():
                candidate = date_dir / f"{thread_id}.jsonl"
                if candidate.exists():
                    return candidate
        return None

    def save_thread(self, thread: ConversationThread) -> bool:
        """Persist a conversation thread to disk.

        Args:
            thread: The conversation thread to save.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Update timestamp
            thread.updated_at = datetime.now(UTC).isoformat()

            # Convert to dict
            data = {
                "thread_id": thread.thread_id,
                "user_id": thread.user_id,
                "org_id": thread.org_id,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata,
                    }
                    for msg in thread.messages
                ],
                "created_at": thread.created_at,
                "updated_at": thread.updated_at,
                "metadata": thread.metadata,
            }

            # Check if thread already exists in a different date directory
            existing_path = self._find_thread_path(thread.thread_id)
            file_path = existing_path or self._thread_path(thread.thread_id)

            # Atomic write: write to temp then rename
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
            temp_path.replace(file_path)

            logger.info(
                "thread_saved",
                thread_id=thread.thread_id,
                message_count=len(thread.messages),
                path=str(file_path),
            )
            return True
        except Exception as exc:
            logger.error("thread_save_failed", thread_id=thread.thread_id, error=str(exc))
            return False

    def load_thread(self, thread_id: str) -> ConversationThread | None:
        """Load a conversation thread from disk.

        Args:
            thread_id: The thread identifier to load.

        Returns:
            The loaded ConversationThread, or None if not found.
        """
        file_path = self._find_thread_path(thread_id)
        if not file_path:
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                line = f.readline().strip()
                if not line:
                    return None
                data = json.loads(line)

            messages = [
                ConversationMessage(
                    role=msg["role"],
                    content=_normalize_citations(msg["content"]),
                    timestamp=msg.get("timestamp", ""),
                    metadata=msg.get("metadata", {}),
                )
                for msg in data.get("messages", [])
            ]

            return ConversationThread(
                thread_id=data["thread_id"],
                user_id=data["user_id"],
                org_id=data["org_id"],
                messages=messages,
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                metadata=data.get("metadata", {}),
            )
        except Exception as exc:
            logger.error("thread_load_failed", thread_id=thread_id, error=str(exc))
            return None

    def list_threads(
        self,
        user_id: str | None = None,
        org_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List conversation threads with optional filtering.

        Args:
            user_id: Filter by user ID.
            org_id: Filter by organization ID.
            limit: Maximum number of threads to return.

        Returns:
            List of thread summary dicts (thread_id, user_id, org_id, message_count, updated_at).
        """
        threads: list[dict[str, Any]] = []

        for date_dir in sorted(self._base_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue
            for file_path in sorted(date_dir.glob("*.jsonl"), reverse=True):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        line = f.readline().strip()
                        if not line:
                            continue
                        data = json.loads(line)

                    if user_id and data.get("user_id") != user_id:
                        continue
                    if org_id and data.get("org_id") != org_id:
                        continue

                    threads.append(
                        {
                            "thread_id": data["thread_id"],
                            "user_id": data.get("user_id", ""),
                            "org_id": data.get("org_id", ""),
                            "message_count": len(data.get("messages", [])),
                            "updated_at": data.get("updated_at", ""),
                            "metadata": data.get("metadata", {}),
                        }
                    )

                    if len(threads) >= limit:
                        break
                except Exception as exc:
                    logger.warning("thread_list_parse_failed", file=str(file_path), error=str(exc))

            if len(threads) >= limit:
                break

        return threads

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread from disk.

        Args:
            thread_id: The thread identifier to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        file_path = self._find_thread_path(thread_id)
        if not file_path:
            return False

        try:
            file_path.unlink()
            logger.info("thread_deleted", thread_id=thread_id, path=str(file_path))
            return True
        except Exception as exc:
            logger.error("thread_delete_failed", thread_id=thread_id, error=str(exc))
            return False

    def append_message(
        self,
        thread_id: str,
        message: ConversationMessage,
        user_id: str = "",
        org_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Append a message to an existing thread, creating it if necessary.

        Args:
            thread_id: The thread identifier.
            message: The message to append.
            user_id: User ID for new threads.
            org_id: Org ID for new threads.
            metadata: Thread-level metadata for new threads.

        Returns:
            True if appended successfully, False otherwise.
        """
        thread = self.load_thread(thread_id)
        if thread is None:
            thread = ConversationThread(
                thread_id=thread_id,
                user_id=user_id,
                org_id=org_id,
                metadata=metadata or {},
            )

        thread.messages.append(message)
        return self.save_thread(thread)

    def cleanup_old_threads(self, retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
        """Delete conversation threads older than the retention period.

        Args:
            retention_days: Number of days to retain conversations. Default 90.

        Returns:
            Number of threads deleted.
        """
        cutoff_timestamp = time.time() - (retention_days * 86400)
        deleted_count = 0

        for date_dir in self._base_dir.iterdir():
            if not date_dir.is_dir():
                continue
            for file_path in date_dir.glob("*.jsonl"):
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime < cutoff_timestamp:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug("old_thread_deleted", path=str(file_path))
                except Exception as exc:
                    logger.warning("thread_cleanup_failed", path=str(file_path), error=str(exc))

            # Remove empty date directories
            try:
                if date_dir.exists() and not any(date_dir.iterdir()):
                    date_dir.rmdir()
            except Exception:
                pass

        logger.info(
            "conversation_cleanup_completed", deleted=deleted_count, retention_days=retention_days
        )
        return deleted_count


# Module-level singleton
conversation_store = ConversationStore()
