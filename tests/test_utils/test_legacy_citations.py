"""Tests for legacy ``[[N]]`` citation normalization on conversation load."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from utils.conversation_store import ConversationMessage, ConversationStore, ConversationThread

if TYPE_CHECKING:
    from pathlib import Path


def _seed(store_dir: Path, *, content: str) -> str:
    """Write a thread JSONL with the given assistant content and return its id."""
    thread_id = "legacy-thread-1"
    # Instantiating the store does not auto-create date subdirs — make one.
    today_dir = store_dir / "2024-01-01"
    today_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "thread_id": thread_id,
        "user_id": "u",
        "org_id": "o",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "messages": [
            {"role": "user", "content": "hello", "timestamp": "", "metadata": {}},
            {"role": "assistant", "content": content, "timestamp": "", "metadata": {}},
        ],
    }
    (today_dir / f"{thread_id}.jsonl").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return thread_id


def test_legacy_double_brackets_rewritten_on_load(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path))
    tid = _seed(tmp_path, content="Look at this [[1]] and also [[2]] sources.")
    loaded = store.load_thread(tid)
    assert loaded is not None
    assert loaded.messages[1].content == "Look at this [1] and also [2] sources."


def test_new_single_brackets_untouched(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path))
    tid = _seed(tmp_path, content="Modern answer [1] with citation [2].")
    loaded = store.load_thread(tid)
    assert loaded is not None
    assert loaded.messages[1].content == "Modern answer [1] with citation [2]."


def test_mixed_brackets_only_legacy_rewritten(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path))
    tid = _seed(tmp_path, content="Old [[1]] mixed with new [2] markers.")
    loaded = store.load_thread(tid)
    assert loaded is not None
    assert loaded.messages[1].content == "Old [1] mixed with new [2] markers."


def test_save_then_load_normalizes_in_memory(tmp_path: Path) -> None:
    store = ConversationStore(base_dir=str(tmp_path))
    thread = ConversationThread(
        thread_id="legacy-roundtrip",
        user_id="u",
        org_id="o",
        messages=[
            ConversationMessage(role="user", content="q"),
            ConversationMessage(role="assistant", content="See [[3]] for details."),
        ],
    )
    store.save_thread(thread)
    loaded = store.load_thread("legacy-roundtrip")
    assert loaded is not None
    assert loaded.messages[1].content == "See [3] for details."
