"""Tamper-evident audit log chain integrity tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from utils.audit import GENESIS_HASH, AuditLogger


def test_chain_valid_after_appends() -> None:
    """Sequential writes must produce a valid chain whose entries link prev_hash."""
    with tempfile.TemporaryDirectory() as d:
        al = AuditLogger(log_dir=d)
        al.log_query(user_id="u1", query="q1")
        al.log_query(user_id="u1", query="q2")
        al.log_access(user_id="u1", documents_accessed=["d1"])

        result = al.verify_chain()
        assert result["valid"]
        assert result["checked"] == 3
        assert result["last_hash"] != GENESIS_HASH


def test_chain_detects_in_place_edit() -> None:
    """Editing any past entry must break the chain."""
    with tempfile.TemporaryDirectory() as d:
        al = AuditLogger(log_dir=d)
        al.log_query(user_id="u1", query="original")
        al.log_query(user_id="u1", query="next")

        f = next(Path(d).glob("*.jsonl"))
        lines = f.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace("original", "tampered")
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = al.verify_chain()
        assert not result["valid"]
        assert any("hash_mismatch" in r for r in result["broken_at"])


def test_chain_detects_deletion() -> None:
    """Removing an entry must break the chain at the next link."""
    with tempfile.TemporaryDirectory() as d:
        al = AuditLogger(log_dir=d)
        al.log_query(user_id="u1", query="q1")
        al.log_query(user_id="u1", query="q2")
        al.log_query(user_id="u1", query="q3")

        f = next(Path(d).glob("*.jsonl"))
        lines = f.read_text(encoding="utf-8").splitlines()
        # Drop the middle entry
        f.write_text(lines[0] + "\n" + lines[2] + "\n", encoding="utf-8")

        result = al.verify_chain()
        assert not result["valid"]
        assert any("chain_broken" in r for r in result["broken_at"])


def test_chain_bootstraps_across_logger_restarts() -> None:
    """A new AuditLogger instance picks up the prior last_hash and keeps chaining."""
    with tempfile.TemporaryDirectory() as d:
        al1 = AuditLogger(log_dir=d)
        al1.log_query(user_id="u1", query="first")

        al2 = AuditLogger(log_dir=d)
        al2.log_query(user_id="u1", query="second")

        result = al2.verify_chain()
        assert result["valid"]
        assert result["checked"] == 2
