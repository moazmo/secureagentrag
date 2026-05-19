"""One-shot migration: rewrite legacy ``[[N]]`` citations to ``[N]`` on disk.

Old synthesizer prompts emitted double-bracket citation markers. The current
prompt emits single brackets and the citation extractor handles both, but
stored conversation threads still render the old form in the UI. This
script walks every JSONL under ``conversations/`` and rewrites the
``content`` of each message in place.

Usage:
    uv run python -m scripts.migrate_legacy_citations
    uv run python -m scripts.migrate_legacy_citations --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_LEGACY_RE = re.compile(r"\[\[(\d+)\]\]")
_ROOT = Path(__file__).resolve().parent.parent


def _migrate_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Return (lines_seen, lines_rewritten) for one JSONL file."""
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines:
        return 0, 0
    seen = 0
    changed = 0
    out: list[str] = []
    for line in lines:
        seen += 1
        if not line.strip():
            out.append(line)
            continue
        try:
            data = json.loads(line)
        except Exception:
            out.append(line)
            continue
        msgs = data.get("messages") or []
        touched = False
        for msg in msgs:
            content = msg.get("content", "")
            new = _LEGACY_RE.sub(r"[\1]", content)
            if new != content:
                msg["content"] = new
                touched = True
        if touched:
            changed += 1
            out.append(json.dumps(data, ensure_ascii=False))
        else:
            out.append(line)
    if changed and not dry_run:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return seen, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conversations-dir",
        default=str(_ROOT / "conversations"),
        help="Conversation root (default: ./conversations)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.conversations_dir)
    if not root.exists():
        print(f"No conversations directory at {root}; nothing to do.")
        return 0

    files = list(root.rglob("*.jsonl"))
    if not files:
        print(f"No JSONL files under {root}.")
        return 0

    total_seen = 0
    total_changed = 0
    files_touched = 0
    for f in files:
        seen, changed = _migrate_file(f, args.dry_run)
        total_seen += seen
        total_changed += changed
        if changed:
            files_touched += 1
            print(
                f"{'DRY' if args.dry_run else 'OK '}  {f.relative_to(root)}  rewrote {changed} line(s)"
            )

    print()
    print(
        f"Scanned {len(files)} file(s) / {total_seen} line(s). "
        f"Rewrote {total_changed} line(s) in {files_touched} file(s)."
        + ("  (dry-run, no writes)" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
