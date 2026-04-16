#!/usr/bin/env python3
"""CLI validator for KV-stress JSONL trace files.

Usage:
    python validate_trace.py traces/smoke_prefix_reuse.jsonl
    python validate_trace.py --strict traces/smoke_offload_cliff.jsonl
    python validate_trace.py traces/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a standalone script from this directory
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from schema import TraceEntry, TraceFile


def validate_jsonl(path: Path, *, strict: bool = False) -> tuple[int, list[str]]:
    """Validate a JSONL trace file.

    Args:
        path: Path to the JSONL file.
        strict: If True, also validate cross-entry consistency
                (unique trace_ids, no orphan parent_turn_ids).

    Returns:
        (valid_count, error_messages)
    """
    errors: list[str] = []
    entries: list[TraceEntry] = []

    if not path.exists():
        return 0, [f"File not found: {path}"]

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON: {e}")
                continue
            try:
                entry = TraceEntry(**data)
                entries.append(entry)
            except Exception as e:
                errors.append(f"Line {line_num}: validation error: {e}")

    if strict and entries:
        try:
            TraceFile(entries=entries)
        except Exception as e:
            errors.append(f"Cross-entry validation: {e}")

    return len(entries), errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate KV-stress JSONL trace files."
    )
    parser.add_argument(
        "trace_files", nargs="+", type=Path, help="Path(s) to JSONL trace file(s)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable cross-entry consistency checks (unique trace_ids, etc)",
    )
    args = parser.parse_args()

    total_valid = 0
    total_errors = 0

    for trace_file in args.trace_files:
        valid_count, errors = validate_jsonl(trace_file, strict=args.strict)
        total_valid += valid_count
        total_errors += len(errors)

        if errors:
            print(f"FAIL: {trace_file} — {len(errors)} error(s)", file=sys.stderr)
            for err in errors:
                print(f"  {err}", file=sys.stderr)
        else:
            print(f"OK: {valid_count} entries validated in {trace_file}")

    if total_errors > 0:
        print(
            f"\n{total_errors} total error(s) across {len(args.trace_files)} file(s)",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print(f"\nAll {total_valid} entries valid across {len(args.trace_files)} file(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
