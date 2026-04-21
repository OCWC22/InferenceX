#!/usr/bin/env python3
"""Validate mooncake trace JSONL files.

Stdlib-only validator for the compact row schema consumed by
`aiperf profile --custom-dataset-type mooncake_trace`.
Supports validating a single JSONL file, a directory of JSONL files, or a glob
pattern.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

VALID_ROLES = {"user", "assistant", "system", "tool"}
REQUIRED_FIELDS = {"session_id", "input", "output_length"}
OPTIONAL_SUPERSET_FIELDS = {"model", "pre_gap"}
CHECK = "✓"
CROSS = "✗"
WARN = "!"


def _looks_like_glob(raw: str) -> bool:
    return any(ch in raw for ch in "*?[")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)


def _add_issue(bucket: list[str], message: str, max_issues: int) -> None:
    if len(bucket) < max_issues:
        bucket.append(message)


def _iter_input_files(input_spec: str) -> list[Path]:
    if _looks_like_glob(input_spec):
        matches = [Path(p).resolve() for p in sorted(glob.glob(input_spec, recursive=True))]
        files = [p for p in matches if p.is_file()]
        if files:
            return files
        raise FileNotFoundError(f"no files matched glob: {input_spec}")

    path = Path(input_spec).resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        return [candidate.resolve() for candidate in sorted(path.rglob("*.jsonl")) if candidate.is_file()]
    raise FileNotFoundError(f"input path not found: {input_spec}")


def _validate_message(
    message: Any,
    *,
    line_no: int,
    message_idx: int,
    errors: list[str],
    categories: Counter[str],
    max_issues: int,
) -> None:
    prefix = f"line {line_no} input[{message_idx}]"
    if not isinstance(message, dict):
        _add_issue(errors, f"{prefix} must be object", max_issues)
        categories["message_not_object"] += 1
        return

    role = message.get("role")
    if not isinstance(role, str) or role not in VALID_ROLES:
        _add_issue(
            errors,
            f"{prefix}.role must be one of {sorted(VALID_ROLES)}",
            max_issues,
        )
        categories["invalid_role"] += 1

    content = message.get("content")
    if not isinstance(content, str):
        _add_issue(errors, f"{prefix}.content must be str", max_issues)
        categories["invalid_content"] += 1


def validate_row(
    row: Any,
    *,
    line_no: int,
    allow_superset: bool,
    strict: bool,
    max_issues: int,
) -> tuple[list[str], list[str], Counter[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    categories: Counter[str] = Counter()

    if not isinstance(row, dict):
        categories["row_not_object"] += 1
        return [f"line {line_no}: row must be object"], warnings, categories

    allowed_fields = set(REQUIRED_FIELDS)
    if allow_superset:
        allowed_fields.update(OPTIONAL_SUPERSET_FIELDS)

    unknown_fields = sorted(key for key in row.keys() if key not in allowed_fields)
    if unknown_fields:
        message = f"line {line_no}: unknown field(s): {', '.join(unknown_fields)}"
        if strict:
            categories["unknown_field"] += len(unknown_fields)
            _add_issue(errors, message, max_issues)
        else:
            _add_issue(warnings, message, max_issues)

    session_id = row.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        _add_issue(errors, f"line {line_no}: session_id must be non-empty str", max_issues)
        categories["missing_session_id"] += 1

    input_messages = row.get("input")
    if not isinstance(input_messages, list) or not input_messages:
        _add_issue(errors, f"line {line_no}: input must be a non-empty list", max_issues)
        categories["invalid_input"] += 1
        input_messages = []

    for message_idx, message in enumerate(input_messages):
        if len(errors) >= max_issues:
            break
        _validate_message(
            message,
            line_no=line_no,
            message_idx=message_idx,
            errors=errors,
            categories=categories,
            max_issues=max_issues,
        )

    output_length = row.get("output_length")
    if not _is_int(output_length) or int(output_length) <= 0:
        _add_issue(errors, f"line {line_no}: output_length must be positive int", max_issues)
        categories["invalid_output_length"] += 1

    if "model" in row and not isinstance(row.get("model"), str):
        _add_issue(errors, f"line {line_no}: model must be str", max_issues)
        categories["invalid_model"] += 1

    if "pre_gap" in row:
        pre_gap = row.get("pre_gap")
        if not _is_number(pre_gap):
            _add_issue(errors, f"line {line_no}: pre_gap must be non-negative float", max_issues)
            categories["invalid_pre_gap"] += 1
        elif float(pre_gap) < 0.0:
            _add_issue(errors, f"line {line_no}: pre_gap must be >= 0.0", max_issues)
            categories["invalid_pre_gap"] += 1

    return errors, warnings, categories


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_mooncake_trace.py",
        description="Validate mooncake trace JSONL files or directories.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL file, directory, or glob pattern to validate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject unknown fields instead of warning on them.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    parser.add_argument(
        "--max-errors-per-file",
        type=int,
        default=5,
        help="Maximum issues reported per file (default: 5).",
    )
    parser.add_argument(
        "--allow-superset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the optional mooncake superset fields model and pre_gap (default: on).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.max_errors_per_file <= 0:
        print("--max-errors-per-file must be > 0", file=sys.stderr)
        return 2

    try:
        files = _iter_input_files(args.input)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not files:
        print("No mooncake JSONL files found", file=sys.stderr)
        return 2

    files_checked = 0
    rows_scanned = 0
    failed_files = 0
    error_categories: Counter[str] = Counter()

    for file_path in files:
        files_checked += 1
        file_errors: list[str] = []
        file_warnings: list[str] = []

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            file_errors.append(f"failed to read file: {exc}")
            error_categories["file_read_error"] += 1
            lines = []

        for line_no, raw_line in enumerate(lines, start=1):
            if len(file_errors) >= args.max_errors_per_file:
                break
            rows_scanned += 1
            try:
                row = json.loads(raw_line)
            except Exception as exc:
                _add_issue(file_errors, f"line {line_no}: invalid JSON: {exc}", args.max_errors_per_file)
                error_categories["invalid_json"] += 1
                continue

            errors, warnings, categories = validate_row(
                row,
                line_no=line_no,
                allow_superset=args.allow_superset,
                strict=args.strict,
                max_issues=args.max_errors_per_file,
            )
            error_categories.update(categories)
            for issue in errors:
                _add_issue(file_errors, issue, args.max_errors_per_file)
            for warning in warnings:
                _add_issue(file_warnings, warning, args.max_errors_per_file)

        if file_errors:
            failed_files += 1
            if not args.quiet:
                print(f"{CROSS} {file_path}")
                for issue in file_errors[: args.max_errors_per_file]:
                    print(f"    {issue}")
        elif file_warnings and not args.quiet:
            print(f"{WARN} {file_path}")
            for warning in file_warnings[: args.max_errors_per_file]:
                print(f"    {warning}")
        elif not args.quiet:
            print(f"{CHECK} {file_path}")

    if error_categories:
        category_summary = ", ".join(
            f"{key}={value}" for key, value in sorted(error_categories.items())
        )
    else:
        category_summary = "none"

    print(
        f"summary: {files_checked} file(s) checked; {rows_scanned} row(s) scanned; "
        f"errors by category: {category_summary}"
    )

    return 0 if failed_files == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
