#!/usr/bin/env python3
"""Validate kv-cache-tester trace JSON files.

Stdlib-only validator for the compact trace schema consumed by
`trace_replay_tester.py` / `normalize_trace()` in Cam's kv-cache-tester.
Supports validating a single JSON file or recursively walking a directory of
trace files. When ``--pressure-manifest`` is provided, the validator also
cross-checks that the curated KV-pressure subset points at real trace files and
that its claimed ISL numbers match the on-disk traces exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

VALID_HASH_ID_SCOPES = {"local", "global"}
MANIFEST_FILENAMES = {"manifest.json"}
CHECK = "✓"
CROSS = "✗"
WARN = "!"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)


def _add_issue(bucket: list[str], message: str, max_issues: int) -> None:
    if len(bucket) < max_issues:
        bucket.append(message)


def _validate_string_list(value: Any, field_name: str, errors: list[str], max_issues: int) -> list[str] | None:
    if not isinstance(value, list) or not value:
        _add_issue(errors, f"{field_name} must be a non-empty list[str]", max_issues)
        return None
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            _add_issue(errors, f"{field_name}[{idx}] must be str, got {type(item).__name__}", max_issues)
    return value if len(errors) < max_issues else None


def _validate_flat_hash_ids(
    hash_ids: list[Any],
    *,
    input_tokens: int,
    block_size: int,
    scope: str | None,
    errors: list[str],
    warnings: list[str],
    max_issues: int,
) -> None:
    expected_len = math.ceil(input_tokens / block_size) if input_tokens > 0 else 0
    if len(hash_ids) != expected_len:
        _add_issue(
            errors,
            f"hash_ids length = {len(hash_ids)}, expected ceil(in={input_tokens} / block_size={block_size}) = {expected_len}",
            max_issues,
        )

    if scope is None:
        _add_issue(
            warnings,
            "hash_id_scope missing; cannot strictly validate flat hash_ids semantics",
            max_issues,
        )

    for idx, value in enumerate(hash_ids):
        if not _is_int(value):
            _add_issue(errors, f"hash_ids[{idx}] must be int, got {type(value).__name__}", max_issues)
            continue
        if value <= 0:
            _add_issue(errors, f"hash_ids[{idx}] = {value}, expected positive int", max_issues)
        if scope == "local":
            expected = idx + 1
            if value != expected:
                _add_issue(
                    errors,
                    f"hash_ids[{idx}] = {value}, expected {expected} (prefix must extend by 1)",
                    max_issues,
                )


def _validate_nested_hash_ids(
    hash_ids: list[Any],
    *,
    scope: str | None,
    errors: list[str],
    warnings: list[str],
    max_issues: int,
) -> None:
    if scope is None:
        _add_issue(
            warnings,
            "hash_id_scope missing; cannot strictly validate nested hash_ids semantics",
            max_issues,
        )
    for outer_idx, group in enumerate(hash_ids):
        if not isinstance(group, list):
            _add_issue(errors, f"hash_ids[{outer_idx}] must be list[int], got {type(group).__name__}", max_issues)
            continue
        for inner_idx, value in enumerate(group):
            if not _is_int(value):
                _add_issue(
                    errors,
                    f"hash_ids[{outer_idx}][{inner_idx}] must be int, got {type(value).__name__}",
                    max_issues,
                )
                continue
            if value <= 0:
                _add_issue(errors, f"hash_ids[{outer_idx}][{inner_idx}] = {value}, expected positive int", max_issues)
            if scope == "local":
                expected = inner_idx + 1
                if value != expected:
                    _add_issue(
                        errors,
                        f"hash_ids[{outer_idx}][{inner_idx}] = {value}, expected {expected} (prefix must extend by 1)",
                        max_issues,
                    )


def _validate_request(
    req: Any,
    *,
    request_idx: int,
    block_size: int,
    scope: str | None,
    errors: list[str],
    warnings: list[str],
    max_issues: int,
) -> None:
    prefix = f"requests[{request_idx}]"
    if not isinstance(req, dict):
        _add_issue(errors, f"{prefix} must be object, got {type(req).__name__}", max_issues)
        return

    req_type = req.get("type")
    if not isinstance(req_type, str):
        _add_issue(errors, f"{prefix}.type must be str", max_issues)

    if req_type == "subagent":
        return

    t_value = req.get("t")
    if not _is_number(t_value):
        _add_issue(errors, f"{prefix}.t must be float >= 0", max_issues)
    elif float(t_value) < 0:
        _add_issue(errors, f"{prefix}.t = {t_value}, expected >= 0", max_issues)

    input_tokens = req.get("in")
    if not _is_int(input_tokens):
        _add_issue(errors, f"{prefix}.in must be int >= 0", max_issues)
        input_tokens = 0
    elif input_tokens < 0:
        _add_issue(errors, f"{prefix}.in = {input_tokens}, expected >= 0", max_issues)

    output_tokens = req.get("out")
    if not _is_int(output_tokens):
        _add_issue(errors, f"{prefix}.out must be int >= 0", max_issues)
    elif output_tokens < 0:
        _add_issue(errors, f"{prefix}.out = {output_tokens}, expected >= 0", max_issues)

    hash_ids = req.get("hash_ids")
    if not isinstance(hash_ids, list):
        _add_issue(errors, f"{prefix}.hash_ids must be list[int] or list[list[int]]", max_issues)
    else:
        is_nested = bool(hash_ids) and all(isinstance(item, list) for item in hash_ids)
        is_flat = not hash_ids or all(not isinstance(item, list) for item in hash_ids)
        if is_nested:
            _validate_nested_hash_ids(
                hash_ids,
                scope=scope,
                errors=errors,
                warnings=warnings,
                max_issues=max_issues,
            )
        elif is_flat:
            _validate_flat_hash_ids(
                hash_ids,
                input_tokens=input_tokens,
                block_size=block_size,
                scope=scope,
                errors=errors,
                warnings=warnings,
                max_issues=max_issues,
            )
        else:
            _add_issue(errors, f"{prefix}.hash_ids must not mix flat and nested entries", max_issues)

    for field_name in ("model", "stop"):
        if field_name in req and not isinstance(req[field_name], str):
            _add_issue(errors, f"{prefix}.{field_name} must be str", max_issues)

    for field_name in ("input_types", "output_types"):
        if field_name in req:
            value = req[field_name]
            if not isinstance(value, list):
                _add_issue(errors, f"{prefix}.{field_name} must be list[str]", max_issues)
                continue
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    _add_issue(errors, f"{prefix}.{field_name}[{idx}] must be str", max_issues)

    for field_name in ("api_time", "think_time"):
        if field_name in req:
            value = req[field_name]
            if not _is_number(value):
                _add_issue(errors, f"{prefix}.{field_name} must be float", max_issues)


def validate_trace(trace: Any, *, max_issues: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(trace, dict):
        return [f"top-level JSON must be object, got {type(trace).__name__}"], warnings

    trace_id = trace.get("id")
    if not isinstance(trace_id, str):
        _add_issue(errors, "id must be str", max_issues)

    _validate_string_list(trace.get("models"), "models", errors, max_issues)

    block_size = trace.get("block_size")
    if not _is_int(block_size):
        _add_issue(errors, "block_size must be int > 0", max_issues)
        block_size = 1
    elif block_size <= 0:
        _add_issue(errors, f"block_size = {block_size}, expected > 0", max_issues)

    requests = trace.get("requests")
    if not isinstance(requests, list) or not requests:
        _add_issue(errors, "requests must be a non-empty list", max_issues)
        requests = []

    scope = trace.get("hash_id_scope")
    if scope is not None and scope not in VALID_HASH_ID_SCOPES:
        _add_issue(
            errors,
            f"hash_id_scope = {scope!r}, expected one of {sorted(VALID_HASH_ID_SCOPES)}",
            max_issues,
        )
        scope = None

    for field_name in ("tool_tokens", "system_tokens"):
        if field_name in trace:
            value = trace[field_name]
            if not _is_int(value):
                _add_issue(errors, f"{field_name} must be int >= 0", max_issues)
            elif value < 0:
                _add_issue(errors, f"{field_name} = {value}, expected >= 0", max_issues)

    for idx, req in enumerate(requests):
        if len(errors) >= max_issues and len(warnings) >= max_issues:
            break
        _validate_request(
            req,
            request_idx=idx,
            block_size=block_size,
            scope=scope,
            errors=errors,
            warnings=warnings,
            max_issues=max_issues,
        )

    return errors, warnings


def compute_trace_isl_stats(trace: dict[str, Any]) -> dict[str, int]:
    requests = [
        req
        for req in trace.get("requests", [])
        if isinstance(req, dict) and req.get("type") != "subagent"
    ]
    in_tokens = [int(req.get("in", 0)) for req in requests if _is_int(req.get("in"))]
    return {
        "cumulative_isl_tokens": sum(in_tokens),
        "max_concurrent_session_isl_tokens": max(in_tokens) if in_tokens else 0,
    }


def validate_pressure_manifest(
    manifest: Any,
    *,
    trace_root: Path,
    max_issues: int,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(manifest, dict):
        return [f"pressure manifest must be a JSON object, got {type(manifest).__name__}"], warnings

    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        _add_issue(errors, "entries must be a non-empty list", max_issues)
        return errors, warnings

    resolved_root = trace_root.resolve()
    seen_files: set[str] = set()

    for idx, entry in enumerate(entries):
        prefix = f"entries[{idx}]"
        if not isinstance(entry, dict):
            _add_issue(errors, f"{prefix} must be object, got {type(entry).__name__}", max_issues)
            continue

        file_value = entry.get("file")
        rationale = entry.get("rationale")
        cumulative = entry.get("cumulative_isl_tokens")
        maximum = entry.get("max_concurrent_session_isl_tokens")

        if not isinstance(file_value, str) or not file_value:
            _add_issue(errors, f"{prefix}.file must be non-empty str", max_issues)
            continue
        if file_value in seen_files:
            _add_issue(warnings, f"{prefix}.file duplicates an earlier entry: {file_value}", max_issues)
        seen_files.add(file_value)

        if not isinstance(rationale, str) or not rationale.strip():
            _add_issue(errors, f"{prefix}.rationale must be non-empty str", max_issues)
        if not _is_int(cumulative) or cumulative < 0:
            _add_issue(errors, f"{prefix}.cumulative_isl_tokens must be int >= 0", max_issues)
        if not _is_int(maximum) or maximum < 0:
            _add_issue(errors, f"{prefix}.max_concurrent_session_isl_tokens must be int >= 0", max_issues)
        if len(errors) >= max_issues:
            continue

        candidate = (trace_root / file_value).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError:
            _add_issue(errors, f"{prefix}.file escapes trace root: {file_value}", max_issues)
            continue
        if not candidate.is_file():
            _add_issue(errors, f"{prefix}.file does not exist under {trace_root}: {file_value}", max_issues)
            continue

        try:
            trace = json.loads(candidate.read_text())
        except Exception as exc:  # pragma: no cover - defensive path
            _add_issue(errors, f"{prefix}.file is not valid JSON: {exc}", max_issues)
            continue

        actual = compute_trace_isl_stats(trace)
        if cumulative != actual["cumulative_isl_tokens"]:
            _add_issue(
                errors,
                f"{prefix}.cumulative_isl_tokens = {cumulative}, expected {actual['cumulative_isl_tokens']} from {file_value}",
                max_issues,
            )
        if maximum != actual["max_concurrent_session_isl_tokens"]:
            _add_issue(
                errors,
                f"{prefix}.max_concurrent_session_isl_tokens = {maximum}, expected {actual['max_concurrent_session_isl_tokens']} from {file_value}",
                max_issues,
            )

    return errors, warnings


def iter_trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for candidate in sorted(path.rglob("*.json")):
            if candidate.name in MANIFEST_FILENAMES:
                continue
            if candidate.is_file():
                files.append(candidate)
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate_kvcache_tester_trace.py",
        description="Validate kv-cache-tester trace files or directories.",
    )
    parser.add_argument("path", metavar="PATH", help="file or directory (recursive glob *.json when directory)")
    parser.add_argument("--quiet", action="store_true", help="only print final summary")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors (e.g. hash_ids scope missing)",
    )
    parser.add_argument(
        "--max-errors-per-file",
        type=int,
        default=5,
        help="maximum errors reported per file (default: 5)",
    )
    parser.add_argument(
        "--pressure-manifest",
        type=Path,
        default=None,
        help="optional manifest.json describing curated high-pressure trace entries to cross-check",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.max_errors_per_file <= 0:
        print("--max-errors-per-file must be > 0", file=sys.stderr)
        return 2

    path = Path(args.path)
    try:
        files = iter_trace_files(path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not files:
        print("No trace JSON files found", file=sys.stderr)
        return 2

    valid_count = 0
    failed_count = 0
    manifest_clean = False

    for file_path in files:
        try:
            trace = json.loads(file_path.read_text())
        except Exception as exc:
            errors = [f"invalid JSON: {exc}"]
            warnings: list[str] = []
        else:
            errors, warnings = validate_trace(trace, max_issues=args.max_errors_per_file)

        effective_errors = list(errors)
        if args.strict:
            effective_errors.extend(warnings)

        if effective_errors:
            failed_count += 1
            if not args.quiet:
                print(f"{CROSS} {file_path}")
                for issue in effective_errors[: args.max_errors_per_file]:
                    print(f"    {issue}")
        else:
            valid_count += 1
            if warnings and not args.quiet:
                print(f"{WARN} {file_path}")
                for warning in warnings[: args.max_errors_per_file]:
                    print(f"    {warning}")

    if args.pressure_manifest is not None:
        try:
            manifest_payload = json.loads(args.pressure_manifest.read_text())
        except FileNotFoundError:
            print(f"Path not found: {args.pressure_manifest}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"invalid pressure manifest JSON: {exc}", file=sys.stderr)
            return 2

        trace_root = path if path.is_dir() else path.parent
        manifest_errors, manifest_warnings = validate_pressure_manifest(
            manifest_payload,
            trace_root=trace_root,
            max_issues=args.max_errors_per_file,
        )
        effective_manifest_errors = list(manifest_errors)
        if args.strict:
            effective_manifest_errors.extend(manifest_warnings)
        if effective_manifest_errors:
            failed_count += 1
            if not args.quiet:
                print(f"{CROSS} {args.pressure_manifest}")
                for issue in effective_manifest_errors[: args.max_errors_per_file]:
                    print(f"    {issue}")
        else:
            manifest_clean = True
            if manifest_warnings and not args.quiet:
                print(f"{WARN} {args.pressure_manifest}")
                for warning in manifest_warnings[: args.max_errors_per_file]:
                    print(f"    {warning}")

    if failed_count == 0:
        if manifest_clean:
            print(f"{CHECK} {valid_count} files valid | pressure manifest clean | 0 failed")
        else:
            print(f"{CHECK} {valid_count} files valid | 0 failed")
        return 0

    plural = "files" if failed_count != 1 else "file"
    print(f"{failed_count} {plural} failed validation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
