#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert ISB1 replay bundles into `mooncake_trace` JSONL files.

Produces per-bundle JSONL files compatible with
`aiperf profile --custom-dataset-type mooncake_trace` input format. Works on
ISB1 export bundles shipped under `datasets/isb1/exports/**` and emits one
`<bundle_id>.jsonl` file per input bundle.

This shim is the ONLY glue between our ISB1 bundles and Cam's aiperf /
mooncake replay path. It does not import or execute any benchmark harness and
has no third-party dependencies: standard library only.

Schema compatibility
--------------------

The `mooncake_trace` rows consumed by aiperf are JSON Lines entries with the
following shape:

    {
      "session_id":    "<conversation identifier>",
      "model":         "<canonical model id>",        # optional, extra field (ignored by aiperf)
      "messages":      [
                           {"role": "user", "content": "..."},
                           {"role": "assistant", "content": "..."},
                         ],
      "output_length": 256,
      "delay":         1500                            # optional, MILLISECONDS
    }

The field names above (`messages` and `delay`) are what aiperf's
`MooncakeTrace` pydantic model validates (see
`aiperf/dataset/loader/models.py`). `model` is retained for traceability
and accepted as an extra field by `AIPerfBaseModel(extra="allow")`.

ISB1 exports store turn history in `events[].input_messages` with typed
`content_blocks`. This exporter flattens those blocks into plain text strings:
text blocks are passed through, code blocks are fenced with Markdown triple
backticks and a language tag when present.

Known limitations
-----------------

- ISB1 tool calls are encoded as text inside `tool` role turns rather than as
  OpenAI `tool_calls` arrays. This exporter preserves that exact text as
  `{"role": "tool", "content": "[tool_call: ...]"}` so request-pattern
  fidelity is retained for KV-cache stress testing.
- Messages whose flattened content is empty are dropped. If an event has no
  remaining non-empty messages after flattening, the event is skipped and a
  warning is raised.
- If the same `session_id` appears across multiple bundles in one invocation,
  later bundles are disambiguated by prefixing `session_id` with
  `<bundle_id>::` and a warning is printed.

Usage
-----

Single bundle:

    python tools/isb1_to_mooncake_trace.py \
        --input datasets/isb1/exports/core/chat_8k1k.json \
        --output-dir /tmp/mooncake/

Whole export tree:

    python tools/isb1_to_mooncake_trace.py \
        --input datasets/isb1/exports/ \
        --output-dir /tmp/mooncake/

Subset by glob:

    python tools/isb1_to_mooncake_trace.py \
        --input 'datasets/isb1/exports/core/*.json' \
        --output-dir /tmp/mooncake/

Dry-run validation:

    python tools/isb1_to_mooncake_trace.py \
        --input datasets/isb1/exports/core/code_8k1k.json \
        --output-dir /tmp/mooncake/ \
        --dry-run --verbose
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

SUPPORTED_ADAPTERS = {"inferencex_trace_replay"}
VALID_ROLES = {"user", "assistant", "system", "tool"}
MANIFEST_FILENAMES = {"manifest.json", "manifest_qwen3.5.json"}


class WarningTracker:
    def __init__(self, *, verbose: bool) -> None:
        self.verbose = verbose
        self.count = 0

    def warn(self, message: str) -> None:
        self.count += 1
        if self.verbose:
            print(f"WARN: {message}", file=sys.stderr)


def _looks_like_glob(raw: str) -> bool:
    return any(ch in raw for ch in "*?[")


def _iter_bundle_files(input_spec: str) -> list[Path]:
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
        files: list[Path] = []
        for candidate in sorted(path.rglob("*.json")):
            if candidate.name in MANIFEST_FILENAMES:
                continue
            if "/prefixes/" in candidate.as_posix():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("adapter_id") not in SUPPORTED_ADAPTERS:
                continue
            files.append(candidate.resolve())
        return files
    raise FileNotFoundError(f"input path not found: {input_spec}")


def _safe_int(value: Any, *, field_name: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {field_name} must be int, got {type(value).__name__}")
    return value


def _load_bundle(bundle_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"failed to read ISB1 bundle {bundle_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"top-level bundle must be object in {bundle_path}")

    adapter_id = str(payload.get("adapter_id") or "")
    if adapter_id not in SUPPORTED_ADAPTERS:
        raise ValueError(
            f"unsupported ISB1 adapter {adapter_id!r} in {bundle_path}. "
            f"Expected one of {sorted(SUPPORTED_ADAPTERS)}."
        )

    bundle_id = payload.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError(f"bundle_id missing or invalid in {bundle_path}")

    exports = payload.get("exports")
    if not isinstance(exports, list) or not exports:
        raise ValueError(f"exports must be a non-empty list in {bundle_path}")

    return payload


def _block_language(block: dict[str, Any]) -> str:
    language = block.get("language")
    if isinstance(language, str) and language:
        return language
    metadata = block.get("metadata")
    if isinstance(metadata, dict):
        meta_lang = metadata.get("language")
        if isinstance(meta_lang, str) and meta_lang:
            return meta_lang
    return ""


def _block_placeholder(block: dict[str, Any]) -> str | None:
    token_count = block.get("token_count")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        return None

    block_type = block.get("type")
    label = block_type.upper() if isinstance(block_type, str) and block_type else "BLOCK"
    return f"[{label} token_count={token_count}]"


def _flatten_blocks(blocks: list[Any]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        raw_block_type = block.get("type")
        block_type = raw_block_type if isinstance(raw_block_type, str) and raw_block_type else None
        text = block.get("text")
        text_value = "" if text is None else str(text)
        if block_type in {None, "text"}:
            if text_value:
                parts.append(text_value)
            elif block_type is None:
                placeholder = _block_placeholder(block)
                if placeholder:
                    parts.append(placeholder)
            continue
        if block_type == "code":
            language = _block_language(block)
            fence = f"```{language}\n{text_value}\n```" if language else f"```\n{text_value}\n```"
            parts.append(fence)
            continue
        if text_value:
            parts.append(text_value)
            continue
        placeholder = _block_placeholder(block)
        if placeholder:
            parts.append(placeholder)
    return "\n\n".join(parts)


def _flatten_message(
    message: dict[str, Any],
    *,
    bundle_id: str,
    export_idx: int,
    event_idx: int,
    message_idx: int,
    warnings: WarningTracker,
) -> dict[str, str] | None:
    role = str(message.get("role") or "")
    if role not in VALID_ROLES:
        raise ValueError(
            f"bundle {bundle_id} export[{export_idx}] event[{event_idx}] "
            f"message[{message_idx}] has unsupported role {role!r}"
        )

    if "content" in message and message.get("content") is not None:
        if not isinstance(message.get("content"), str):
            raise ValueError(
                f"bundle {bundle_id} export[{export_idx}] event[{event_idx}] "
                f"message[{message_idx}].content must be str or null"
            )
        content = message["content"]
    else:
        blocks = message.get("content_blocks")
        if blocks is None:
            blocks = []
        if not isinstance(blocks, list):
            raise ValueError(
                f"bundle {bundle_id} export[{export_idx}] event[{event_idx}] "
                f"message[{message_idx}].content_blocks must be list"
            )
        content = _flatten_blocks(blocks)

    if content == "":
        warnings.warn(
            f"bundle {bundle_id} export[{export_idx}] event[{event_idx}] "
            f"message[{message_idx}] flattened to empty content and was dropped"
        )
        return None

    return {"role": role, "content": content}


def _session_id_for_event(
    *,
    bundle_id: str,
    export_cell: dict[str, Any],
    event: dict[str, Any],
    seen_session_owners: dict[str, str],
    warned_collisions: set[tuple[str, str]],
    warnings: WarningTracker,
) -> str:
    base_session_id = event.get("session_id") or export_cell.get("trace_id")
    if not isinstance(base_session_id, str) or not base_session_id:
        raise ValueError(
            f"bundle {bundle_id} export {export_cell.get('trace_id')!r} has no usable session_id or trace_id"
        )

    owner = seen_session_owners.get(base_session_id)
    if owner is None:
        seen_session_owners[base_session_id] = bundle_id
        return base_session_id
    if owner == bundle_id:
        return base_session_id

    collision_key = (base_session_id, bundle_id)
    if collision_key not in warned_collisions:
        warnings.warn(
            f"session_id collision across bundles for {base_session_id!r}: "
            f"first seen in {owner}, prefixing rows emitted from {bundle_id}"
        )
        warned_collisions.add(collision_key)
    return f"{bundle_id}::{base_session_id}"


def _event_delay_ms(
    *,
    bundle_id: str,
    export_idx: int,
    event_idx: int,
    session_id: str,
    arrival_time_offset_ms: int,
    prior_offsets_ms: dict[str, int],
    warnings: WarningTracker,
) -> float:
    """Compute the inter-turn delay in MILLISECONDS for aiperf's `delay` field.

    aiperf's `MooncakeTrace.delay` is specified in milliseconds (see upstream
    loader schema). First event in a session returns 0.0; negative deltas are
    clamped to 0.0 with a warning.
    """
    prior = prior_offsets_ms.get(session_id)
    prior_offsets_ms[session_id] = arrival_time_offset_ms
    if prior is None:
        return 0.0

    delta_ms = arrival_time_offset_ms - prior
    if delta_ms < 0:
        warnings.warn(
            f"bundle {bundle_id} export[{export_idx}] event[{event_idx}] session {session_id!r} "
            f"has negative arrival delta ({delta_ms} ms); clamping delay to 0.0"
        )
        return 0.0
    return float(delta_ms)


def _convert_bundle(
    *,
    bundle_path: Path,
    include_model: bool,
    include_delay: bool,
    seen_session_owners: dict[str, str],
    warned_collisions: set[tuple[str, str]],
    warnings: WarningTracker,
) -> tuple[str, list[dict[str, Any]], int]:
    payload = _load_bundle(bundle_path)
    bundle_id = str(payload["bundle_id"])
    rows: list[dict[str, Any]] = []
    emitted_sessions: set[str] = set()
    prior_offsets_ms: dict[str, int] = {}

    for export_idx, export_cell in enumerate(payload.get("exports") or []):
        if not isinstance(export_cell, dict):
            raise ValueError(f"bundle {bundle_id} export[{export_idx}] must be object")

        canonical_model_id = export_cell.get("canonical_model_id")
        if include_model and (not isinstance(canonical_model_id, str) or not canonical_model_id):
            raise ValueError(
                f"bundle {bundle_id} export[{export_idx}] missing canonical_model_id"
            )

        events = export_cell.get("events")
        if not isinstance(events, list):
            raise ValueError(f"bundle {bundle_id} export[{export_idx}] events must be list")

        for event_idx, event in enumerate(events):
            if not isinstance(event, dict):
                raise ValueError(
                    f"bundle {bundle_id} export[{export_idx}] event[{event_idx}] must be object"
                )

            context = f"bundle {bundle_id} export[{export_idx}] event[{event_idx}]"
            output_length = event.get("target_output_tokens")
            if output_length is None:
                raise ValueError(f"{context}: missing target_output_tokens")
            output_length = _safe_int(
                output_length,
                field_name="target_output_tokens",
                context=context,
            )
            if output_length <= 0:
                raise ValueError(f"{context}: target_output_tokens must be > 0")

            arrival_time_offset_ms = event.get("arrival_time_offset_ms")
            if arrival_time_offset_ms is None:
                raise ValueError(f"{context}: missing arrival_time_offset_ms")
            arrival_time_offset_ms = _safe_int(
                arrival_time_offset_ms,
                field_name="arrival_time_offset_ms",
                context=context,
            )

            input_messages = event.get("input_messages")
            if not isinstance(input_messages, list):
                raise ValueError(f"{context}: input_messages must be list")

            flattened_messages: list[dict[str, str]] = []
            for message_idx, message in enumerate(input_messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{context}: input_messages[{message_idx}] must be object"
                    )
                flattened = _flatten_message(
                    message,
                    bundle_id=bundle_id,
                    export_idx=export_idx,
                    event_idx=event_idx,
                    message_idx=message_idx,
                    warnings=warnings,
                )
                if flattened is not None:
                    flattened_messages.append(flattened)

            if not flattened_messages:
                warnings.warn(f"{context}: skipped event because every message flattened to empty content")
                continue

            session_id = _session_id_for_event(
                bundle_id=bundle_id,
                export_cell=export_cell,
                event=event,
                seen_session_owners=seen_session_owners,
                warned_collisions=warned_collisions,
                warnings=warnings,
            )
            emitted_sessions.add(session_id)

            row: dict[str, Any] = {
                "session_id": session_id,
                "messages": flattened_messages,
                "output_length": output_length,
            }
            if include_model:
                row["model"] = str(canonical_model_id)
            if include_delay:
                row["delay"] = _event_delay_ms(
                    bundle_id=bundle_id,
                    export_idx=export_idx,
                    event_idx=event_idx,
                    session_id=session_id,
                    arrival_time_offset_ms=arrival_time_offset_ms,
                    prior_offsets_ms=prior_offsets_ms,
                    warnings=warnings,
                )
            rows.append(row)

    return bundle_id, rows, len(emitted_sessions)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="isb1_to_mooncake_trace",
        description=(
            "Convert ISB1 replay bundles into mooncake_trace-compatible JSONL files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Single ISB1 bundle JSON, directory, or glob pattern.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for emitted <bundle_id>.jsonl files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate inputs but do not write JSONL files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-bundle progress and warnings.",
    )
    parser.add_argument(
        "--include-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the model field in emitted rows (default: on).",
    )
    parser.add_argument(
        "--include-delay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the delay field (milliseconds) in emitted rows (default: on).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    try:
        bundle_paths = _iter_bundle_files(args.input)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if not bundle_paths:
        print("ERROR: no ISB1 bundles found", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    warnings = WarningTracker(verbose=args.verbose)
    seen_session_owners: dict[str, str] = {}
    warned_collisions: set[tuple[str, str]] = set()

    processed_bundles = 0
    rows_emitted = 0
    rows_written = 0
    sessions_emitted = 0
    errors = 0

    for bundle_path in bundle_paths:
        try:
            bundle_id, rows, bundle_sessions = _convert_bundle(
                bundle_path=bundle_path,
                include_model=args.include_model,
                include_delay=args.include_delay,
                seen_session_owners=seen_session_owners,
                warned_collisions=warned_collisions,
                warnings=warnings,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {bundle_path}: {exc}", file=sys.stderr)
            errors += 1
            continue

        processed_bundles += 1
        rows_emitted += len(rows)
        sessions_emitted += bundle_sessions

        out_path = output_dir / f"{bundle_id}.jsonl"
        if not rows:
            warnings.warn(f"bundle {bundle_id} emitted 0 rows after filtering and was not written")
        elif not args.dry_run:
            _write_jsonl(out_path, rows)
            rows_written += len(rows)

        if args.verbose:
            action = "would write" if args.dry_run else "wrote"
            print(
                f"ok  {bundle_path}: {action} {len(rows)} row(s) "
                f"for {bundle_sessions} session(s) -> {out_path}"
            )

    print(
        f"done: {processed_bundles} bundle(s) processed; "
        f"{rows_written} row(s) written; "
        f"{sessions_emitted} session(s) emitted; "
        f"{warnings.count} warning(s) raised"
    )
    if args.dry_run:
        print(f"note: dry-run enabled; {rows_emitted} row(s) validated and 0 written")

    if rows_emitted == 0:
        print("ERROR: no mooncake rows emitted", file=sys.stderr)
        return 1

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
