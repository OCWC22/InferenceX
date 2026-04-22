#!/usr/bin/env python3
"""Offline smoke test for aiperf MooncakeTraceDatasetLoader."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_AIPERF_SRC = os.environ.get("AIPERF_SRC") or (
    "/tmp/cam-pr993-full/experimental/multiturn/vllm_benchmark/aiperf/src"
)


class SmokeFailure(RuntimeError):
    """Raised when the offline smoke check fails."""


class PromptGeneratorStub:
    """Minimal prompt-generator surface required by the aiperf loader."""

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(resolved_name="offline-smoke-tokenizer")
        self._decoded_cache: dict[tuple[Any, ...], str] = {}

    def generate(self, *, mean: int, stddev: int = 0, hash_ids: list[int] | None = None) -> str:
        return f"[generated mean={mean} stddev={stddev} hash_ids={len(hash_ids or [])}]"

    def _build_token_sequence(self, input_length: int, hash_ids: list[int] | None, block_size: int) -> list[int]:
        blocks = max(1, input_length // max(block_size, 1))
        return list(range(blocks + len(hash_ids or [])))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline smoke test for MooncakeTraceDatasetLoader.")
    parser.add_argument("--input", required=True, help="Mooncake JSONL file or directory of JSONL files.")
    parser.add_argument(
        "--aiperf-src",
        default=DEFAULT_AIPERF_SRC,
        help=f"Path to aiperf src/ (default: AIPERF_SRC env if set, otherwise {DEFAULT_AIPERF_SRC}).",
    )
    return parser.parse_args(argv)


def iter_jsonl_files(input_spec: str) -> list[Path]:
    path = Path(input_spec).resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(candidate.resolve() for candidate in path.rglob("*.jsonl"))
        if files:
            return files
    raise SmokeFailure(f"No JSONL files found under {input_spec}")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SmokeFailure(f"{path}: line {line_no} is invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise SmokeFailure(f"{path}: line {line_no} must decode to a JSON object")
            rows.append(row)
    if not rows:
        raise SmokeFailure(f"{path}: no rows found")
    return rows


def import_aiperf(aiperf_src: Path):
    if sys.version_info < (3, 10):
        raise SmokeFailure("aiperf import requires Python 3.10+; rerun this script with python3.10 or newer")
    if not aiperf_src.is_dir():
        raise SmokeFailure(f"aiperf src path not found: {aiperf_src}")
    sys.path.insert(0, str(aiperf_src))
    try:
        from aiperf.common.config import EndpointConfig, UserConfig  # type: ignore
        from aiperf.dataset.loader import base_trace_loader  # type: ignore
        from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise SmokeFailure(
            f"failed to import aiperf from {aiperf_src}: {exc}. Use an environment with aiperf dependencies installed."
        ) from exc

    base_trace_loader.parallel_decode = lambda token_sequences, *_a, **_k: [
        f"[decoded tokens={len(seq)}]" for seq in token_sequences
    ]
    return EndpointConfig, UserConfig, MooncakeTraceDatasetLoader


def assert_session_ids(raw_rows: list[dict[str, Any]], dataset: dict[str, list[Any]], conversations: list[Any], path: Path) -> None:
    prefixed = {
        row["session_id"]
        for row in raw_rows
        if isinstance(row.get("session_id"), str) and "::" in row["session_id"]
    }
    if not prefixed:
        return
    dataset_ids = set(dataset.keys())
    conversation_ids = {conversation.session_id for conversation in conversations}
    if missing := sorted(prefixed - dataset_ids):
        raise SmokeFailure(f"{path}: prefixed session IDs missing from dataset: {missing}")
    if missing := sorted(prefixed - conversation_ids):
        raise SmokeFailure(f"{path}: prefixed session IDs missing from conversations: {missing}")


def assert_turns(dataset: dict[str, list[Any]], conversations: list[Any], path: Path) -> tuple[Any, Any]:
    conversation_map = {conversation.session_id: conversation for conversation in conversations}
    ordered_delays: list[Any] = []
    for session_id, traces in dataset.items():
        conversation = conversation_map.get(session_id)
        if conversation is None:
            raise SmokeFailure(f"{path}: missing conversation for session {session_id}")
        if len(conversation.turns) != len(traces):
            raise SmokeFailure(f"{path}: session {session_id} trace/turn count mismatch")
        for trace, turn in zip(traces, conversation.turns, strict=True):
            if getattr(trace, "messages", None) is not None and turn.raw_messages != trace.messages:
                raise SmokeFailure(f"{path}: raw_messages mismatch for session {session_id}")
            if turn.delay != getattr(trace, "delay", None):
                raise SmokeFailure(f"{path}: delay mismatch for session {session_id}")
            ordered_delays.append(turn.delay)
    if not ordered_delays:
        raise SmokeFailure(f"{path}: no turn delays captured")
    return ordered_delays[0], ordered_delays[-1]


def smoke_file(path: Path, EndpointConfig, UserConfig, MooncakeTraceDatasetLoader) -> None:
    raw_rows = load_rows(path)
    loader = MooncakeTraceDatasetLoader(
        filename=str(path),
        user_config=UserConfig(endpoint=EndpointConfig(model_names=["offline-smoke-model"])),
        prompt_generator=PromptGeneratorStub(),
    )
    dataset = loader.load_dataset()
    total_rows = sum(len(traces) for traces in dataset.values())
    total_sessions = len(dataset)
    if total_rows <= 0 or total_sessions <= 0:
        raise SmokeFailure(f"{path}: expected positive row and session counts")
    conversations = loader.convert_to_conversations(dataset)
    if len(conversations) != total_sessions:
        raise SmokeFailure(f"{path}: session count mismatch after conversation conversion")
    assert_session_ids(raw_rows, dataset, conversations, path)
    first_delay, last_delay = assert_turns(dataset, conversations, path)
    print(f"OK {path}: rows={total_rows} sessions={total_sessions} first_delay={first_delay} last_delay={last_delay}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        files = iter_jsonl_files(args.input)
        EndpointConfig, UserConfig, MooncakeTraceDatasetLoader = import_aiperf(Path(args.aiperf_src).resolve())
        for file_path in files:
            smoke_file(file_path, EndpointConfig, UserConfig, MooncakeTraceDatasetLoader)
    except SmokeFailure as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
