# SPDX-License-Identifier: Apache-2.0
"""Contract tests for ``tools/isb1_to_mooncake_trace.py``.

These tests lock the emitted mooncake JSONL row schema so ISB1 -> mooncake
conversion cannot silently drift from the `aiperf --custom-dataset-type
mooncake_trace` contract or from the assist-mode plan that introduced this
exporter.

The test suite uses stdlib ``unittest`` only and exercises the exporter through
its public CLI entrypoint (`main`) so flag handling, warnings, summary output,
and on-disk artifacts are all covered.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from tools import isb1_to_mooncake_trace as exporter


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _code_block(text: str | None, *, language: str | None = None, metadata_language: str | None = None) -> dict:
    block = {"type": "code", "text": text}
    if language:
        block["language"] = language
    if metadata_language:
        block["metadata"] = {"language": metadata_language}
    return block


def _message(role: str, *, blocks: list[dict] | None = None, content: str | None = None) -> dict:
    payload = {"role": role}
    if content is not None:
        payload["content"] = content
    else:
        payload["content_blocks"] = blocks or []
    return payload


def _event(
    *,
    session_id: str | None = "sess-1",
    offset_ms: int = 0,
    messages: list[dict] | None = None,
    output_tokens: int | None = 16,
) -> dict:
    payload = {
        "arrival_time_offset_ms": offset_ms,
        "input_messages": messages or [_message("user", blocks=[_text_block("hello")])],
    }
    if session_id is not None:
        payload["session_id"] = session_id
    if output_tokens is not None:
        payload["target_output_tokens"] = output_tokens
    return payload


def _export(trace_id: str, *, model: str = "model-a", events: list[dict] | None = None) -> dict:
    return {
        "trace_id": trace_id,
        "canonical_model_id": model,
        "events": events or [],
    }


def _bundle(bundle_id: str, *, exports: list[dict] | None = None) -> dict:
    return {
        "adapter_id": "inferencex_trace_replay",
        "schema_version": "0.1.0",
        "bundle_id": bundle_id,
        "exports": exports or [],
    }


def _load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


class MooncakeExporterTests(unittest.TestCase):
    def _run_main(self, input_spec: str, output_dir: Path, *extra_args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        argv = ["--input", input_spec, "--output-dir", str(output_dir), *extra_args]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = exporter.main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def _write_bundle(self, directory: Path, filename: str, payload: dict) -> Path:
        path = directory / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_single_turn_event_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_single",
                exports=[
                    _export(
                        "trace-single",
                        events=[_event(session_id="sess-single", messages=[_message("user", blocks=[_text_block("hello world")])], output_tokens=42)],
                    )
                ],
            )
            bundle_path = self._write_bundle(root, "single.json", bundle)

            exit_code, stdout, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            self.assertIn("1 bundle(s) processed", stdout)

            rows = _load_jsonl(out_dir / "bundle_single.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["session_id"], "sess-single")
            self.assertEqual(rows[0]["model"], "model-a")
            self.assertEqual(rows[0]["output_length"], 42)
            self.assertEqual(rows[0]["delay"], 0.0)
            self.assertEqual(rows[0]["messages"], [{"role": "user", "content": "hello world"}])

    def test_multi_turn_session_rows_grouped_by_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_grouped",
                exports=[
                    _export(
                        "trace-grouped",
                        events=[
                            _event(session_id="sess-grouped", offset_ms=0, messages=[_message("user", blocks=[_text_block("turn one")])]),
                            _event(session_id="sess-grouped", offset_ms=1500, messages=[_message("assistant", blocks=[_text_block("turn two")])]),
                        ],
                    )
                ],
            )
            bundle_path = self._write_bundle(root, "grouped.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_grouped.jsonl")
            self.assertEqual([row["session_id"] for row in rows], ["sess-grouped", "sess-grouped"])
            self.assertEqual(rows[0]["messages"][0]["content"], "turn one")
            self.assertEqual(rows[1]["messages"][0]["content"], "turn two")

    def test_content_blocks_text_flattening(self) -> None:
        blocks = [_text_block("alpha"), _text_block("beta")]
        self.assertEqual(exporter._flatten_blocks(blocks), "alpha\n\nbeta")

    def test_content_blocks_code_flattening_fences_with_language(self) -> None:
        blocks = [_code_block("print('hi')", language="python")]
        self.assertEqual(exporter._flatten_blocks(blocks), "```python\nprint('hi')\n```")

    def test_mixed_text_and_code_blocks_in_one_message(self) -> None:
        blocks = [_text_block("before"), _code_block("SELECT 1", metadata_language="sql")]
        self.assertEqual(exporter._flatten_blocks(blocks), "before\n\n```sql\nSELECT 1\n```")

    def test_tool_role_message_is_preserved_as_is(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_tool",
                exports=[
                    _export(
                        "trace-tool",
                        events=[
                            _event(
                                session_id="sess-tool",
                                messages=[_message("tool", blocks=[_text_block("[tool_call: ls -R repo/]")])],
                            )
                        ],
                    )
                ],
            )
            bundle_path = self._write_bundle(root, "tool.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_tool.jsonl")
            self.assertEqual(rows[0]["messages"], [{"role": "tool", "content": "[tool_call: ls -R repo/]"}])

    def test_empty_content_blocks_skip_event_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_empty",
                exports=[
                    _export(
                        "trace-empty",
                        events=[
                            _event(session_id="sess-empty", messages=[_message("user", blocks=[])]),
                            _event(session_id="sess-valid", offset_ms=1000, messages=[_message("user", blocks=[_text_block("still here")])]),
                        ],
                    )
                ],
            )
            bundle_path = self._write_bundle(root, "empty.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir, "--verbose")
            self.assertEqual(exit_code, 0, stderr)
            self.assertIn("skipped event because every message flattened to empty content", stderr)
            rows = _load_jsonl(out_dir / "bundle_empty.jsonl")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["session_id"], "sess-valid")

    def test_delay_ms_computation_across_consecutive_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_gap",
                exports=[
                    _export(
                        "trace-gap",
                        events=[
                            _event(session_id="sess-gap", offset_ms=0),
                            _event(session_id="sess-gap", offset_ms=2500),
                        ],
                    )
                ],
            )
            bundle_path = self._write_bundle(root, "gap.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_gap.jsonl")
            self.assertEqual(rows[1]["delay"], 2500.0)

    def test_first_event_in_session_has_zero_delay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle("bundle_first", exports=[_export("trace-first", events=[_event(session_id="sess-first", offset_ms=9000)])])
            bundle_path = self._write_bundle(root, "first.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_first.jsonl")
            self.assertEqual(rows[0]["delay"], 0.0)

    def test_negative_delta_is_clamped_to_zero_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_negative_gap",
                exports=[
                    _export(
                        "trace-negative-gap",
                        events=[
                            _event(session_id="sess-negative", offset_ms=5000),
                            _event(session_id="sess-negative", offset_ms=2000),
                        ],
                    )
                ],
            )
            bundle_path = self._write_bundle(root, "negative.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir, "--verbose")
            self.assertEqual(exit_code, 0, stderr)
            self.assertIn("negative arrival delta", stderr)
            rows = _load_jsonl(out_dir / "bundle_negative_gap.jsonl")
            self.assertEqual(rows[1]["delay"], 0.0)

    def test_no_include_model_flag_strips_model_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle_path = self._write_bundle(root, "model_off.json", _bundle("bundle_no_model", exports=[_export("trace-model-off", events=[_event(session_id="sess-no-model")])]))

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir, "--no-include-model")
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_no_model.jsonl")
            self.assertNotIn("model", rows[0])

    def test_no_include_delay_flag_strips_delay_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle_path = self._write_bundle(root, "gap_off.json", _bundle("bundle_no_gap", exports=[_export("trace-gap-off", events=[_event(session_id="sess-no-gap")])]))

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir, "--no-include-delay")
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_no_gap.jsonl")
            self.assertNotIn("delay", rows[0])

    def test_dry_run_writes_nothing_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle_path = self._write_bundle(root, "dry.json", _bundle("bundle_dry", exports=[_export("trace-dry", events=[_event(session_id="sess-dry")])]))

            exit_code, stdout, stderr = self._run_main(str(bundle_path), out_dir, "--dry-run")
            self.assertEqual(exit_code, 0, stderr)
            self.assertIn("dry-run enabled", stdout)
            self.assertFalse(out_dir.exists())

    def test_missing_target_output_tokens_errors_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bad_bundle = _bundle(
                "bundle_missing_output",
                exports=[_export("trace-missing-output", events=[_event(session_id="sess-missing-output", output_tokens=None)])],
            )
            bundle_path = self._write_bundle(root, "missing_output.json", bad_bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 1)
            self.assertIn("missing target_output_tokens", stderr)

    def test_session_collision_across_bundles_warns_and_disambiguates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle_one = _bundle("bundle_one", exports=[_export("trace-one", events=[_event(session_id="sess-collision")])])
            bundle_two = _bundle("bundle_two", exports=[_export("trace-two", events=[_event(session_id="sess-collision")])])
            self._write_bundle(root, "a_bundle.json", bundle_one)
            self._write_bundle(root, "b_bundle.json", bundle_two)

            exit_code, _, stderr = self._run_main(str(root), out_dir, "--verbose")
            self.assertEqual(exit_code, 0, stderr)
            self.assertIn("session_id collision across bundles", stderr)

            rows_one = _load_jsonl(out_dir / "bundle_one.jsonl")
            rows_two = _load_jsonl(out_dir / "bundle_two.jsonl")
            self.assertEqual(rows_one[0]["session_id"], "sess-collision")
            self.assertEqual(rows_two[0]["session_id"], "bundle_two::sess-collision")

    def test_directory_input_processes_multiple_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            self._write_bundle(root, "dir_a.json", _bundle("bundle_dir_a", exports=[_export("trace-dir-a", events=[_event(session_id="sess-dir-a")])]))
            self._write_bundle(root, "dir_b.json", _bundle("bundle_dir_b", exports=[_export("trace-dir-b", events=[_event(session_id="sess-dir-b")])]))

            exit_code, stdout, stderr = self._run_main(str(root), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            self.assertIn("2 bundle(s) processed", stdout)
            self.assertTrue((out_dir / "bundle_dir_a.jsonl").exists())
            self.assertTrue((out_dir / "bundle_dir_b.jsonl").exists())

    def test_fallback_to_trace_id_when_event_session_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "out"
            bundle = _bundle(
                "bundle_trace_fallback",
                exports=[_export("trace-fallback", events=[_event(session_id=None, messages=[_message("user", content="hi")])])],
            )
            bundle_path = self._write_bundle(root, "trace_fallback.json", bundle)

            exit_code, _, stderr = self._run_main(str(bundle_path), out_dir)
            self.assertEqual(exit_code, 0, stderr)
            rows = _load_jsonl(out_dir / "bundle_trace_fallback.jsonl")
            self.assertEqual(rows[0]["session_id"], "trace-fallback")


if __name__ == "__main__":
    unittest.main()
