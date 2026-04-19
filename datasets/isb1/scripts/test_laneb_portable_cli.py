"""Lane B CLI + helpers tests.

These tests exercise only the things we can check without a GPU or Docker:

  * `_gmi_common.sh` validators and mappers.
  * `isb1_results_db.py` schema/migrations accept the new `cloud` column and
    the full Blackwell GPU surface (b200, b300, gb200, gb300) and round-trip
    a cloud-aware insert.

Anything that requires a live container, `nvidia-smi`, or an export file
lives in the per-run smoke checks documented in `GMI_EXECUTION_PLAN.md`
\u2014 those cannot run on CI hosts without GPUs.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

COMMON_SH = SCRIPT_DIR / "_gmi_common.sh"
DB_SCRIPT = SCRIPT_DIR / "isb1_results_db.py"
PORTABLE_SCRIPT = SCRIPT_DIR / "gmi_portable_benchmark.sh"


def _run_bash(snippet: str, *, check: bool = False) -> subprocess.CompletedProcess:
    """Run a bash snippet with _gmi_common.sh already sourced."""
    wrapped = f'set -Eeuo pipefail; source "{COMMON_SH}"\n{snippet}'
    return subprocess.run(
        ["bash", "-c", wrapped],
        capture_output=True,
        text=True,
        check=check,
    )


# ── _gmi_common.sh validators ────────────────────────────────────────

@pytest.mark.parametrize("gpu", ["h100", "h200", "b200", "b300", "gb200", "gb300"])
def test_validate_gpu_type_accepts_supported_gpus(gpu: str) -> None:
    result = _run_bash(f'validate_gpu_type {gpu}')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("gpu", ["mi300x", "a100", "v100", "", "h100-sxm"])
def test_validate_gpu_type_rejects_others(gpu: str) -> None:
    result = _run_bash(f'validate_gpu_type "{gpu}"')
    assert result.returncode == 1
    assert "validate_gpu_type" in result.stderr


@pytest.mark.parametrize("cloud", ["gmi", "aws"])
def test_validate_cloud_accepts_known_clouds(cloud: str) -> None:
    result = _run_bash(f'validate_cloud {cloud}')
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("cloud", ["azure", "gcp", "oracle", ""])
def test_validate_cloud_rejects_unknown_clouds(cloud: str) -> None:
    result = _run_bash(f'validate_cloud "{cloud}"')
    assert result.returncode == 1


# ── _gmi_common.sh mappers ───────────────────────────────────────────

@pytest.mark.parametrize("gpu,cloud,expected", [
    ("h100",  "gmi", "h100-gmi-baremetal"),
    ("h200",  "aws", "h200-aws-baremetal"),
    ("b200",  "gmi", "b200-gmi-baremetal"),
    ("b300",  "gmi", "b300-gmi-baremetal"),
    ("b300",  "aws", "b300-aws-baremetal"),
    ("gb200", "aws", "gb200-aws-baremetal"),
    ("gb300", "gmi", "gb300-gmi-baremetal"),
    ("gb300", "aws", "gb300-aws-baremetal"),
])
def test_laneb_runner_type_format(gpu: str, cloud: str, expected: str) -> None:
    result = _run_bash(f'laneb_runner_type {gpu} {cloud}')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("gpu,expected", [
    ("h100",  "nvidia:h100_sxm_80gb"),
    ("h200",  "nvidia:h200_sxm_141gb"),
    ("b200",  "nvidia:b200_sxm_180gb"),
    ("b300",  "nvidia:b300_sxm_288gb"),
    ("gb200", "nvidia:gb200_nvl72_192gb"),
    ("gb300", "nvidia:gb300_nvl72_288gb"),
])
def test_laneb_hardware_profile_id_matches_exports(gpu: str, expected: str) -> None:
    """The canonical IDs must match what committed ISB1 exports use."""
    result = _run_bash(f'laneb_hardware_profile_id {gpu}')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("gpu,expected", [
    ("h100",  "0"),
    ("h200",  "0"),
    ("b200",  "0"),
    ("b300",  "0"),
    ("gb200", "1"),
    ("gb300", "1"),
])
def test_laneb_gpu_aarch64_flags_grace_hardware(gpu: str, expected: str) -> None:
    result = _run_bash(f'laneb_gpu_aarch64 {gpu}')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


# ── image resolution precedence ──────────────────────────────────────

def test_laneb_default_image_returns_known_defaults() -> None:
    result = _run_bash('laneb_default_image gmi h100 vllm')
    assert result.returncode == 0
    assert result.stdout.strip() == "vllm/vllm-openai:v0.18.0"

    result = _run_bash('laneb_default_image gmi b200 vllm')
    assert result.returncode == 0
    assert result.stdout.strip() == "vllm/vllm-openai:v0.19.0-cu130"

    result = _run_bash('laneb_default_image gmi h200 sglang')
    assert result.returncode == 0
    assert result.stdout.strip() == "lmsysorg/sglang:v0.5.9-cu130"


def test_laneb_default_image_is_empty_for_gb200() -> None:
    """GB200 aarch64: no pinned public tag, default must be empty."""
    for engine in ("vllm", "sglang"):
        result = _run_bash(f'laneb_default_image gmi gb200 {engine}')
        assert result.returncode == 0
        assert result.stdout.strip() == ""


def test_laneb_default_image_is_empty_for_b300() -> None:
    """B300 Blackwell Ultra (sm_103, x86): no pinned public tag yet.
    Default must be empty so callers die with a precise override message
    rather than silently inheriting a b200 image."""
    for engine in ("vllm", "sglang"):
        result = _run_bash(f'laneb_default_image gmi b300 {engine}')
        assert result.returncode == 0
        assert result.stdout.strip() == ""


def test_laneb_default_image_is_empty_for_gb300() -> None:
    """GB300 aarch64 Ultra: no pinned public tag, default must be empty."""
    for engine in ("vllm", "sglang"):
        result = _run_bash(f'laneb_default_image gmi gb300 {engine}')
        assert result.returncode == 0
        assert result.stdout.strip() == ""


def test_laneb_resolve_image_prefers_cloud_specific_override() -> None:
    result = subprocess.run(
        ["bash", "-c",
         f'source "{COMMON_SH}"; laneb_resolve_image gmi gb200 vllm'],
        env={**os.environ, "LANEB_IMAGE_GMI_GB200_VLLM": "custom/vllm-grace:2026-04"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "custom/vllm-grace:2026-04"


def test_laneb_resolve_image_falls_through_to_gpu_engine_override() -> None:
    result = subprocess.run(
        ["bash", "-c",
         f'source "{COMMON_SH}"; laneb_resolve_image aws gb200 sglang'],
        env={**os.environ, "LANEB_IMAGE_GB200_SGLANG": "fallback/sglang-grace:2026-04"},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fallback/sglang-grace:2026-04"


def test_laneb_resolve_image_returns_empty_for_gb200_without_override() -> None:
    """Without an override, GB200 resolves to the empty default; the caller
    (portable script) is what actually dies. The helper itself must succeed
    and return empty so callers can print a precise error."""
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("LANEB_IMAGE_")
    }
    result = subprocess.run(
        ["bash", "-c",
         f'source "{COMMON_SH}"; laneb_resolve_image aws gb200 vllm'],
        env=clean_env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# ── gmi_portable_benchmark.sh CLI acceptance (syntax + arg surface) ──

def test_portable_script_help_advertises_gb200_and_cloud_flag() -> None:
    result = subprocess.run(
        ["bash", str(PORTABLE_SCRIPT), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    # Canonical order locks drift — all four Blackwell SKUs must appear.
    assert "h100|h200|b200|b300|gb200|gb300" in result.stdout
    assert "--cloud <gmi|aws>" in result.stdout
    assert "LANEB_IMAGE_" in result.stdout


def test_portable_script_rejects_unknown_cloud_before_docker_call() -> None:
    result = subprocess.run(
        [
            "bash", str(PORTABLE_SCRIPT),
            "--gpu-type", "h100", "--cloud", "azure",
            "--model", "dsr1", "--engine", "vllm",
            "--context-band", "131k", "--workload", "code",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Unsupported --cloud" in result.stderr or "unsupported cloud" in result.stderr


def test_portable_script_rejects_unknown_gpu_type() -> None:
    result = subprocess.run(
        [
            "bash", str(PORTABLE_SCRIPT),
            "--gpu-type", "mi300x", "--cloud", "aws",
            "--model", "dsr1", "--engine", "vllm",
            "--context-band", "131k", "--workload", "code",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Unsupported --gpu-type" in result.stderr or "unsupported GPU" in result.stderr


# ── isb1_results_db.py: cloud column + gb200 choice ──────────────────

def _load_db_module():
    spec = importlib.util.spec_from_file_location("isb1_results_db", DB_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_isb1_results_db_schema_includes_cloud_column(tmp_path: Path) -> None:
    mod = _load_db_module()
    db_path = tmp_path / "schema.db"
    conn = mod.connect_db(db_path)
    try:
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({mod.TABLE_NAME})"
        )]
    finally:
        conn.close()
    assert "cloud" in cols
    assert "gpu_type" in cols


def test_isb1_results_db_migrates_legacy_db_idempotently(tmp_path: Path) -> None:
    mod = _load_db_module()
    db_path = tmp_path / "legacy.db"

    # Build a schema that predates the cloud column.
    legacy_schema = mod.SCHEMA_SQL.replace("  cloud TEXT,\n", "")
    conn = sqlite3.connect(db_path)
    conn.executescript(legacy_schema)
    conn.commit()
    conn.close()

    # First open must migrate; second open must be a no-op.
    for _ in range(2):
        conn = mod.connect_db(db_path)
        cols = [row[1] for row in conn.execute(
            f"PRAGMA table_info({mod.TABLE_NAME})"
        )]
        conn.close()
        assert "cloud" in cols


def test_isb1_results_db_insert_with_gb200_aws(tmp_path: Path) -> None:
    mod = _load_db_module()
    db_path = tmp_path / "ingest.db"

    # Minimal processed-result JSON that the ingest path can read.
    processed_json = tmp_path / "agg_laneb.json"
    processed_json.write_text('{"selection": {}, "aggregate_metrics": {}}')

    old_argv = sys.argv
    try:
        sys.argv = [
            "isb1_results_db.py",
            "ingest",
            str(processed_json),
            "--db-path", str(db_path),
            "--gpu-type", "gb200",
            "--cloud", "aws",
            "--model", "qwen3.5",
            "--engine", "vllm",
            "--context-band", "1m",
            "--run-id", str(uuid.uuid4()),
        ]
        ns = mod.parse_args()
        mod.insert_run(ns)
    finally:
        sys.argv = old_argv

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            f"SELECT gpu_type, cloud, model, engine, context_band "
            f"FROM {mod.TABLE_NAME}"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("gb200", "aws", "qwen3.5", "vllm", "1m")


def test_isb1_results_db_insert_with_b300_gmi(tmp_path: Path) -> None:
    """b300 (Blackwell Ultra x86) round-trips through ingest at 131k."""
    mod = _load_db_module()
    db_path = tmp_path / "ingest_b300.db"

    processed_json = tmp_path / "agg_laneb.json"
    processed_json.write_text('{"selection": {}, "aggregate_metrics": {}}')

    old_argv = sys.argv
    try:
        sys.argv = [
            "isb1_results_db.py", "ingest", str(processed_json),
            "--db-path", str(db_path),
            "--gpu-type", "b300",
            "--cloud", "gmi",
            "--model", "qwen3.5", "--engine", "vllm", "--context-band", "131k",
            "--run-id", str(uuid.uuid4()),
        ]
        ns = mod.parse_args()
        mod.insert_run(ns)
    finally:
        sys.argv = old_argv

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            f"SELECT gpu_type, cloud, model, engine, context_band "
            f"FROM {mod.TABLE_NAME}"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("b300", "gmi", "qwen3.5", "vllm", "131k")


def test_isb1_results_db_insert_with_gb300_aws(tmp_path: Path) -> None:
    """gb300 (aarch64 Ultra) round-trips through ingest at 1m."""
    mod = _load_db_module()
    db_path = tmp_path / "ingest_gb300.db"

    processed_json = tmp_path / "agg_laneb.json"
    processed_json.write_text('{"selection": {}, "aggregate_metrics": {}}')

    old_argv = sys.argv
    try:
        sys.argv = [
            "isb1_results_db.py", "ingest", str(processed_json),
            "--db-path", str(db_path),
            "--gpu-type", "gb300",
            "--cloud", "aws",
            "--model", "qwen3.5", "--engine", "vllm", "--context-band", "1m",
            "--run-id", str(uuid.uuid4()),
        ]
        ns = mod.parse_args()
        mod.insert_run(ns)
    finally:
        sys.argv = old_argv

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            f"SELECT gpu_type, cloud, model, engine, context_band "
            f"FROM {mod.TABLE_NAME}"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("gb300", "aws", "qwen3.5", "vllm", "1m")


def test_isb1_results_db_cloud_is_groupable(tmp_path: Path) -> None:
    mod = _load_db_module()
    assert "cloud" in mod.GROUPABLE_COLUMNS


def test_isb1_results_db_default_cloud_is_gmi_when_flag_omitted(tmp_path: Path) -> None:
    """Omitting --cloud must land the row as cloud='gmi' for Lane B back-compat."""
    mod = _load_db_module()
    db_path = tmp_path / "default_cloud.db"

    processed_json = tmp_path / "agg.json"
    processed_json.write_text('{"selection": {}, "aggregate_metrics": {}}')

    old_argv = sys.argv
    try:
        sys.argv = [
            "isb1_results_db.py", "ingest", str(processed_json),
            "--db-path", str(db_path),
            "--gpu-type", "h100",
            # No --cloud passed on purpose.
            "--model", "dsr1", "--engine", "vllm", "--context-band", "8k",
        ]
        ns = mod.parse_args()
        mod.insert_run(ns)
    finally:
        sys.argv = old_argv

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            f"SELECT gpu_type, cloud FROM {mod.TABLE_NAME}"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("h100", "gmi")


def test_isb1_results_db_migration_raises_on_non_duplicate_errors(tmp_path: Path) -> None:
    """Regression guard: the migration except-handler must not swallow
    arbitrary OperationalErrors — only 'duplicate column name'. We feed it
    a syntactically invalid migration and assert it surfaces.
    """
    mod = _load_db_module()
    db_path = tmp_path / "broken.db"
    # Ensure the base schema exists first.
    conn = mod.connect_db(db_path); conn.close()

    # Monkey-patch a single bogus migration and confirm it raises.
    original = list(mod._MIGRATIONS)
    try:
        mod._MIGRATIONS.append("ALTER TABLE nonexistent_table ADD COLUMN x TEXT")
        with pytest.raises(sqlite3.OperationalError):
            mod.connect_db(db_path)
    finally:
        mod._MIGRATIONS[:] = original
