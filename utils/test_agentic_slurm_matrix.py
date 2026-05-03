import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_agentic_slurm_matrix.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("run_agentic_slurm_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentic_slurm_dry_run_writes_plan_contract_and_sbatch(tmp_path):
    runner = load_runner()
    rc = runner.main(
        [
            "--dry-run",
            "--results-root",
            str(tmp_path),
            "--hardware",
            "b200",
            "--context-buckets",
            "8k",
            "--concurrency",
            "1,2",
        ]
    )
    assert rc == 0

    plan = json.loads((tmp_path / "matrix_plan.json").read_text())
    assert plan["scenario"] == "agentic-coding"
    assert plan["total_jobs"] == 2
    assert {job["concurrency"] for job in plan["jobs"]} == {1, 2}
    assert all(job["model_prefix"] == "dsv4" for job in plan["jobs"])

    contract = json.loads((tmp_path / "expected_artifact_contract.json").read_text())
    assert contract["total_jobs"] == 2
    required = contract["per_job"][0]["required_before_claiming_success"]
    assert any(path.endswith("/trace_replay/detailed_results.csv") for path in required)
    assert any(path.endswith("/provenance_preflight.jsonl") for path in required)

    sbatch_files = sorted((tmp_path / "sbatch").glob("*.sbatch"))
    assert len(sbatch_files) == 2
    rendered = sbatch_files[0].read_text()
    assert 'SCENARIO_TYPE="agentic-coding"' in rendered
    assert 'MODEL_PREFIX="dsv4"' in rendered
    assert 'TRACE_SOURCE="semianalysisai/cc-traces-weka-042026"' in rendered
    assert "nvidia-smi topo -m" in rendered
    assert "all_reduce_perf" in rendered
    assert "Dry-run sbatch rendered successfully" in rendered


def test_agentic_slurm_matrix_can_filter_gb200_multinode(tmp_path):
    runner = load_runner()
    rc = runner.main(
        [
            "--dry-run",
            "--results-root",
            str(tmp_path),
            "--hardware",
            "gb200",
            "--context-buckets",
            "8k",
            "--concurrency",
            "1",
            "--max-jobs",
            "1",
        ]
    )
    assert rc == 0

    plan = json.loads((tmp_path / "matrix_plan.json").read_text())
    assert plan["total_jobs"] == 1
    job = plan["jobs"][0]
    assert job["hardware"] == "gb200"
    assert job["is_multinode"] is True
    assert job["config_file"].endswith("disagg-gb200-low-latency.yaml")
