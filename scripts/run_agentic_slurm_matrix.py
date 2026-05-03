#!/usr/bin/env python3
"""Generate and optionally submit a GMI-facing agentic Slurm benchmark matrix.

The runner is intentionally dry-run-first: it renders sbatch files, a matrix
plan, and an expected artifact contract without claiming GPU behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "agentic_slurm_matrix.json"
SBATCH_TEMPLATE = REPO_ROOT / "scripts" / "slurm" / "agentic_job.sbatch.tmpl"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _slug(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return value.strip("-")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        if path.suffix == ".json":
            data = json.load(handle)
        else:
            try:
                import yaml  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"{path} requires PyYAML. Use the default JSON config or install pyyaml."
                ) from exc
            data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


@dataclass(frozen=True)
class MatrixJob:
    job_id: str
    hardware: str
    framework: str
    topology: str
    context_bucket: str
    concurrency: int
    arrival_mode: str
    cache_mode: str
    tenant_mode: str
    duration_seconds: int
    runner_script: str
    runner_name: str
    slurm_nodes: int
    gpus_per_node: int
    cpus_per_task: int
    time_limit: str
    model_prefix: str
    precision: str
    tp: int
    ep: int
    dp_attention: bool
    disagg: bool
    config_file: str
    is_multinode: bool
    prefill_num_workers: int
    prefill_tp: int
    prefill_ep: int
    prefill_dp_attention: bool
    decode_num_workers: int
    decode_tp: int
    decode_ep: int
    decode_dp_attention: bool
    trace_source: str

    @property
    def exp_name(self) -> str:
        return (
            f"{self.model_prefix}_{self.hardware}_{self.framework}_"
            f"{self.context_bucket}_conc{self.concurrency}"
        )

    @property
    def result_filename(self) -> str:
        return _slug(
            f"agentic_{self.exp_name}_{self.arrival_mode}_"
            f"{self.cache_mode}_{self.tenant_mode}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "hardware": self.hardware,
            "framework": self.framework,
            "topology": self.topology,
            "context_bucket": self.context_bucket,
            "concurrency": self.concurrency,
            "arrival_mode": self.arrival_mode,
            "cache_mode": self.cache_mode,
            "tenant_mode": self.tenant_mode,
            "duration_seconds": self.duration_seconds,
            "runner_script": self.runner_script,
            "runner_name": self.runner_name,
            "slurm_nodes": self.slurm_nodes,
            "gpus_per_node": self.gpus_per_node,
            "cpus_per_task": self.cpus_per_task,
            "time_limit": self.time_limit,
            "model_prefix": self.model_prefix,
            "precision": self.precision,
            "tp": self.tp,
            "ep": self.ep,
            "dp_attention": self.dp_attention,
            "disagg": self.disagg,
            "config_file": self.config_file,
            "is_multinode": self.is_multinode,
            "trace_source": self.trace_source,
            "result_filename": self.result_filename,
            "exp_name": self.exp_name,
            "prefill_num_workers": self.prefill_num_workers,
            "prefill_tp": self.prefill_tp,
            "prefill_ep": self.prefill_ep,
            "prefill_dp_attention": self.prefill_dp_attention,
            "decode_num_workers": self.decode_num_workers,
            "decode_tp": self.decode_tp,
            "decode_ep": self.decode_ep,
            "decode_dp_attention": self.decode_dp_attention,
        }


def expand_jobs(config: dict[str, Any], args: argparse.Namespace) -> list[MatrixJob]:
    defaults = config.get("defaults", {})
    hardware_cfg = config.get("hardware", {})
    if not isinstance(hardware_cfg, dict):
        raise ValueError("hardware must be a mapping")

    selected_hw = set(_parse_csv_strings(args.hardware) or hardware_cfg.keys())
    contexts = _parse_csv_strings(args.context_buckets) or _as_list(defaults.get("context_buckets"))
    concurrencies = _parse_csv_ints(args.concurrency) or _as_list(defaults.get("concurrency"))
    arrival_modes = _parse_csv_strings(args.arrival_modes) or _as_list(defaults.get("arrival_modes"))
    cache_modes = _parse_csv_strings(args.cache_modes) or _as_list(defaults.get("cache_modes"))
    tenant_modes = _parse_csv_strings(args.tenant_modes) or _as_list(defaults.get("tenant_modes"))

    jobs: list[MatrixJob] = []
    for hardware, hw in hardware_cfg.items():
        if hardware not in selected_hw or not hw.get("enabled", True):
            continue
        runner_script = REPO_ROOT / str(hw["runner_script"])
        if not runner_script.exists():
            raise FileNotFoundError(f"runner_script not found for {hardware}: {runner_script}")
        script_expected = hw.get("script_expected")
        if script_expected and not (REPO_ROOT / str(script_expected)).exists():
            raise FileNotFoundError(f"script_expected not found for {hardware}: {script_expected}")

        is_multinode = hw.get("topology") == "multi_node_disagg"
        prefill = hw.get("prefill", {})
        decode = hw.get("decode", {})
        tp = int(hw.get("tp", prefill.get("tp", 1)))
        ep = int(hw.get("ep", prefill.get("ep", 1)))
        dp_attention = bool(hw.get("dp_attention", prefill.get("dp_attention", False)))

        for context_bucket in contexts:
            for concurrency in concurrencies:
                for arrival_mode in arrival_modes:
                    for cache_mode in cache_modes:
                        for tenant_mode in tenant_modes:
                            key = "|".join(
                                [
                                    hardware,
                                    str(hw["framework"]),
                                    str(context_bucket),
                                    str(concurrency),
                                    str(arrival_mode),
                                    str(cache_mode),
                                    str(tenant_mode),
                                ]
                            )
                            job_id = hashlib.sha1(key.encode()).hexdigest()[:10]
                            jobs.append(
                                MatrixJob(
                                    job_id=job_id,
                                    hardware=hardware,
                                    framework=str(hw["framework"]),
                                    topology=str(hw["topology"]),
                                    context_bucket=str(context_bucket),
                                    concurrency=int(concurrency),
                                    arrival_mode=str(arrival_mode),
                                    cache_mode=str(cache_mode),
                                    tenant_mode=str(tenant_mode),
                                    duration_seconds=int(args.duration or defaults.get("duration_seconds", 1800)),
                                    runner_script=str(hw["runner_script"]),
                                    runner_name=str(hw["runner_name"]),
                                    slurm_nodes=int(hw.get("slurm_nodes", 1)),
                                    gpus_per_node=int(defaults.get("gpus_per_node", 8)),
                                    cpus_per_task=int(defaults.get("cpus_per_task", 64)),
                                    time_limit=str(defaults.get("time_limit", "04:00:00")),
                                    model_prefix=str(hw["model_prefix"]),
                                    precision=str(hw["precision"]),
                                    tp=tp,
                                    ep=ep,
                                    dp_attention=dp_attention,
                                    disagg=is_multinode,
                                    config_file=str(hw.get("config_file", "")),
                                    is_multinode=is_multinode,
                                    prefill_num_workers=int(prefill.get("num_worker", 0)),
                                    prefill_tp=int(prefill.get("tp", 0)),
                                    prefill_ep=int(prefill.get("ep", 0)),
                                    prefill_dp_attention=bool(prefill.get("dp_attention", False)),
                                    decode_num_workers=int(decode.get("num_worker", 0)),
                                    decode_tp=int(decode.get("tp", 0)),
                                    decode_ep=int(decode.get("ep", 0)),
                                    decode_dp_attention=bool(decode.get("dp_attention", False)),
                                    trace_source=str(defaults.get("trace_source", "")),
                                )
                            )

    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    return jobs


def expected_paths(job: MatrixJob) -> list[str]:
    return [
        f"{job.job_id}/{job.result_filename}.json",
        f"{job.job_id}/results/benchmark.log",
        f"{job.job_id}/results/benchmark_command.txt",
        f"{job.job_id}/results/server.log",
        f"{job.job_id}/results/trace_replay/detailed_results.csv",
        f"{job.job_id}/results/trace_replay/debug_trace.jsonl",
        f"{job.job_id}/preflight.log",
        f"{job.job_id}/provenance_preflight.jsonl",
    ]


def render_sbatch(job: MatrixJob, config: dict[str, Any], results_root: Path, dry_run_guard: bool) -> str:
    defaults = config.get("defaults", {})
    template = SBATCH_TEMPLATE.read_text()
    job_dir = results_root / job.job_id
    values = {
        **job.to_dict(),
        "job_name": f"agentic-{job.hardware}-{job.job_id}",
        "job_dir": str(job_dir),
        "partition_env": defaults.get("partition_env", "GMI_SLURM_PARTITION"),
        "account_env": defaults.get("account_env", "GMI_SLURM_ACCOUNT"),
        "results_root_env": defaults.get("results_root_env", "GMI_RESULTS_ROOT"),
        "container_image_env": defaults.get("container_image_env", "GMI_CONTAINER_IMAGE"),
        "model_path_env": defaults.get("model_path_env", "GMI_MODEL_PATH"),
        "dp_attention": str(job.dp_attention).lower(),
        "disagg": str(job.disagg).lower(),
        "is_multinode": "1" if job.is_multinode else "0",
        "prefill_dp_attention": str(job.prefill_dp_attention).lower(),
        "decode_dp_attention": str(job.decode_dp_attention).lower(),
        "dry_run_guard": "1" if dry_run_guard else "0",
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def write_outputs(config: dict[str, Any], jobs: list[MatrixJob], results_root: Path, dry_run_guard: bool) -> None:
    sbatch_dir = results_root / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job_dir = results_root / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        sbatch_text = render_sbatch(job, config, results_root, dry_run_guard)
        (sbatch_dir / f"{job.job_id}.sbatch").write_text(sbatch_text)

    plan = {
        "scenario": "agentic-coding",
        "total_jobs": len(jobs),
        "jobs": [job.to_dict() for job in jobs],
    }
    (results_root / "matrix_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    contract = {
        "scenario": "agentic-coding",
        "total_jobs": len(jobs),
        "per_job": [
            {
                "job_id": job.job_id,
                "result_filename": job.result_filename,
                "expected_paths": expected_paths(job),
                "required_before_claiming_success": [
                    f"{job.job_id}/{job.result_filename}.json",
                    f"{job.job_id}/results/trace_replay/detailed_results.csv",
                    f"{job.job_id}/preflight.log",
                    f"{job.job_id}/provenance_preflight.jsonl",
                ],
            }
            for job in jobs
        ],
    }
    (results_root / "expected_artifact_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def submit_jobs(config: dict[str, Any], results_root: Path, jobs: list[MatrixJob]) -> None:
    defaults = config.get("defaults", {})
    partition_env = defaults.get("partition_env", "GMI_SLURM_PARTITION")
    account_env = defaults.get("account_env", "GMI_SLURM_ACCOUNT")
    partition = os.environ.get(partition_env)
    account = os.environ.get(account_env)
    if not partition:
        raise RuntimeError(f"{partition_env} must be set when --submit is used")
    for job in jobs:
        sbatch_path = results_root / "sbatch" / f"{job.job_id}.sbatch"
        cmd = ["sbatch", "--partition", partition]
        if account:
            cmd.extend(["--account", account])
        cmd.append(str(sbatch_path))
        subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=Path(os.environ.get("GMI_RESULTS_ROOT", "agentic-slurm-results")))
    parser.add_argument("--hardware", help="Comma-separated hardware filter, e.g. b200,b300")
    parser.add_argument("--context-buckets", help="Comma-separated context buckets")
    parser.add_argument("--concurrency", help="Comma-separated concurrency values")
    parser.add_argument("--arrival-modes", help="Comma-separated arrival modes")
    parser.add_argument("--cache-modes", help="Comma-separated cache modes")
    parser.add_argument("--tenant-modes", help="Comma-separated tenant modes")
    parser.add_argument("--duration", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Render files only")
    parser.add_argument("--submit", action="store_true", help="Submit rendered sbatch jobs")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.submit and args.dry_run:
        parser.error("--submit and --dry-run are mutually exclusive")

    config = _load_config(args.config)
    jobs = expand_jobs(config, args)
    args.results_root.mkdir(parents=True, exist_ok=True)
    write_outputs(config, jobs, args.results_root, dry_run_guard=args.dry_run or not args.submit)

    if args.submit:
        submit_jobs(config, args.results_root, jobs)

    print(f"Wrote {len(jobs)} agentic Slurm jobs to {args.results_root}")
    print(f"Matrix plan: {args.results_root / 'matrix_plan.json'}")
    print(f"Artifact contract: {args.results_root / 'expected_artifact_contract.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
