#!/usr/bin/env python3
"""Generate deterministic synthetic JSONL traces for KV-cache stress scenarios.

All generation uses an explicit random.Random(seed) instance for full
reproducibility. The same seed + scenario always produces identical output.

Usage:
    python generate_synthetic_traces.py --scenario prefix_reuse_sweep --seed 42
    python generate_synthetic_traces.py --scenario kv_offload_cliff --output traces/out.jsonl
    python generate_synthetic_traces.py --scenario long_reasoning_growth --seed 42
    python generate_synthetic_traces.py --all --seed 42 --output-dir traces/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as a standalone script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from schema import TraceEntry


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def generate_prefix_reuse_sweep(rng: random.Random, num_entries: int = 20) -> list[dict]:
    """Controlled prefix hit-rate stress.

    Creates multiple sessions sharing system prefixes with varying
    shared_prefix_tokens and expected_prefix_hit_rate values.
    """
    prefixes = ["sys_prefix_coding", "sys_prefix_chat"]
    hit_rates = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
    prefix_lengths = [128, 512, 2048, 4096]
    suffix_lengths = [32, 128, 512]

    entries: list[dict] = []
    for i in range(num_entries):
        prefix_id = prefixes[i % len(prefixes)]
        hit_rate = hit_rates[i % len(hit_rates)]
        prefix_len = prefix_lengths[i % len(prefix_lengths)]
        suffix_len = rng.choice(suffix_lengths)
        input_total = prefix_len + suffix_len
        session_id = f"sess_prefix_{i // 4:03d}"
        turn_id = i % 4

        entries.append({
            "trace_id": f"prefix_reuse_{i:04d}",
            "session_id": session_id,
            "turn_id": turn_id,
            "workload_family": "kv_cache_stress",
            "scenario": "prefix_reuse_sweep",
            "prompt": f"[synthetic prefix-reuse prompt {i}]",
            "system_prefix_id": prefix_id,
            "parent_turn_id": turn_id - 1 if turn_id > 0 else None,
            "input_tokens_total": input_total,
            "shared_prefix_tokens": prefix_len,
            "unique_suffix_tokens": suffix_len,
            "expected_output_tokens": rng.choice([64, 128, 256, 512]),
            "expected_prefix_hit_rate": hit_rate,
            "idle_ms_since_last_turn": 0.0 if turn_id == 0 else rng.uniform(100, 5000),
            "arrival_time_ms": float(i * 1000),
            "residency_tier_hint": "gpu",
            "requires_quality_eval": False,
            "quality_task": None,
            "gold_answer": None,
            "tags": ["prefix_reuse", f"hit_rate_{hit_rate}"],
        })

    return entries


def generate_kv_offload_cliff(rng: random.Random, num_entries: int = 15) -> list[dict]:
    """Force GPU KV pressure and offload/restore behavior.

    Simulates sessions that go idle for varying durations, causing KV cache
    to migrate from GPU -> CPU -> disk. Tests reactivation latency.
    """
    idle_times_ms = [0, 100, 500, 1000, 5000, 10000, 30000, 60000, 120000, 300000, 600000]
    tiers = ["gpu", "gpu", "gpu", "gpu", "cpu", "cpu", "cpu", "cpu", "disk", "disk", "disk"]
    context_lengths = [1024, 2048, 4096, 8192, 16384]

    entries: list[dict] = []
    for i in range(num_entries):
        idle_ms = idle_times_ms[i % len(idle_times_ms)]
        tier = tiers[i % len(tiers)]
        ctx_len = rng.choice(context_lengths)
        session_id = f"sess_offload_{i // 5:03d}"
        turn_id = i % 5

        entries.append({
            "trace_id": f"offload_cliff_{i:04d}",
            "session_id": session_id,
            "turn_id": turn_id,
            "workload_family": "kv_cache_stress",
            "scenario": "kv_offload_cliff",
            "prompt": f"[synthetic offload-cliff prompt {i}]",
            "system_prefix_id": None,
            "parent_turn_id": turn_id - 1 if turn_id > 0 else None,
            "input_tokens_total": ctx_len,
            "shared_prefix_tokens": 0,
            "unique_suffix_tokens": ctx_len,
            "expected_output_tokens": rng.choice([128, 256, 512]),
            "expected_prefix_hit_rate": 0.0,
            "idle_ms_since_last_turn": float(idle_ms),
            "arrival_time_ms": float(i * 2000),
            "residency_tier_hint": tier,
            "requires_quality_eval": False,
            "quality_task": None,
            "gold_answer": None,
            "tags": ["offload_cliff", f"tier_{tier}", f"idle_{idle_ms}ms"],
        })

    return entries


def generate_long_reasoning_growth(rng: random.Random, num_entries: int = 10) -> list[dict]:
    """Long reasoning decode stress for quality tradeoff measurement.

    Simulates chain-of-thought sessions where context grows substantially
    across turns. Each entry has quality evaluation enabled.
    """
    context_schedule = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 131072, 131072]
    output_schedule = [512, 1024, 2048, 4096, 8192, 16384, 32768, 32768, 32768, 32768]
    quality_tasks = ["gsm8k", "gpqa_diamond", "aime25", "math500"]

    entries: list[dict] = []
    for i in range(num_entries):
        ctx_len = context_schedule[min(i, len(context_schedule) - 1)]
        out_len = output_schedule[min(i, len(output_schedule) - 1)]
        session_id = f"sess_reasoning_{i // 5:03d}"
        turn_id = i % 5
        task = rng.choice(quality_tasks)

        entries.append({
            "trace_id": f"reasoning_growth_{i:04d}",
            "session_id": session_id,
            "turn_id": turn_id,
            "workload_family": "kv_cache_stress",
            "scenario": "long_reasoning_growth",
            "prompt": f"[synthetic reasoning prompt {i}: solve step {turn_id + 1}]",
            "system_prefix_id": None,
            "parent_turn_id": turn_id - 1 if turn_id > 0 else None,
            "input_tokens_total": ctx_len,
            "shared_prefix_tokens": 0,
            "unique_suffix_tokens": ctx_len,
            "expected_output_tokens": out_len,
            "expected_prefix_hit_rate": 0.0,
            "idle_ms_since_last_turn": 0.0,
            "arrival_time_ms": float(i * 5000),
            "residency_tier_hint": "gpu",
            "requires_quality_eval": True,
            "quality_task": task,
            "gold_answer": f"[gold answer placeholder for {task} step {turn_id + 1}]",
            "tags": ["long_reasoning", f"ctx_{ctx_len}", f"task_{task}"],
        })

    return entries


GENERATORS = {
    "prefix_reuse_sweep": generate_prefix_reuse_sweep,
    "kv_offload_cliff": generate_kv_offload_cliff,
    "long_reasoning_growth": generate_long_reasoning_growth,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_jsonl(entries: list[dict], output: Path | None) -> None:
    """Write entries as JSONL to a file or stdout."""
    # Validate all entries before writing
    for i, entry in enumerate(entries):
        try:
            TraceEntry(**entry)
        except Exception as e:
            raise ValueError(f"Generated entry {i} failed validation: {e}") from e

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    else:
        for entry in entries:
            print(json.dumps(entry, separators=(",", ":")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic KV-stress traces."
    )
    parser.add_argument(
        "--scenario",
        choices=list(GENERATORS.keys()),
        help="Scenario to generate (required unless --all is used)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--num-entries",
        type=int,
        default=None,
        help="Override default entry count for the scenario",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output JSONL file (default: stdout)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all scenarios. Requires --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for --all mode",
    )
    args = parser.parse_args()

    if args.all:
        if not args.output_dir:
            parser.error("--output-dir is required with --all")
        filenames = {
            "prefix_reuse_sweep": "smoke_prefix_reuse.jsonl",
            "kv_offload_cliff": "smoke_offload_cliff.jsonl",
            "long_reasoning_growth": "smoke_long_reasoning_growth.jsonl",
        }
        for scenario, gen_fn in GENERATORS.items():
            rng = random.Random(args.seed)
            kwargs = {}
            if args.num_entries is not None:
                kwargs["num_entries"] = args.num_entries
            entries = gen_fn(rng, **kwargs)
            out_path = args.output_dir / filenames[scenario]
            write_jsonl(entries, out_path)
            print(f"Wrote {len(entries)} entries to {out_path}")
    else:
        if not args.scenario:
            parser.error("--scenario is required unless --all is used")
        rng = random.Random(args.seed)
        gen_fn = GENERATORS[args.scenario]
        kwargs = {}
        if args.num_entries is not None:
            kwargs["num_entries"] = args.num_entries
        entries = gen_fn(rng, **kwargs)
        write_jsonl(entries, args.output)
        if args.output:
            print(f"Wrote {len(entries)} entries to {args.output}")


if __name__ == "__main__":
    main()
