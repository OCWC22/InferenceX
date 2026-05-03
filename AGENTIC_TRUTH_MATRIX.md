# SemiAnalysis InferenceX Agentic/WEKA Truth Matrix

Date: 2026-05-03

Scope: local `InferenceX` checkout, focused on the PR path that adds the `agentic-coding` scenario and WEKA trace replay. This is a truth matrix for deciding what still needs to be built before a GMI Cloud or other neocloud platform engineer can use the harness to evaluate real long-context chat and coding inference workloads.

## Bottom Line

The current InferenceX implementation is a real but experimental **agentic trace replay harness**. It replays recorded WEKA coding/chat traces against an OpenAI-compatible serving endpoint and emits latency, throughput, cache, workload-distribution, and artifact outputs.

It is **not yet** a complete GMI/neocloud evaluation harness for DeepSeek-V4 on B200/B300/GB200. The biggest gap is that `agentic-coding` is wired for some DeepSeek-R1 and GPT-OSS/Kimi paths, while the DeepSeek-V4 GB200/B300/B200 surface is still mostly fixed-sequence or srt-slurm recipe driven. The harness also does not yet produce the full cluster, network, reliability, cost, and operator-readiness evidence that a cloud platform engineer would need.

## Actual Code Path Today

| Step | What happens | Actual code | Truth status |
|---|---|---|---|
| 1 | Config declares an optional `agentic-coding` scenario. | `.github/configs/CONFIGS.md` | Exists |
| 2 | NVIDIA/AMD master configs include a small number of `agentic-coding` entries. | `.github/configs/nvidia-master.yaml`, `.github/configs/amd-master.yaml` | Exists, narrow |
| 3 | Matrix generator expands agentic entries across concurrency, TP, EP, DP attention, offload, runner, image, model, and duration. | `utils/matrix_logic/generate_sweep_configs.py` | Exists |
| 4 | GitHub workflow sets agentic routing env vars. | `.github/workflows/benchmark-tmpl.yml` | Exists |
| 5 | Runner selects `benchmarks/single_node/agentic/...` instead of normal fixed-seq scripts. | `runners/launch_*.sh` via `SCENARIO_SUBDIR=agentic/` | Exists |
| 6 | Shared library resolves WEKA trace source and builds the replay command. | `benchmarks/benchmark_lib.sh` | Exists |
| 7 | Agentic script starts the serving backend and runs trace replay. | `benchmarks/single_node/agentic/dsr1_fp4_b200.sh`, peers | Exists |
| 8 | Multi-node agentic path runs client-only replay against an already-started srt-slurm frontend. | `benchmarks/multi_node/agentic_srt.sh` | Exists, experimental |
| 9 | Aggregator turns replay CSVs into InferenceX-like JSON. | `utils/process_agentic_result.py` | Exists |
| 10 | Workflow uploads raw and aggregated artifacts. | `.github/workflows/benchmark-tmpl.yml`, `.github/workflows/e2e-tests.yml` | Exists |

## Actual Code Snippets

The trace source is hardcoded to a Hugging Face dataset:

```bash
local dataset="semianalysisai/cc-traces-weka-042026"
TRACE_SOURCE_FLAG="--hf-dataset $dataset"
```

Source: `benchmarks/benchmark_lib.sh`

Agentic replay is built as a client workload against the local serving endpoint:

```bash
REPLAY_CMD="python3 $TRACE_REPLAY_DIR/trace_replay_tester.py"
REPLAY_CMD+=" --api-endpoint http://localhost:$PORT"
REPLAY_CMD+=" $TRACE_SOURCE_FLAG"
REPLAY_CMD+=" --output-dir $result_dir/trace_replay"
REPLAY_CMD+=" --start-users $CONC"
REPLAY_CMD+=" --max-users $CONC"
REPLAY_CMD+=" --test-duration $duration"
REPLAY_CMD+=" --recycle"
REPLAY_CMD+=" --warmup-enabled"
REPLAY_CMD+=" --seed 42"
```

Source: `benchmarks/benchmark_lib.sh`

The workflow routes agentic jobs by setting:

```yaml
SCENARIO_SUBDIR: ${{ inputs.scenario-type == 'agentic-coding' && 'agentic/' || '' }}
IS_AGENTIC: ${{ inputs.scenario-type == 'agentic-coding' && '1' || '0' }}
RESULT_DIR: /workspace/results
```

Source: `.github/workflows/benchmark-tmpl.yml`

The B200 DeepSeek-R1 agentic script starts SGLang, waits for readiness, runs replay, then aggregates:

```bash
resolve_trace_source
install_agentic_deps
python3 -m sglang.launch_server ... --enable-metrics > "$SERVER_LOG" 2>&1 &
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
build_replay_cmd "$RESULT_DIR"
$REPLAY_CMD 2>&1 | tee "$RESULT_DIR/benchmark.log" || true
write_agentic_result_json "$RESULT_DIR"
```

Source: `benchmarks/single_node/agentic/dsr1_fp4_b200.sh`

Aggregated JSON includes scenario identity, topology, success counts, latency, throughput, token distributions, cache stats, and per-GPU throughput:

```python
agg = {
    "hw": os.environ.get('RUNNER_TYPE', ''),
    "conc": conc,
    "model": os.environ.get('MODEL', ''),
    "framework": os.environ.get('FRAMEWORK', ''),
    "scenario_type": "agentic-coding",
    "is_multinode": is_multinode,
    "tp": tp,
    "ep": ep,
    "offloading": os.environ.get('OFFLOADING', 'none'),
    "num_requests_total": len(rows),
    "num_requests_successful": len(successful),
}
```

Source: `utils/process_agentic_result.py`

## What The Use Case Actually Is

The current use case is:

| Use case | Current behavior |
|---|---|
| Replay realistic coding/chat request traces | Yes, via `semianalysisai/cc-traces-weka-042026`. |
| Drive a serving endpoint with concurrent users | Yes, with `--start-users $CONC` and `--max-users $CONC`. |
| Measure request-level TTFT / E2E / ITL / TPOT | Yes, from `trace_replay/detailed_results.csv`. |
| Measure throughput and throughput per GPU | Yes, from completed request timestamps and configured GPU counts. |
| Measure input/output token distribution | Yes, from replay rows. |
| Estimate cache reuse | Partially. It reports theoretical replay cache hit rate and server prefix-cache counters when metrics exist. |
| Evaluate real autonomous coding agent behavior | No. It replays traces; it does not run an agent loop with tools, repo edits, tests, retries, or feedback. |
| Evaluate GMI customer traffic | No, unless GMI traffic is converted into the same trace-replay format. |

## Current Coverage Matrix

| Surface | Current status | Notes |
|---|---|---|
| DeepSeek-R1 FP4 B200 SGLang single-node agentic | Exists | `benchmarks/single_node/agentic/dsr1_fp4_b200.sh`. |
| DeepSeek-R1 FP4 B200 Dynamo/TRT multi-node agentic | Exists, experimental | Uses a special `cquil11/srt-slurm-nv` branch and a `128k_agentic` recipe. |
| DeepSeek-R1 FP4 MI355X SGLang single-node agentic | Exists | AMD entry in `.github/configs/amd-master.yaml`. |
| GPT-OSS FP4 H100/H200/MI300X/MI325X agentic scripts | Exists as scripts | Need config coverage and live validation per target. |
| Kimi K2.5 FP4 B200 agentic script | Exists as script | Need config coverage and live validation. |
| DeepSeek-V4 B200/B300 SGLang fixed-seq | Exists | Fixed 1k/8k surfaces, not agentic trace replay. |
| DeepSeek-V4 B200/B300 vLLM fixed-seq/MTP | Exists | Fixed-seq path with DSV4 chat encoding. |
| DeepSeek-V4 GB200 vLLM srt-slurm recipes | Exists | Recipe set for 8k1k, not agentic trace replay. |
| DeepSeek-V4 GB200 agentic trace replay | Missing | No `agentic-coding` config or DSV4-specific agentic launcher found. |
| B300 agentic trace replay | Mostly missing | B300 has fixed-seq DSR1/DSV4 surfaces, not a clear agentic path. |
| LMCache/TensorMesh agentic comparison | Missing | No direct LMCache/TensorMesh metrics integration in InferenceX agentic path. |

## What A GMI/Neocloud Platform Engineer Actually Cares About

| Category | What they need to decide | Current harness answer | Gap |
|---|---|---|---|
| Capacity planning | How many concurrent coding/chat sessions per node or rack before SLO violation? | Partial: concurrency sweep and request latencies. | Needs SLO pass/fail curves, saturation point, and capacity recommendations. |
| Latency SLO | P50/P90/P99 TTFT, TPOT, E2E for long-context chat/coding. | Partial: computes latency stats. | Needs explicit SLO config, pass/fail, and stable run windows. |
| Long context | How 32k/64k/128k/256k+ context behaves under realistic reuse. | Partial: WEKA traces may include realistic shapes, but context buckets are not first-class in matrix. | Needs explicit context-length stratification and reporting. |
| Coding workload realism | Does traffic resemble coding assistants, repo Q&A, edits, tests, tool calls? | Partial: recorded traces, but no task taxonomy shown in core benchmark output. | Needs workload classes: code chat, repo QA, patch generation, test/debug loop, long-doc coding. |
| Cache value | Does prefix/KV reuse improve latency, cost, and throughput? | Partial: theoretical and server prefix-cache hit metrics. | Needs engine-specific cache event metrics, eviction, residency, fragmentation, reuse distance, cache salt/isolation. |
| Multi-tenant isolation | Does one tenant poison or evict another tenant's cache? | Missing. | Needs tenant IDs, cache salts, fairness and isolation reports. |
| Memory pressure | When do KV cache, CPU offload, swap, or SSD tiers collapse? | Partial: offloading field and a few counters. | Needs GPU memory, HBM pressure, CPU memory, SSD bandwidth, eviction storms, OOM attribution. |
| Slurm operator flow | Can an operator dry-run, submit, monitor, cancel, and collect artifacts? | Partial in InferenceX CI and srt-slurm paths. | Needs portable Slurm matrix runner, sbatch rendering, env-only cluster config, and artifact contract. |
| Network health | Are NCCL/RDMA/NVLink/IB topology problems caught before benchmark? | Missing in InferenceX agentic path. | Needs preflight topology and network smoke checks. |
| Reproducibility | Can results be traced to image digest, repo SHA, GPU inventory, driver, topology, and versions? | Partial: CI has image/model/framework fields. | Needs full provenance captured per job. |
| Reliability | Do runs survive cold start, warmup, long duration, failed requests, server restarts? | Partial: success counts and raw logs. | Needs failure taxonomy, retry policy, health timeline, and soak tests. |
| Cost model | Which hardware/runtime gives best $/successful-session or $/M tokens at SLO? | Missing. | Needs GPU-hour pricing input and cost-per-SLO report. |
| Hardware comparison | B200 vs B300 vs GB200 for the same DSV4 workload. | Missing for agentic. | Need same workload across same engines and configs. |
| Runtime comparison | vLLM vs SGLang vs TRT/Dynamo under identical trace replay. | Partial for some models. | Need normalized DSV4 matrix and identical trace/scheduler settings. |
| Production readiness | What config should GMI actually offer customers? | Missing. | Needs recommended SKUs, caveats, and no-go thresholds. |

## Truth Matrix: Current vs Required

Legend:

- Yes: implemented in the local InferenceX path.
- Partial: implemented but too narrow, experimental, or missing key evidence.
- No: not implemented.
- Unknown: cannot be proven from this repo without live cluster results or external data.

| Requirement | Current truth | Evidence | Needed build |
|---|---:|---|---|
| Agentic scenario flag and config schema | Yes | `agentic-coding` in config docs and validation. | Keep. |
| WEKA trace replay source | Yes | `semianalysisai/cc-traces-weka-042026` in `resolve_trace_source`. | Make dataset configurable; keep WEKA as default/example. |
| Single-node trace replay execution | Yes | `benchmarks/single_node/agentic/*.sh`. | Add DSV4 B200/B300 launchers. |
| Multi-node trace replay execution | Partial | `benchmarks/multi_node/agentic_srt.sh`; special srt-slurm branch. | First-class srt-slurm support, no special private branch dependency. |
| DeepSeek-V4 B200 agentic | No | DSV4 B200 configs are fixed-seq, not `agentic-coding`. | Add config + launcher + validated run. |
| DeepSeek-V4 B300 agentic | No | B300 has DSV4 fixed-seq scripts/recipes, not agentic. | Add config + launcher + validated run. |
| DeepSeek-V4 GB200 agentic | No | GB200 DSV4 recipes exist, but no agentic scenario. | Add srt-slurm agentic recipe and config. |
| B200/B300/GB200 apples-to-apples matrix | No | Current surfaces differ by model/runtime/scenario. | Build normalized matrix over hardware, engine, context, concurrency. |
| vLLM/SGLang/TRT/Dynamo comparison for same workload | Partial | Some engines covered for some models. | Normalize exact model, precision, prompt encoding, trace, and duration. |
| Long-context buckets | Partial | Fixed-seq has 1k/8k; trace replay may have varied token lengths. | Add explicit 8k/32k/64k/128k/256k+ bins in reports and optional filters. |
| Coding workload taxonomy | Partial | Trace replay exists; distribution plot exists. | Add task labels and per-class metrics. |
| TTFT/TPOT/E2E latency metrics | Yes | `compute_latency_stats`. | Add SLO pass/fail summary. |
| Throughput per GPU | Yes | `tput_per_gpu` in processor. | Add SLO-qualified throughput, not just raw throughput. |
| Failed request taxonomy | Partial | Success count exists. | Add HTTP error class, timeout, OOM, scheduler reject, engine crash. |
| Prefix/KV cache hit metrics | Partial | Theoretical + server prefix counters when present. | Add LMCache/TensorMesh/vLLM/SGLang metric adapters with measured-vs-inferred flags. |
| Eviction/fragmentation proof | No | No live cache event schema in agentic path. | Add engine metric scraping and artifact schema. |
| Multi-tenant cache isolation | No | No tenant IDs or cache salt model. | Add multi-tenant trace mode and isolation metrics. |
| CPU/SSD offload analysis | Partial | `offloading` field and some counters in processor. | Add tier residency, bandwidth, latency, and failure attribution. |
| Slurm dry-run matrix generation | Partial | InferenceX CI/srt-slurm flow exists; not a portable GMI operator runner. | Add portable Slurm matrix runner and artifact contract. |
| NCCL/RDMA/topology preflight | No | Not in agentic path. | Add pre-benchmark smoke checks. |
| Full provenance capture | Partial | JSON includes image/model/framework; raw logs upload. | Add digest, repo SHA, driver, CUDA, GPU inventory, topology, package versions. |
| Cost and capacity report | No | No pricing or recommendation layer. | Add cost inputs and capacity planning report. |
| Customer-ready operator report | No | Raw/aggregated artifacts only. | Add one-page operator brief with recommendations and caveats. |

## Recommended Build Matrix For GMI/Neocloud Evaluation

This is the minimum useful matrix for a GMI cloud engineer evaluating long-context chat and coding workloads. It is intentionally smaller than a full combinatorial sweep.

| Axis | Required values | Why it matters |
|---|---|---|
| Hardware | B200, B300, GB200 | These are the procurement/deployment choices. |
| Model | DeepSeek-V4-Pro first; DeepSeek-R1 as control | DSV4 is the target; DSR1 provides existing harness continuity. |
| Runtime | vLLM, SGLang, Dynamo/TRT where supported | GMI needs runtime/SKU decision data. |
| Topology | single-node, multi-node disagg | Long context and MoE behavior differ sharply by topology. |
| Context bucket | 8k, 32k, 64k, 128k, 256k+ | Cloud operators need max supported context and degradation curve. |
| Workload type | long chat, repo QA, code generation, test/debug loop, multi-turn agent | Coding traffic is not one workload. |
| Concurrency | 1, 2, 4, 8, 16, 32, 64, 128, then saturation search | Finds knee of curve and failure region. |
| Arrival mode | closed-loop and burst/open-loop | Closed-loop measures users; open-loop exposes queue collapse. |
| Cache mode | cache off, engine prefix cache, LMCache/TensorMesh if available | Proves whether cache stack actually helps. |
| Tenant mode | single tenant, multi-tenant with cache salt | Proves isolation and fairness. |
| Duration | 10 min smoke, 30 min curve, 2-4 hr soak | Separates launch success from operational stability. |

## What To Build Next

| Priority | Build item | Acceptance criteria |
|---:|---|---|
| P0 | DSV4 `agentic-coding` configs for B200/B300/GB200 | Matrix generator emits DSV4 agentic jobs for each target hardware without touching fixed-seq paths. |
| P0 | DSV4 agentic launchers | Single-node launchers exist for B200/B300; GB200 multi-node agentic recipe exists or maps cleanly to srt-slurm custom benchmark. |
| P0 | Portable Slurm matrix runner | GMI operator can dry-run and submit without GitHub Actions; no hardcoded cluster IDs; all cluster settings via env/YAML. |
| P0 | Artifact contract | Every run emits a normalized JSON, raw CSV/JSONL, server log, config, command, provenance, and expected-path manifest. |
| P1 | Workload taxonomy and context buckets | Report breaks down metrics by workload class and context-length bucket. |
| P1 | SLO/capacity report | For each cell, report max concurrency at TTFT/TPOT/E2E SLO and failure reason beyond it. |
| P1 | Provenance capture | Per-job artifact records image digest, repo SHA, CUDA/driver, GPU inventory, topology, runtime versions, Slurm job ID, nodelist. |
| P1 | NCCL/RDMA/topology preflight | Preflight emits pass/fail/skipped before benchmark execution. |
| P1 | Cache metrics adapters | vLLM/SGLang/LMCache/TensorMesh metrics are normalized with measured vs inferred labels. |
| P2 | Multi-tenant replay mode | Tenant IDs, cache salt/isolation, fairness, noisy-neighbor metrics. |
| P2 | Cost model | Add GPU-hour price input and output $/successful-session, $/M input tokens, $/M output tokens at SLO. |
| P2 | Operator brief | Generate a human-readable recommendation with caveats: best config, no-go configs, saturation point, and missing proof. |

## Non-Claims To Preserve

Do not claim any of the following until live artifacts prove them:

- DeepSeek-V4 agentic performance on GB200.
- B200/B300/GB200 parity under the same long-context trace replay.
- LMCache/TensorMesh benefit.
- Cache eviction or fragmentation behavior.
- Multi-tenant isolation.
- Production readiness for GMI customer workloads.
- Autonomous agent performance; this is trace replay, not a tool-using agent loop.

## Proposed File/Code Changes For The Next PR

| Area | Candidate files |
|---|---|
| DSV4 agentic configs | `.github/configs/nvidia-master.yaml`, possibly a separate GMI/GPU pilot config. |
| DSV4 single-node launchers | `benchmarks/single_node/agentic/dsv4_fp4_b200_sglang.sh`, `benchmarks/single_node/agentic/dsv4_fp4_b300_sglang.sh`, vLLM variants if supported. |
| GB200 multi-node agentic | `benchmarks/multi_node/agentic_srt.sh`, `benchmarks/multi_node/srt-slurm-recipes/.../deepseek-v4/...`, `runners/launch_gb200-nv.sh`. |
| Slurm operator harness | `scripts/slurm/`, `scripts/run_agentic_slurm_matrix.py`, `configs/agentic_slurm_matrix.yaml`. |
| Metrics schema | `utils/process_agentic_result.py` plus a new normalized metrics schema module. |
| Artifact contract tests | `utils/matrix_logic/test_*.py` or new repo-level tests for dry-run contract. |
| Operator report | new `utils/summarize_agentic.py` or integration into `utils/summarize.py`. |

## Decision

For GMI/neocloud evaluation, the current InferenceX PR is a **good starting mechanism**, not a finished benchmark product. Build the missing DSV4+B200/B300/GB200 agentic Slurm surface, add provenance/preflight/cache/SLO reporting, and keep every unmeasured claim explicitly labeled as unproven.
