# ISB1 KV Cache Benchmark — Lane B Bare-Metal Execution Plan

Bare-metal execution runbook for ISB1 replay bundles on **Hopper (H100, H200)**
and the full **Blackwell family (B200, B300, GB200, GB300)**, across
**GMI Cloud** and **AWS** EC2 bare-metal instances. All runs described here are `support_status=reviewed_preview`
with `benchmark_certification_status=dataset_replay_verified` — i.e. replay and
export certification, not live-serving certification.

Lane B is deliberately **not Slurm** — every job launches the engine directly
via `docker run` from a single node. Lane A (Slurm-backed) is a separate
runbook and lives on the SemiAnalysis side.

## Available Hardware

| GPU | HBM | Cloud notes | Max Context Before Offload (FP8 KV) |
|-----|-----|-------------|-------------------------------------|
| **H100 SXM 80GB** | 80 GB HBM3 | GMI, AWS `p5.48xlarge` | ~128K tokens |
| **H200 SXM 141GB** | 141 GB HBM3e | GMI, AWS `p5e.48xlarge` | ~200K tokens |
| **B200 SXM 180GB** | 180 GB HBM3e | GMI, AWS `p6-b200.48xlarge` | ~384K tokens |
| **B300 SXM 288GB** *(Blackwell Ultra)* | 288 GB HBM3e | GMI (not yet announced), AWS `p6-b300` (GovCloud GA April 2026) | ~500K+ tokens *(preliminary)* |
| **GB200 NVL72 192GB** | 192 GB HBM3e (Grace-attached, **aarch64**) | GMI, AWS `p6e-gb200` | ~500K+ tokens |
| **GB300 NVL72 288GB** *(Blackwell Ultra)* | 288 GB HBM3e (Grace-attached, **aarch64**) | GMI (not yet announced), AWS `p6e-gb300` (coming soon) | ~1M+ tokens *(preliminary)* |

GB200 and GB300 are the aarch64 targets in this matrix; B300 is x86 but
Blackwell Ultra (sm_103) with no committed public image tag yet. The
portable script will refuse to launch without an appropriate image
override for any of these three SKUs — see
[Image resolution](#image-resolution).

## Script Surface (canonical)

All Lane B jobs go through `gmi_portable_benchmark.sh`. The `--cloud` flag
is new and defaults to `gmi` for backward compatibility. Runner labels are
`<gpu>-<cloud>-baremetal` (e.g. `gb200-aws-baremetal`). SQLite rows carry a
matching `cloud` column.

```bash
datasets/isb1/scripts/gmi_portable_benchmark.sh \
  --gpu-type <h100|h200|b200|b300|gb200|gb300> \
  [--cloud <gmi|aws>] \
  --model <qwen3.5|gptoss|dsr1> \
  --engine <vllm|sglang> \
  --context-band <8k|32k|64k|131k|500k|1m> \
  --workload <chat|code> \
  [--benchmark-type <isb1_replay|isb1_kv_stress>]
```

## Image resolution

Lookup order (first non-empty wins):

1. `LANEB_IMAGE_<CLOUD>_<GPU>_<ENGINE>` — most specific, e.g.
   `LANEB_IMAGE_GMI_GB200_VLLM`.
2. `LANEB_IMAGE_<GPU>_<ENGINE>` — cloud-agnostic per-GPU override.
3. Default baked into `_gmi_common.sh` (`laneb_default_image`). **GB200,
   B300, and GB300 have no default** — the script hard-fails until you
   export an override for each. This is intentional: B300 is a distinct
   Blackwell Ultra (sm_103), GB200/GB300 are aarch64 Grace-attached, and
   silently inheriting a B200 image would fail deep inside the container
   with a much less obvious error.

Worked examples:

```bash
# H100/H200/B200 work out of the box with committed defaults.
#   (vllm:   vllm/vllm-openai:v0.18.0, or v0.19.0-cu130 on b200)
#   (sglang: lmsysorg/sglang:v0.5.9-cu130)

# GB200 on GMI — pin your private aarch64 image:
export LANEB_IMAGE_GMI_GB200_VLLM="vllm/vllm-openai:v0.19.0-cu130-arm64"
export LANEB_IMAGE_GMI_GB200_SGLANG="lmsysorg/sglang:v0.5.9-cu130-arm64"

# GB200 on AWS — usually different registry/auth, override per-cloud:
export LANEB_IMAGE_AWS_GB200_VLLM="public.ecr.aws/aws-inferencex/vllm-grace:2026-04"

# B300 (Blackwell Ultra x86) — pin a sm_103-capable image:
export LANEB_IMAGE_AWS_B300_VLLM="<vllm image built for sm_103 / CUDA 13.x>"
export LANEB_IMAGE_AWS_B300_SGLANG="<sglang image built for sm_103 / CUDA 13.x>"

# GB300 (Blackwell Ultra aarch64) — override per cloud:
export LANEB_IMAGE_AWS_GB300_VLLM="<vllm image: sm_103 + aarch64 Grace>"
export LANEB_IMAGE_AWS_GB300_SGLANG="<sglang image: sm_103 + aarch64 Grace>"
```

If the script dies with `No container image resolved`, the error message
tells you exactly which env var to set.

## Execution Order

Run benchmarks in this order — cheapest/fastest first to validate the setup
works before burning expensive GPU-hours on long-context runs.

### Phase 1: Validation Run (~1 hour per GPU)

Prove the pipeline works end-to-end before burning GPU hours.

```bash
# On H100 (GMI), single curated cell, 5 min.
export HF_TOKEN=hf_...
datasets/isb1/scripts/gmi_portable_benchmark.sh \
  --gpu-type h100 --cloud gmi \
  --model dsr1 --engine vllm \
  --context-band 131k --workload code \
  --max-concurrency 2 --benchmark-duration-s 300
```

**Pass criteria:** process exits 0, `agg_*.json` lands in
`datasets/isb1/results/gmi/laneb-gmi-h100-.../`, and the run appears when
you query the DB:

```bash
python3 datasets/isb1/scripts/isb1_results_db.py query \
  --group-by cloud,gpu_type,model,engine
```

### Phase 2: H100/H200 KV Stress Sweep (8–12 hours per GPU)

H100 80GB is the interesting Hopper GPU — KV cache fills up first. H200
141GB pushes the cliff out.

```bash
# GMI example:
datasets/isb1/scripts/gmi_kv_sweep.sh \
  --gpu-type h100 --cloud gmi \
  --model dsr1 --engine vllm \
  --context-band 131k --workload code \
  --users "2,4,8,16,32,64" \
  --offload-modes "on,off,noprefix" \
  --benchmark-duration-s 1800

# AWS example — identical flags, only --cloud changes:
datasets/isb1/scripts/gmi_kv_sweep.sh \
  --gpu-type h100 --cloud aws \
  --model dsr1 --engine vllm \
  --context-band 131k --workload code \
  --users "2,4,8,16,32,64" \
  --offload-modes "on,off,noprefix" \
  --benchmark-duration-s 1800
```

**What to look for:**
- Offload cliff: at what concurrency does `--offload-mode on` start helping?
- Prefix cache hit rate: does it stay >50% under load?
- Preemption count: how many requests get evicted?
- TTFT degradation: when does p99 TTFT exceed 10s?

### Phase 3: Blackwell KV Stress Sweep (8–18 hours per GPU)

Blackwell standard has ~2.25× (B200) and ~2.4× (GB200) the HBM of H100 —
the cliff comes later and sustained concurrency is higher. Blackwell Ultra
(B300, GB300) pushes HBM to 288 GB per GPU (~3.6× H100), giving another
step in sustained concurrency before offload kicks in.

> **Note:** B300 and GB300 will emit an immediate `No container image
> resolved` error until you export the corresponding `LANEB_IMAGE_*`
> override — no public images are pinned yet for Blackwell Ultra.

```bash
# B200 on GMI:
datasets/isb1/scripts/gmi_kv_sweep.sh \
  --gpu-type b200 --cloud gmi \
  --model qwen3.5 --engine vllm \
  --context-band 131k --workload code \
  --users "2,4,8,16,32,64,128,256" \
  --offload-modes "on,off,noprefix" \
  --benchmark-duration-s 1800

# GB200 on GMI — requires image override (aarch64):
export LANEB_IMAGE_GMI_GB200_VLLM=...
datasets/isb1/scripts/gmi_kv_sweep.sh \
  --gpu-type gb200 --cloud gmi \
  --model qwen3.5 --engine vllm \
  --context-band 131k --workload code \
  --users "2,4,8,16,32,64,128,256" \
  --offload-modes "on,off,noprefix" \
  --benchmark-duration-s 1800
```

**What to look for:**
- Does the cliff scale with HBM capacity ratio?
- Does 192GB let prefix caching stay effective longer?
- GB200 NVLink-C2C (Grace↔Blackwell) latency vs PCIe on B200 — visible in
  offload-mode=on transfer timings.

### Phase 4: Long Context Preview (4–8 hours, Blackwell SKUs only)

500K and 1M token traces — all four Blackwell SKUs (B200, B300, GB200,
GB300) have enough memory for Qwen 3.5. GPT-OSS and DSR1 have hard
model-side context limits (131K and 164K respectively) so they are
intentionally absent from 500K/1M runs.

```bash
# 500K Qwen (both B200 and GB200 can run this):
datasets/isb1/scripts/gmi_portable_benchmark.sh \
  --gpu-type b200 --cloud gmi \
  --model qwen3.5 --engine vllm \
  --context-band 500k --workload code

# 1M Qwen (B200/GB200 only; qwen3.5 is the only model with 1M support):
datasets/isb1/scripts/gmi_portable_benchmark.sh \
  --gpu-type gb200 --cloud gmi \
  --model qwen3.5 --engine vllm \
  --context-band 1m --workload code
```

The curated `gmi_full_suite.sh` and `gmi_test_matrix.sh` wrappers gate
the 1M lane to `{b200, b300, gb200, gb300}` — every Blackwell SKU, both
standard and Ultra.

## Curated suites

```bash
# Full matrix across all committed cells for one GPU×cloud:
datasets/isb1/scripts/gmi_full_suite.sh \
  --gpu-type gb200 --cloud gmi \
  --db-path datasets/isb1/results/gb200_gmi_full.db

# Minimal smoke matrix (one run per headline cell):
datasets/isb1/scripts/gmi_test_matrix.sh \
  --gpu-type h200 --cloud aws
```

## Estimated GPU Time

| Phase | GPU | Duration | Cost (est, GMI spot) |
|-------|-----|----------|----------------------|
| 1. Validation | any | 1 hour | ~$3–$8 |
| 2. H100 sweep | H100 | 9 hours | ~$27 |
| 2. H200 sweep | H200 | 9 hours | ~$35 |
| 3. B200 sweep | B200 | 18 hours | ~$90 |
| 3. B300 sweep *(Ultra)* | B300 | 18 hours | *(preliminary, pending image + cloud GA)* |
| 3. GB200 sweep | GB200 | 18 hours | ~$150 |
| 3. GB300 sweep *(Ultra)* | GB300 | 18 hours | *(preliminary, pending image + cloud GA)* |
| 4. Long context (Qwen 500K+1M) | Blackwell ×4 | 4–8 hours per SKU | ~$20–$60 per SKU |

AWS on-demand runs ~1.5–2× higher; use spot capacity blocks when available.

## Result Collection

Each run writes to `datasets/isb1/results/gmi/laneb-<cloud>-<gpu>-...`:

```
datasets/isb1/results/gmi/
├── laneb-gmi-h100-dsr1-vllm-code-131k-.../
│   ├── agg_*.json               # processed ISB1 summary
│   ├── laneb-...json            # raw replay result
│   ├── server.log               # engine stdout/stderr
│   ├── manifest.json            # execution_mode=direct, cloud, gpu_type, gpu_arch, ...
│   ├── kv_metrics.csv           # (if metrics_collector is enabled)
│   └── kv_stress_campaign_metadata.json  # only for --benchmark-type isb1_kv_stress
└── laneb-aws-gb200-qwen3.5-.../
    └── ...
```

Aggregate and query:

```bash
# Everything lands in a single SQLite DB by default
python3 datasets/isb1/scripts/isb1_results_db.py summary

# Compare cloud cost/perf:
python3 datasets/isb1/scripts/isb1_results_db.py query \
  --group-by cloud,gpu_type,context_band

# Pareto (once enough rows land):
python3 datasets/isb1/scripts/plot_pareto.py \
  --summary datasets/isb1/results/isb1_results.db \
  --output  datasets/isb1/results/pareto_frontier.png
```

## What Success Looks Like

After all phases, we have, per `(cloud, gpu_type)`:

1. **Pareto frontier chart:** throughput vs p99 TTFT across H100/H200/B200/GB200
   on both GMI and AWS.
2. **Offload cliff identification:** exact concurrency where
   `--offload-mode on` starts helping, per GPU/cloud.
3. **Prefix cache benefit:** measured hit rate under realistic multi-turn load.
4. **HBM scaling evidence:** does 2.25–2.4× more HBM give 2.25–2.4× more
   concurrency at the cliff?
5. **Long-context feasibility:** can B200/GB200 serve 500K/1M Qwen at all?
6. **Cloud-vs-cloud parity:** identical flags on GMI vs AWS, same GPU SKU,
   should produce comparable TTFT/throughput within a few percent; larger
   gaps point to networking/host differences worth investigating.

Results feed the Pareto summaries and capacity-cliff annotations consumed by
the ISB1 replay analyzers (`datasets/isb1/scripts/gmi_analyze_sweep.py`,
`datasets/isb1/scripts/plot_pareto.py`).

## Lane A relationship

Lane A = SemiAnalysis's Slurm-backed workflow path (srt-slurm, pinned
SHAs, GitHub Actions dispatch). Nothing in this runbook touches Slurm.
Every row produced here is tagged `execution_mode=direct` in the manifest,
which makes them easy to filter out when Lane A validation eventually
consumes Lane B results as a baseline.
