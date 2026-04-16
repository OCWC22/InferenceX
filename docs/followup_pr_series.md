# Follow-up PR Series: KV Cache Benchmark

Concrete PR sequence for building on the phase 1 substrate.

## PR 1 — Traces + Validator + Docs (this PR)

**Files:**
- `experimental/kv_cache_stress/schema.py`
- `experimental/kv_cache_stress/metrics.py`
- `experimental/kv_cache_stress/policy.py`
- `experimental/kv_cache_stress/validate_trace.py`
- `experimental/kv_cache_stress/generate_synthetic_traces.py`
- `experimental/kv_cache_stress/traces/*.jsonl`
- `experimental/kv_cache_stress/tests/*`
- `experimental/kv_cache_stress/README.md`
- `docs/prd_kv_benchmark.md`
- `docs/technical_design_kv_benchmark.md`
- `docs/followup_pr_series.md`

**Reviewable as-is.** Zero changes to existing code. Entirely additive under `experimental/`.

## PR 2 — Trace Runner

**Goal:** Drive an inference engine using trace files instead of random inputs.

**Changes:**
- `experimental/kv_cache_stress/trace_runner.py` — loads JSONL, converts entries to `RequestFuncInput`, drives `benchmark_serving.py` backend
- Add `--trace-file` flag to `benchmark_serving.py` (opt-in, no behavior change when absent)
- Emit `KVResultExtension` fields in output JSON
- Integration test: run smoke trace against a local vLLM/SGLang instance

**Depends on:** PR 1

## PR 3 — KVPolicy Baselines

**Goal:** Implement real baseline policies with tensor operations.

**Changes:**
- `FullBF16KVPolicy`: real PyTorch encode/decode/attention
- `UniformINT8KVPolicy`: per-tensor INT8 cast quantization
- `UniformINT4KVPolicy`: group-wise INT4 quantization (requires calibration)
- `KeepRecentWindowKVPolicy`: sliding window eviction
- Metric collection hooks in trace runner

**Depends on:** PR 2

## PR 4 — Long-Context Quality Workloads

**Goal:** Real quality evaluation for long-context KV benchmark entries.

**Changes:**
- Integration with `utils/evals/` (lm-eval harness)
- Real prompt text for `long_reasoning_growth` traces (GSM8K, GPQA, AIME25)
- Quality scorer callbacks in trace runner
- Quality delta reporting in results

**Depends on:** PR 2, PR 3

## PR 5 — TriAttention Adapter

**Goal:** Benchmark TriAttention selective KV retention.

**Changes:**
- `TriAttentionKVPolicy` implementation
- Pre-RoPE attention score computation
- Retained token fraction tracking
- Long-reasoning decode workload results

**Key metrics:** retained_token_fraction, reasoning quality at 32K generation, throughput gain

**Depends on:** PR 3, PR 4

## PR 6 — TurboQuant Adapter

**Goal:** Benchmark TurboQuant structured KV quantization.

**Changes:**
- `TurboQuantKVPolicy` implementation (3-bit and 4-bit modes)
- Effective bits accounting including metadata
- Attention-logit distortion diagnostics (`qk_logit_mse`)
- Long-context quality cliff tracking

**Key metrics:** effective_bits_per_element, compression_ratio, quality_delta, encode/decode throughput

**Depends on:** PR 3, PR 4

## PR 7 — KVTC Reactivation/Offload

**Goal:** Benchmark KVTC transform-coded storage and reuse.

**Changes:**
- `KVTCPolicy` implementation (PCA + quantize + entropy coding)
- Offload/restore benchmarks (GPU → CPU → disk → GPU)
- Reusable prefix cache benchmarks
- `kv_offload_cliff` scenario integration

**Key metrics:** compression_ratio (up to 20x), reactivation_ms, offload throughput, prefix reuse efficiency

**Depends on:** PR 3

## PR 8 — Dashboard / Analysis

**Goal:** Visualization and comparison tools.

**Changes:**
- Result aggregation across policies and scenarios
- Policy family comparison tables
- Quality-vs-compression Pareto frontiers
- Latency-vs-memory scatter plots
- Integration with InferenceX website (if applicable)

**Depends on:** PR 3+

## Dependency graph

```
PR1 (substrate)
 |
 v
PR2 (trace runner)
 |
 +--> PR3 (baselines)
 |     |
 |     +--> PR4 (quality workloads)
 |     |     |
 |     |     +--> PR5 (TriAttention)
 |     |     +--> PR6 (TurboQuant)
 |     |
 |     +--> PR7 (KVTC offload)
 |
 +--> PR8 (dashboard) [after PR3+]
```
