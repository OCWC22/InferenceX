# KV Cache Stress Benchmark Substrate

Experimental foundation for benchmarking realistic KV cache behavior in InferenceX.

## Why this exists

InferenceX currently benchmarks with random single-turn inputs and disables prefix caching (random data has ~0% prefix hit rate). This means the benchmark cannot measure:

- Prefix reuse patterns from real multi-turn conversations
- KV cache offload/reactivation behavior under memory pressure
- Quality degradation from KV compression at long context lengths
- The difference between pruning, quantization, and transform-coded compression

This substrate provides the **trace schema, workload fixtures, metric definitions, and policy interface** needed to benchmark these behaviors fairly across methods like TurboQuant, TriAttention, KVTC, AQUA-KV, SnapKV, H2O, and SpeCache.

## Status

**Phase 1 — substrate only.** No actual inference engine integration yet. The trace format, schemas, and policy interface are stable enough for review and iteration.

## Quick start

```bash
# Validate smoke traces
python validate_trace.py --strict traces/smoke_prefix_reuse.jsonl

# Generate new traces
python generate_synthetic_traces.py --scenario prefix_reuse_sweep --seed 42 --output traces/custom.jsonl

# Generate all scenarios
python generate_synthetic_traces.py --all --seed 42 --output-dir traces/

# Run tests
python -m pytest tests/ -v
```

## Directory layout

```
kv_cache_stress/
  schema.py                     # Pydantic models: TraceEntry, KVResultExtension
  metrics.py                    # Metric formulas: effective bits, compression ratio, quality delta
  policy.py                     # KVPolicy protocol + FullBF16KVPolicy + stubs
  validate_trace.py             # CLI: validate JSONL trace files
  generate_synthetic_traces.py  # CLI: generate deterministic synthetic traces
  traces/
    smoke_prefix_reuse.jsonl         # 20 entries: prefix hit-rate sweep
    smoke_offload_cliff.jsonl        # 15 entries: GPU/CPU/disk offload stress
    smoke_long_reasoning_growth.jsonl # 10 entries: long reasoning decode stress
  tests/
    test_schema.py              # TraceEntry and KVResultExtension validation
    test_traces.py              # Smoke trace loading + end-to-end data flow
    test_generate.py            # Generation determinism + schema compliance
    test_metrics.py             # Metric formula correctness
    test_policy.py              # KVPolicy protocol + baseline behavior
```

## Scenarios

### prefix_reuse_sweep
Controlled prefix hit-rate stress. Tests how KV cache policies handle shared system prefixes across sessions with varying reuse ratios (0% to 95%).

### kv_offload_cliff
Forces GPU KV pressure by simulating sessions that go idle for varying durations (100ms to 600s), causing cache migration from GPU to CPU to disk. Tests reactivation latency.

### long_reasoning_growth
Long-decode reasoning stress where context grows from 1K to 128K tokens across turns. All entries require quality evaluation to measure accuracy degradation under compression.

## Trace schema

Each JSONL line is a `TraceEntry` with fields for identity, token geometry, timing, residency, and optional quality evaluation. See `schema.py` for the full model and validation rules.

Key validation rules:
- `input_tokens_total > 0`
- `expected_output_tokens > 0`
- `shared_prefix_tokens <= input_tokens_total`
- `0.0 <= expected_prefix_hit_rate <= 1.0`
- If `requires_quality_eval`, then `quality_task` and `gold_answer` must be set

## KV policy families

Policies are categorized into families that require different benchmark scenarios:

| Family | Examples | Key metrics |
|--------|----------|-------------|
| Full cache | BF16 baseline | Memory, latency (reference) |
| Quantization | INT8, INT4, TurboQuant, AQUA-KV | Effective bits, quality delta, calibration time |
| Pruning/eviction | SnapKV, H2O, TriAttention | Retained token fraction, reasoning quality |
| Structured compression | KVTC | Compression ratio, transform state overhead |
| Offload/prefetch | SpeCache | Reactivation latency, offload throughput |

## Metric definitions

- **effective_bits_per_element**: Includes ALL overhead (payload + metadata + codebook + index + transform state). Compression ratio alone is insufficient.
- **compression_ratio_vs_bf16**: Full BF16 bytes / total compressed bytes.
- **quality_delta_abs/rel**: Score difference vs full-KV baseline.

See `metrics.py` for implementations and `docs/technical_design_kv_benchmark.md` for rationale.

## Adding a new scenario

1. Add a generator function in `generate_synthetic_traces.py`
2. Register it in the `GENERATORS` dict
3. Generate a smoke trace and commit it to `traces/`
4. Add scenario-specific validation tests in `tests/test_traces.py`

## Adding a new KV policy

1. Implement the `KVPolicy` protocol from `policy.py`
2. Register in `BUILTIN_POLICIES`
3. Add tests in `tests/test_policy.py`
4. Phase 2+: integrate with a trace runner for actual benchmarking
