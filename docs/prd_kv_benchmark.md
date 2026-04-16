# PRD: KV Cache Benchmark Extension for InferenceX

## Problem statement

InferenceX benchmarks LLM inference engines using random single-turn inputs with prefix caching disabled. This methodology cannot measure KV cache behavior that matters in production:

- **Prefix reuse**: Multi-turn conversations and shared system prompts create high prefix hit rates. Random data has ~0% hit rate.
- **Memory pressure**: Real workloads cause KV caches to be offloaded to CPU/disk and reactivated. Random single-turn traffic never exercises this path.
- **Compression quality tradeoffs**: Methods like TurboQuant (3-bit structured quantization), TriAttention (selective KV retention), KVTC (transform-coded storage), and AQUA-KV (adaptive quantization) each trade memory for quality differently. No existing benchmark compares them fairly.
- **Long-context quality cliffs**: Aggressive KV compression degrades quality at long context lengths. Without quality evaluation, compression ratios are meaningless.

## Goals

1. Define a deterministic JSONL trace schema for KV cache stress workloads
2. Provide a validator, synthetic generator, and smoke traces for three core scenarios
3. Define an additive result schema extension for KV-specific metrics
4. Define a pluggable KVPolicy interface for future compression policy integration
5. Implement metric helpers (effective bits, compression ratio, quality delta)
6. Ship with 100% passing tests and clear documentation

## Non-goals

- Implement actual KV compression algorithms (phase 2+)
- Require custom CUDA/Triton kernels
- Replace or modify the existing InferenceX benchmark pipeline
- Benchmark against live inference engines in phase 1
- Require private datasets or cloud-scale CI

## User stories

1. **Benchmark maintainer**: "I want to add KV cache compression methods to InferenceX comparisons without rewriting the benchmark framework."
2. **KV compression researcher**: "I want to evaluate my method against baselines using standardized workloads and metrics."
3. **ML infra engineer**: "I want to understand the memory/quality/latency tradeoff of different KV policies for my deployment."
4. **CI system**: "I want to run smoke-level KV cache benchmarks in <1 second to gate merges."

## Success metrics

- All smoke traces validate against schema
- 100 tests pass in <1 second
- A new KV policy can be added by implementing a Protocol with ~6 methods
- Result schema extension can be merged into existing InferenceX results without breaking anything
- Phase 2 PR can add a trace runner without modifying phase 1 code

## Scope — Phase 1

| In scope | Out of scope |
|----------|--------------|
| Trace schema (Pydantic) | Trace runner / engine integration |
| Validator CLI | Live benchmarking |
| Synthetic generator (3 scenarios) | Real-world trace datasets |
| Result schema extension | Dashboard / visualization |
| KVPolicy Protocol + FullBF16 baseline | TurboQuant / TriAttention / KVTC adapters |
| Metric helpers | Custom kernels |
| 100 tests | CI workflow integration |
| Documentation | Multi-node support |

## Dependency graph

```
schema.py  <-- metrics.py
    ^            ^
    |            |
policy.py  validate_trace.py  generate_synthetic_traces.py
    ^                               |
    |                               v
tests/*                        traces/*.jsonl
```

## Review / merge strategy

Phase 1 is entirely additive under `experimental/kv_cache_stress/`. It:
- Creates no new top-level directories (except `docs/`)
- Modifies no existing files
- Has no runtime dependencies beyond pydantic
- Can be reviewed as a single PR or split into smaller PRs per the follow-up plan
