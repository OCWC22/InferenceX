# Technical Design: KV Cache Benchmark Substrate

## Overview

This document describes the architecture of the KV cache benchmark extension for InferenceX. Phase 1 establishes the substrate: schemas, workload definitions, metric formulas, and a policy interface. Phase 2+ adds engine integration, real policies, and dashboard support.

## Data flow

```
generate_synthetic_traces.py
         |
         v
    traces/*.jsonl          (JSONL workload fixtures)
         |
         v
    validate_trace.py       (schema validation CLI)
         |
         v
    TraceEntry              (Pydantic model per-line)
         |
         v
   [future] trace_runner.py (phase 2: drives inference engine)
         |
         v
    KVPolicy.encode/decode/select/attention
         |
         v
    KVResultExtension       (additive result fields)
         |
         v
   [future] merge with existing process_result.py output
```

## JSONL trace schema

Each line in a trace file is a JSON object validated as a `TraceEntry`. Key design choices:

- **`extra='forbid'`**: Strict validation catches typos and schema drift early. Matches the pattern in `utils/matrix_logic/validation.py`.
- **Scenario enum**: Exactly three scenarios in phase 1 (`prefix_reuse_sweep`, `kv_offload_cliff`, `long_reasoning_growth`). New scenarios are added by extending the `Scenario` enum.
- **`workload_family` field**: Always `"kv_cache_stress"` for this extension. Allows future workload families to coexist.
- **Optional quality fields**: `requires_quality_eval`, `quality_task`, `gold_answer` are only mandatory when quality evaluation is needed. Cross-validated by a `model_validator`.

### Validation rules

| Rule | Enforced by |
|------|------------|
| `input_tokens_total > 0` | `Field(gt=0)` |
| `expected_output_tokens > 0` | `Field(gt=0)` |
| `shared_prefix_tokens <= input_tokens_total` | `model_validator` |
| `shared_prefix_tokens + unique_suffix_tokens <= input_tokens_total` | `model_validator` |
| `0.0 <= expected_prefix_hit_rate <= 1.0` | `Field(ge=0.0, le=1.0)` |
| `parent_turn_id < turn_id` | `model_validator` |
| quality fields required when `requires_quality_eval=True` | `model_validator` |
| No extra fields | `ConfigDict(extra='forbid')` |
| Unique trace_ids across file | `TraceFile.model_validator` |

## Result schema extension

`KVResultExtension` uses `extra='allow'` (intentionally different from `TraceEntry`) so it can be merged with existing InferenceX result dicts:

```python
existing = {"hw": "b200", "conc": 64, ...}
kv_fields = policy.report_stats()
merged = KVResultExtension(**(existing | kv_fields))
```

This mirrors how `process_result.py` merges single-node and multi-node data via `data = data | multi_node_data`.

### Field groups

| Group | Fields | Purpose |
|-------|--------|---------|
| Policy metadata | `kv_policy_name`, `kv_policy_family`, `kv_policy_version`, `requires_calibration`, `calibration_seconds` | Identify what policy was used |
| Compression accounting | `full_bf16_kv_bytes`, `compressed_kv_payload_bytes`, `metadata_bytes`, `codebook_bytes`, `index_bytes`, `effective_bits_per_element`, `compression_ratio_vs_bf16` | True cost including all overhead |
| Pruning | `retained_token_fraction` | What fraction of KV was kept |
| Prefix cache | `prefix_hit_rate_target`, `prefix_hit_rate_observed` | Cache reuse efficiency |
| Offload | `kv_gpu_bytes_peak`, `kv_cpu_bytes_peak`, `offload_read_gbps`, `offload_write_gbps`, `reactivation_ms` | Tiered storage behavior |
| Latency | `ttft_ms_p50/p95`, `itl_ms_p50/p95/p99`, `decode_tok_s` | End-user visible performance |
| Quality | `quality_metric`, `quality_score`, `baseline_quality_score`, `quality_delta_abs`, `quality_delta_rel` | Accuracy degradation |
| Diagnostics | `encode_gbs`, `decode_gbs`, `qk_logit_mse`, `kv_reconstruction_mse` | Internal distortion |

## KVPolicy interface

A `typing.Protocol` with `@runtime_checkable` for duck-type verification:

```python
@runtime_checkable
class KVPolicy(Protocol):
    name: str           # "full_bf16", "turbo_quant_3bit", etc.
    family: str         # "full_cache", "quantization", etc.
    requires_calibration: bool
    supports_fused_attention: bool
    supports_reactivation: bool

    def calibrate(model, loader, *, seed, max_samples) -> dict
    def encode(layer_id, k, v, metadata) -> Any
    def decode(layer_id, compressed_kv, metadata) -> (k, v)
    def select(layer_id, q, kv, metadata) -> Any
    def attention(q, compressed_kv, metadata) -> Any
    def report_stats() -> dict
```

Phase 1 uses `Any` types to avoid mandatory PyTorch dependency. Phase 2 will tighten to `Optional[torch.Tensor]`.

### Policy families

| Family | Phase 1 status | Phase 2+ target |
|--------|---------------|-----------------|
| `full_cache` | FullBF16KVPolicy (implemented) | Engine-native FP8/INT8 if exposed |
| `quantization` | UniformINT8/INT4 stubs | TurboQuant, AQUA-KV adapters |
| `pruning_eviction` | KeepRecentWindow, OracleTopAttention stubs | SnapKV, H2O, TriAttention adapters |
| `structured_compression` | — | KVTC adapter |
| `offload_prefetch` | — | SpeCache adapter |

## Metric formulas

### Why compression ratio alone is not sufficient

A method reporting "20x compression" could have:
- 1-bit payload but 15 bits of metadata per element → effective 16 bits (no real savings)
- 3-bit payload, 0.5 bits metadata → effective 3.5 bits (real savings)
- 2-bit payload, great ratio, but 10% quality degradation → unusable

The benchmark MUST report `effective_bits_per_element` (includes ALL overhead) and quality jointly with compression.

### Formulas

```
full_bf16_kv_bytes = batch * seq_len * layers * 2 * kv_heads * head_dim * 2

effective_bits_per_element = 8 * (payload + metadata + index + codebook + transform_state) / num_elements

compression_ratio_vs_bf16 = full_bf16_kv_bytes / total_compressed_bytes

quality_delta_abs = score - baseline_score
quality_delta_rel = (score - baseline_score) / baseline_score
```

## Runner integration (phase 2)

The trace runner will:
1. Load a JSONL trace file
2. For each `TraceEntry`, construct a request with the specified token geometry
3. Apply the configured `KVPolicy` to compress/select KV
4. Drive the inference engine and collect latency metrics
5. Optionally run quality evaluation
6. Emit `KVResultExtension` records

Integration with `benchmark_serving.py` will be via a new `--trace-file` flag that replaces `sample_random_requests()` with trace-driven request generation.

## Test plan

| Test file | Coverage |
|-----------|----------|
| `test_schema.py` | TraceEntry validation, KVResultExtension merge, TraceFile consistency |
| `test_traces.py` | Smoke trace loading, scenario-specific properties, end-to-end data flow |
| `test_generate.py` | Determinism, schema compliance, custom entry counts |
| `test_metrics.py` | Formula correctness, edge cases, error handling |
| `test_policy.py` | Protocol conformance, baseline behavior, stub behavior, registry |

All 100 tests pass in <0.3 seconds with zero external dependencies beyond pydantic and pytest.

## Known limitations

- No actual inference engine integration in phase 1
- Smoke traces use synthetic/placeholder prompts, not real text
- KVPolicy stubs raise NotImplementedError — real implementations are phase 2+
- No multi-node KV cache scenarios yet
- No CI workflow integration yet
