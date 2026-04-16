"""Pydantic models for KV-cache stress trace entries and result extensions.

Trace entries describe workload scenarios for benchmarking KV cache policies.
Result extensions add KV-specific metrics to existing InferenceX result dicts.

Follows the Pydantic V2 patterns from utils/matrix_logic/validation.py:
  - ConfigDict(extra='forbid') for strict input validation
  - model_validator(mode='after') for cross-field checks
  - Field() constraints for range validation
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Scenario(str, Enum):
    """Supported KV-stress benchmark scenarios."""
    PREFIX_REUSE_SWEEP = "prefix_reuse_sweep"
    KV_OFFLOAD_CLIFF = "kv_offload_cliff"
    LONG_REASONING_GROWTH = "long_reasoning_growth"


class ResidencyTier(str, Enum):
    """Where the KV cache is expected to reside."""
    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"
    UNKNOWN = "unknown"


class KVPolicyFamily(str, Enum):
    """Families of KV cache management policies."""
    FULL_CACHE = "full_cache"
    QUANTIZATION = "quantization"
    PRUNING_EVICTION = "pruning_eviction"
    STRUCTURED_COMPRESSION = "structured_compression"
    OFFLOAD_PREFETCH = "offload_prefetch"


# ---------------------------------------------------------------------------
# Trace entry schema
# ---------------------------------------------------------------------------

class TraceEntry(BaseModel):
    """A single entry in a KV-stress JSONL trace file.

    Each entry represents one request/turn in a benchmark scenario.
    Validation rules:
      - input_tokens_total > 0
      - expected_output_tokens > 0
      - 0 <= shared_prefix_tokens <= input_tokens_total
      - shared_prefix_tokens + unique_suffix_tokens <= input_tokens_total
      - 0.0 <= expected_prefix_hit_rate <= 1.0
      - if requires_quality_eval: quality_task and gold_answer must be set
      - parent_turn_id < turn_id (if set)
    """
    model_config = ConfigDict(extra='forbid')

    # Identity
    trace_id: str
    session_id: str
    turn_id: int = Field(ge=0)
    workload_family: str = Field(default="kv_cache_stress")
    scenario: Scenario

    # Content
    prompt: str = Field(default="")
    system_prefix_id: Optional[str] = None
    parent_turn_id: Optional[int] = Field(default=None, ge=0)

    # Token geometry
    input_tokens_total: int = Field(gt=0)
    shared_prefix_tokens: int = Field(ge=0, default=0)
    unique_suffix_tokens: int = Field(ge=0, default=0)
    expected_output_tokens: int = Field(gt=0)
    expected_prefix_hit_rate: float = Field(ge=0.0, le=1.0, default=0.0)

    # Timing / scheduling
    idle_ms_since_last_turn: float = Field(ge=0.0, default=0.0)
    arrival_time_ms: float = Field(ge=0.0, default=0.0)
    residency_tier_hint: ResidencyTier = Field(default=ResidencyTier.GPU)

    # Quality evaluation
    requires_quality_eval: bool = Field(default=False)
    quality_task: Optional[str] = None
    gold_answer: Optional[str] = None

    # Metadata
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_cross_field(self) -> TraceEntry:
        if self.shared_prefix_tokens > self.input_tokens_total:
            raise ValueError(
                f"shared_prefix_tokens ({self.shared_prefix_tokens}) "
                f"cannot exceed input_tokens_total ({self.input_tokens_total})"
            )
        total_specified = self.shared_prefix_tokens + self.unique_suffix_tokens
        if total_specified > self.input_tokens_total:
            raise ValueError(
                f"shared_prefix_tokens + unique_suffix_tokens ({total_specified}) "
                f"cannot exceed input_tokens_total ({self.input_tokens_total})"
            )
        if self.parent_turn_id is not None and self.parent_turn_id >= self.turn_id:
            raise ValueError(
                f"parent_turn_id ({self.parent_turn_id}) must be < turn_id ({self.turn_id})"
            )
        if self.requires_quality_eval:
            if not self.quality_task:
                raise ValueError("quality_task is required when requires_quality_eval is true")
            if not self.gold_answer:
                raise ValueError("gold_answer is required when requires_quality_eval is true")
        return self


class TraceFile(BaseModel):
    """Validated collection of trace entries (entire JSONL file)."""
    model_config = ConfigDict(extra='forbid')

    entries: list[TraceEntry] = Field(min_length=1)

    @model_validator(mode='after')
    def _validate_consistency(self) -> TraceFile:
        trace_ids = [e.trace_id for e in self.entries]
        if len(trace_ids) != len(set(trace_ids)):
            dupes = [tid for tid in trace_ids if trace_ids.count(tid) > 1]
            raise ValueError(f"Duplicate trace_ids: {sorted(set(dupes))}")
        return self


# ---------------------------------------------------------------------------
# Result schema extension
# ---------------------------------------------------------------------------

class KVResultExtension(BaseModel):
    """Additive result fields for KV-cache benchmarks.

    All fields are Optional so this can be merged into existing InferenceX
    result dicts without breaking anything. Uses extra='allow' so unknown
    fields from the existing result pass through during merge.

    Usage:
        existing_result = {"hw": "b200", "conc": 64, ...}
        kv_fields = {"kv_policy_name": "full_bf16", "kv_bits_per_element": 16.0}
        merged = KVResultExtension(**(existing_result | kv_fields))
    """
    model_config = ConfigDict(extra='allow')

    # KV policy metadata
    kv_policy_name: Optional[str] = None
    kv_policy_family: Optional[KVPolicyFamily] = None
    kv_policy_version: Optional[str] = None
    requires_calibration: Optional[bool] = None
    calibration_seconds: Optional[float] = None

    # Compression accounting
    full_bf16_kv_bytes: Optional[int] = None
    compressed_kv_payload_bytes: Optional[int] = None
    metadata_bytes: Optional[int] = None
    codebook_bytes: Optional[int] = None
    index_bytes: Optional[int] = None
    effective_bits_per_element: Optional[float] = None
    compression_ratio_vs_bf16: Optional[float] = None

    # Pruning / selection
    retained_token_fraction: Optional[float] = None

    # Prefix cache
    prefix_hit_rate_target: Optional[float] = None
    prefix_hit_rate_observed: Optional[float] = None

    # Offload / reactivation
    kv_gpu_bytes_peak: Optional[int] = None
    kv_cpu_bytes_peak: Optional[int] = None
    offload_read_gbps: Optional[float] = None
    offload_write_gbps: Optional[float] = None
    reactivation_ms: Optional[float] = None

    # Latency (these extend existing TTFT/ITL fields)
    ttft_ms_p50: Optional[float] = None
    ttft_ms_p95: Optional[float] = None
    itl_ms_p50: Optional[float] = None
    itl_ms_p95: Optional[float] = None
    itl_ms_p99: Optional[float] = None
    decode_tok_s: Optional[float] = None

    # Quality
    quality_metric: Optional[str] = None
    quality_score: Optional[float] = None
    baseline_quality_score: Optional[float] = None
    quality_delta_abs: Optional[float] = None
    quality_delta_rel: Optional[float] = None

    # Diagnostic distortion metrics
    encode_gbs: Optional[float] = None
    decode_gbs: Optional[float] = None
    qk_logit_mse: Optional[float] = None
    kv_reconstruction_mse: Optional[float] = None
