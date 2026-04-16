"""Metric helper functions for KV-cache benchmarks.

These implement the canonical formulas for comparing KV cache policies.
Compression ratio alone is NOT sufficient -- effective bits, quality delta,
and latency must all be reported together.
"""
from __future__ import annotations


def full_bf16_kv_bytes(
    *,
    batch_size: int,
    sequence_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Full BF16 KV cache size in bytes.

    Formula:
        batch_size * sequence_length * num_layers * 2 (K+V)
        * num_kv_heads * head_dim * 2 (bytes per bf16)
    """
    return batch_size * sequence_length * num_layers * 2 * num_kv_heads * head_dim * 2


def effective_bits_per_element(
    *,
    compressed_payload_bytes: int,
    metadata_bytes: int = 0,
    index_bytes: int = 0,
    codebook_bytes: int = 0,
    transform_state_bytes: int = 0,
    num_total_elements: int,
) -> float:
    """Effective bits per KV element including all overhead.

    This is the true cost metric. Compression schemes that report low
    payload bits but have large metadata/codebook overhead will show
    higher effective bits here.

    Args:
        compressed_payload_bytes: Core compressed data size.
        metadata_bytes: Per-layer/per-head metadata (scales, zeros, etc).
        index_bytes: Index structures (for sparse/selected entries).
        codebook_bytes: Codebook/dictionary storage.
        transform_state_bytes: PCA/decorrelation state (e.g. KVTC).
        num_total_elements: Total scalar KV elements (tokens * layers * 2 * heads * dim).

    Returns:
        Bits per element. Full BF16 = 16.0, INT8 = 8.0, etc.
    """
    if num_total_elements <= 0:
        raise ValueError("num_total_elements must be > 0")
    total_bytes = (
        compressed_payload_bytes
        + metadata_bytes
        + index_bytes
        + codebook_bytes
        + transform_state_bytes
    )
    return (total_bytes * 8) / num_total_elements


def compression_ratio_vs_bf16(
    *,
    full_bf16_bytes: int,
    total_compressed_kv_bytes: int,
) -> float:
    """Compression ratio relative to full BF16 baseline.

    Returns:
        Ratio >= 1.0 where 1.0 = no compression, 2.0 = 50% size reduction.
    """
    if total_compressed_kv_bytes <= 0:
        raise ValueError("total_compressed_kv_bytes must be > 0")
    return full_bf16_bytes / total_compressed_kv_bytes


def quality_delta_abs(*, quality_score: float, baseline_quality_score: float) -> float:
    """Absolute quality difference: score - baseline.

    Positive means the policy improved over baseline (rare for compression).
    Negative means degradation.
    """
    return quality_score - baseline_quality_score


def quality_delta_rel(*, quality_score: float, baseline_quality_score: float) -> float:
    """Relative quality difference: (score - baseline) / baseline.

    Returns a fraction. -0.01 means 1% degradation vs baseline.
    """
    if baseline_quality_score == 0.0:
        raise ValueError("baseline_quality_score must be non-zero for relative delta")
    return (quality_score - baseline_quality_score) / baseline_quality_score


def num_kv_elements(
    *,
    sequence_length: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Total number of scalar KV elements (for effective_bits_per_element denominator).

    Formula: sequence_length * num_layers * 2 (K+V) * num_kv_heads * head_dim
    """
    return sequence_length * num_layers * 2 * num_kv_heads * head_dim
