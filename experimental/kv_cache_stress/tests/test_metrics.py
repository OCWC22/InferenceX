"""Tests for metric helper functions."""
import pytest

from metrics import (
    full_bf16_kv_bytes,
    effective_bits_per_element,
    compression_ratio_vs_bf16,
    quality_delta_abs,
    quality_delta_rel,
    num_kv_elements,
)


class TestFullBF16KVBytes:

    def test_basic_calculation(self):
        # 1 batch, 1024 tokens, 32 layers, 8 heads, 128 dim
        # = 1 * 1024 * 32 * 2 * 8 * 128 * 2 = 134,217,728
        result = full_bf16_kv_bytes(
            batch_size=1, sequence_length=1024,
            num_layers=32, num_kv_heads=8, head_dim=128,
        )
        assert result == 1 * 1024 * 32 * 2 * 8 * 128 * 2

    def test_llama_70b_config(self):
        # Llama-70B: 80 layers, 8 GQA heads, 128 dim
        result = full_bf16_kv_bytes(
            batch_size=1, sequence_length=4096,
            num_layers=80, num_kv_heads=8, head_dim=128,
        )
        expected = 1 * 4096 * 80 * 2 * 8 * 128 * 2
        assert result == expected

    def test_batch_size_scales_linearly(self):
        base = full_bf16_kv_bytes(
            batch_size=1, sequence_length=1024,
            num_layers=32, num_kv_heads=8, head_dim=128,
        )
        doubled = full_bf16_kv_bytes(
            batch_size=2, sequence_length=1024,
            num_layers=32, num_kv_heads=8, head_dim=128,
        )
        assert doubled == 2 * base


class TestEffectiveBitsPerElement:

    def test_bf16_is_16_bits(self):
        # 100 elements, each 2 bytes = 200 bytes -> 16 bits per element
        result = effective_bits_per_element(
            compressed_payload_bytes=200, num_total_elements=100,
        )
        assert result == 16.0

    def test_int8_is_8_bits(self):
        result = effective_bits_per_element(
            compressed_payload_bytes=100, num_total_elements=100,
        )
        assert result == 8.0

    def test_metadata_overhead_increases_effective_bits(self):
        # 100 bytes payload + 20 bytes metadata for 100 elements
        result = effective_bits_per_element(
            compressed_payload_bytes=100,
            metadata_bytes=20,
            num_total_elements=100,
        )
        assert result == pytest.approx(9.6)

    def test_all_overhead_types(self):
        result = effective_bits_per_element(
            compressed_payload_bytes=100,
            metadata_bytes=10,
            index_bytes=5,
            codebook_bytes=3,
            transform_state_bytes=2,
            num_total_elements=100,
        )
        assert result == pytest.approx((120 * 8) / 100)

    def test_zero_elements_raises(self):
        with pytest.raises(ValueError, match="num_total_elements"):
            effective_bits_per_element(
                compressed_payload_bytes=100, num_total_elements=0,
            )


class TestCompressionRatio:

    def test_no_compression_is_1(self):
        result = compression_ratio_vs_bf16(
            full_bf16_bytes=1000, total_compressed_kv_bytes=1000,
        )
        assert result == 1.0

    def test_half_size_is_2(self):
        result = compression_ratio_vs_bf16(
            full_bf16_bytes=1000, total_compressed_kv_bytes=500,
        )
        assert result == 2.0

    def test_20x_compression(self):
        result = compression_ratio_vs_bf16(
            full_bf16_bytes=20000, total_compressed_kv_bytes=1000,
        )
        assert result == 20.0

    def test_zero_compressed_raises(self):
        with pytest.raises(ValueError, match="total_compressed_kv_bytes"):
            compression_ratio_vs_bf16(
                full_bf16_bytes=1000, total_compressed_kv_bytes=0,
            )


class TestQualityDelta:

    def test_no_change(self):
        assert quality_delta_abs(quality_score=0.85, baseline_quality_score=0.85) == 0.0
        assert quality_delta_rel(quality_score=0.85, baseline_quality_score=0.85) == 0.0

    def test_improvement(self):
        assert quality_delta_abs(quality_score=0.90, baseline_quality_score=0.85) == pytest.approx(0.05)
        assert quality_delta_rel(quality_score=0.90, baseline_quality_score=0.85) == pytest.approx(0.05 / 0.85)

    def test_degradation(self):
        delta = quality_delta_abs(quality_score=0.80, baseline_quality_score=0.85)
        assert delta == pytest.approx(-0.05)

    def test_rel_zero_baseline_raises(self):
        with pytest.raises(ValueError, match="baseline_quality_score"):
            quality_delta_rel(quality_score=0.5, baseline_quality_score=0.0)


class TestNumKVElements:

    def test_basic(self):
        result = num_kv_elements(
            sequence_length=1024, num_layers=32, num_kv_heads=8, head_dim=128,
        )
        assert result == 1024 * 32 * 2 * 8 * 128
