"""Tests for trace entry and result extension Pydantic models."""
import pytest
from pydantic import ValidationError

from schema import TraceEntry, KVResultExtension, TraceFile, Scenario, ResidencyTier


class TestTraceEntry:

    def test_minimal_valid_entry(self, sample_trace_entry):
        entry = TraceEntry(**sample_trace_entry)
        assert entry.trace_id == "test-001"
        assert entry.input_tokens_total == 1024
        assert entry.shared_prefix_tokens == 0  # default
        assert entry.residency_tier_hint == ResidencyTier.GPU  # default

    def test_full_valid_entry(self, sample_prefix_reuse_entry):
        entry = TraceEntry(**sample_prefix_reuse_entry)
        assert entry.shared_prefix_tokens == 3072
        assert entry.expected_prefix_hit_rate == 0.75
        assert entry.system_prefix_id == "sys_prefix_001"

    def test_input_tokens_must_be_positive(self, sample_trace_entry):
        sample_trace_entry["input_tokens_total"] = 0
        with pytest.raises(ValidationError, match="input_tokens_total"):
            TraceEntry(**sample_trace_entry)

    def test_output_tokens_must_be_positive(self, sample_trace_entry):
        sample_trace_entry["expected_output_tokens"] = 0
        with pytest.raises(ValidationError, match="expected_output_tokens"):
            TraceEntry(**sample_trace_entry)

    def test_shared_prefix_cannot_exceed_input(self, sample_trace_entry):
        sample_trace_entry["shared_prefix_tokens"] = 2000
        with pytest.raises(ValidationError, match="cannot exceed input_tokens_total"):
            TraceEntry(**sample_trace_entry)

    def test_shared_plus_unique_cannot_exceed_input(self, sample_trace_entry):
        sample_trace_entry["shared_prefix_tokens"] = 800
        sample_trace_entry["unique_suffix_tokens"] = 800
        # 800 + 800 = 1600 > 1024
        with pytest.raises(ValidationError, match="cannot exceed input_tokens_total"):
            TraceEntry(**sample_trace_entry)

    def test_parent_turn_id_must_be_less_than_turn_id(self):
        with pytest.raises(ValidationError, match="parent_turn_id"):
            TraceEntry(
                trace_id="t1", session_id="s1", turn_id=2,
                scenario="prefix_reuse_sweep",
                parent_turn_id=3,  # >= turn_id
                input_tokens_total=100, expected_output_tokens=50,
            )

    def test_prefix_hit_rate_bounds(self, sample_trace_entry):
        sample_trace_entry["expected_prefix_hit_rate"] = 1.5
        with pytest.raises(ValidationError):
            TraceEntry(**sample_trace_entry)

        sample_trace_entry["expected_prefix_hit_rate"] = -0.1
        with pytest.raises(ValidationError):
            TraceEntry(**sample_trace_entry)

    def test_extra_fields_forbidden(self, sample_trace_entry):
        sample_trace_entry["bogus_field"] = "nope"
        with pytest.raises(ValidationError, match="extra"):
            TraceEntry(**sample_trace_entry)

    def test_scenario_enum_values(self):
        assert Scenario.PREFIX_REUSE_SWEEP.value == "prefix_reuse_sweep"
        assert Scenario.KV_OFFLOAD_CLIFF.value == "kv_offload_cliff"
        assert Scenario.LONG_REASONING_GROWTH.value == "long_reasoning_growth"

    def test_residency_tier_enum_values(self):
        assert ResidencyTier.GPU.value == "gpu"
        assert ResidencyTier.CPU.value == "cpu"
        assert ResidencyTier.DISK.value == "disk"
        assert ResidencyTier.UNKNOWN.value == "unknown"

    def test_quality_eval_requires_task_and_answer(self, sample_trace_entry):
        sample_trace_entry["requires_quality_eval"] = True
        with pytest.raises(ValidationError, match="quality_task"):
            TraceEntry(**sample_trace_entry)

    def test_quality_eval_with_task_and_answer(self, sample_trace_entry):
        sample_trace_entry["requires_quality_eval"] = True
        sample_trace_entry["quality_task"] = "gsm8k"
        sample_trace_entry["gold_answer"] = "42"
        entry = TraceEntry(**sample_trace_entry)
        assert entry.requires_quality_eval is True


class TestKVResultExtension:

    def test_all_fields_optional(self):
        result = KVResultExtension()
        assert result.kv_policy_name is None
        assert result.effective_bits_per_element is None

    def test_partial_fields(self):
        result = KVResultExtension(
            kv_policy_name="full_bf16",
            effective_bits_per_element=16.0,
            compression_ratio_vs_bf16=1.0,
        )
        assert result.kv_policy_name == "full_bf16"
        assert result.compression_ratio_vs_bf16 == 1.0

    def test_merge_with_existing_result(self):
        existing = {"hw": "b200", "conc": 64, "model": "deepseek-ai/DeepSeek-R1"}
        kv_fields = {"kv_policy_name": "full_bf16", "effective_bits_per_element": 16.0}
        merged = KVResultExtension(**(existing | kv_fields))
        assert merged.kv_policy_name == "full_bf16"
        # extra fields pass through since extra='allow'
        assert merged.model_extra["hw"] == "b200"

    def test_latency_fields(self):
        result = KVResultExtension(
            ttft_ms_p50=15.5, ttft_ms_p95=25.0,
            itl_ms_p50=3.2, itl_ms_p95=5.1, itl_ms_p99=8.3,
            decode_tok_s=150.0,
        )
        assert result.decode_tok_s == 150.0


class TestTraceFile:

    def test_empty_entries_rejected(self):
        with pytest.raises(ValidationError):
            TraceFile(entries=[])

    def test_duplicate_trace_ids_rejected(self, sample_trace_entry):
        entry1 = TraceEntry(**sample_trace_entry)
        entry2 = TraceEntry(**sample_trace_entry)  # same trace_id
        with pytest.raises(ValidationError, match="Duplicate trace_ids"):
            TraceFile(entries=[entry1, entry2])

    def test_valid_trace_file(self, sample_trace_entry):
        e1 = TraceEntry(**sample_trace_entry)
        e2_data = {**sample_trace_entry, "trace_id": "test-002"}
        e2 = TraceEntry(**e2_data)
        tf = TraceFile(entries=[e1, e2])
        assert len(tf.entries) == 2
