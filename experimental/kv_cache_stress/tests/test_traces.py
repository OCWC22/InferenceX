"""Tests for loading and validating the committed smoke trace files."""
import json
from pathlib import Path

import pytest

from schema import TraceEntry, TraceFile, KVResultExtension
from policy import FullBF16KVPolicy


class TestSmokeTraces:
    """Validate that committed smoke traces load and pass schema validation."""

    @pytest.fixture(params=[
        "smoke_prefix_reuse.jsonl",
        "smoke_offload_cliff.jsonl",
        "smoke_long_reasoning_growth.jsonl",
    ])
    def trace_path(self, traces_dir, request):
        return traces_dir / request.param

    def _load_entries(self, path: Path) -> list[TraceEntry]:
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(TraceEntry(**json.loads(line)))
        return entries

    def test_trace_loads_and_validates(self, trace_path):
        entries = self._load_entries(trace_path)
        assert len(entries) > 0

    def test_trace_passes_strict_validation(self, trace_path):
        entries = self._load_entries(trace_path)
        TraceFile(entries=entries)

    def test_all_entries_have_kv_cache_stress_family(self, trace_path):
        entries = self._load_entries(trace_path)
        for entry in entries:
            assert entry.workload_family == "kv_cache_stress"


class TestPrefixReuseTrace:

    def test_has_shared_prefixes(self, traces_dir):
        entries = []
        with open(traces_dir / "smoke_prefix_reuse.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        prefix_ids = {e.system_prefix_id for e in entries if e.system_prefix_id}
        assert len(prefix_ids) >= 2, "Should have at least 2 distinct system prefix IDs"

    def test_has_varying_hit_rates(self, traces_dir):
        entries = []
        with open(traces_dir / "smoke_prefix_reuse.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        rates = {e.expected_prefix_hit_rate for e in entries}
        assert len(rates) >= 3, "Should have at least 3 distinct hit rates"


class TestOffloadCliffTrace:

    def test_has_increasing_idle_times(self, traces_dir):
        entries = []
        with open(traces_dir / "smoke_offload_cliff.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        idle_times = sorted({e.idle_ms_since_last_turn for e in entries})
        assert idle_times[0] == 0.0
        assert idle_times[-1] >= 60000.0, "Should have at least one entry with long idle"

    def test_has_multiple_residency_tiers(self, traces_dir):
        entries = []
        with open(traces_dir / "smoke_offload_cliff.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        tiers = {e.residency_tier_hint for e in entries}
        assert len(tiers) >= 2, "Should exercise multiple residency tiers"


class TestLongReasoningTrace:

    def test_has_growing_context(self, traces_dir):
        entries = []
        with open(traces_dir / "smoke_long_reasoning_growth.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        ctx_lengths = [e.input_tokens_total for e in entries]
        assert ctx_lengths[-1] > ctx_lengths[0], "Context should grow across entries"

    def test_all_require_quality_eval(self, traces_dir):
        entries = []
        with open(traces_dir / "smoke_long_reasoning_growth.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        for entry in entries:
            assert entry.requires_quality_eval is True
            assert entry.quality_task is not None
            assert entry.gold_answer is not None


class TestEndToEndDataFlow:
    """Prove the data path: JSONL -> TraceEntry -> KVPolicy -> KVResultExtension."""

    def test_trace_to_policy_to_result(self, traces_dir):
        # Load one trace
        entries = []
        with open(traces_dir / "smoke_prefix_reuse.jsonl") as f:
            for line in f:
                if line.strip():
                    entries.append(TraceEntry(**json.loads(line.strip())))
        assert len(entries) > 0

        # Create a policy and exercise it
        policy = FullBF16KVPolicy()
        sample_k, sample_v = [1, 2, 3], [4, 5, 6]
        encoded = policy.encode(layer_id=0, k=sample_k, v=sample_v, metadata={})
        k_out, v_out = policy.decode(layer_id=0, compressed_kv=encoded, metadata={})
        assert k_out == sample_k
        assert v_out == sample_v

        # Get stats and merge into result extension
        stats = policy.report_stats()
        result = KVResultExtension(**stats)
        assert result.kv_policy_name == "full_bf16"
        assert result.effective_bits_per_element == 16.0
