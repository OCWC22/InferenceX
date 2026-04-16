"""Tests for deterministic synthetic trace generation."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_synthetic_traces import (
    generate_prefix_reuse_sweep,
    generate_kv_offload_cliff,
    generate_long_reasoning_growth,
    GENERATORS,
)
from schema import TraceEntry


class TestDeterminism:
    """Same seed must produce identical output."""

    @pytest.mark.parametrize("scenario", list(GENERATORS.keys()))
    def test_same_seed_same_output(self, scenario):
        gen_fn = GENERATORS[scenario]
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        entries1 = gen_fn(rng1)
        entries2 = gen_fn(rng2)
        assert entries1 == entries2

    @pytest.mark.parametrize("scenario", list(GENERATORS.keys()))
    def test_different_seeds_different_output(self, scenario):
        gen_fn = GENERATORS[scenario]
        rng1 = random.Random(42)
        rng2 = random.Random(99)
        entries1 = gen_fn(rng1)
        entries2 = gen_fn(rng2)
        assert entries1 != entries2


class TestGeneratedEntriesValidate:
    """All generated entries must pass TraceEntry validation."""

    @pytest.mark.parametrize("scenario", list(GENERATORS.keys()))
    def test_all_entries_valid(self, scenario):
        gen_fn = GENERATORS[scenario]
        rng = random.Random(42)
        entries = gen_fn(rng)
        for i, entry_dict in enumerate(entries):
            try:
                TraceEntry(**entry_dict)
            except Exception as e:
                pytest.fail(f"Entry {i} in {scenario} failed validation: {e}")

    @pytest.mark.parametrize("scenario", list(GENERATORS.keys()))
    def test_unique_trace_ids(self, scenario):
        gen_fn = GENERATORS[scenario]
        rng = random.Random(42)
        entries = gen_fn(rng)
        trace_ids = [e["trace_id"] for e in entries]
        assert len(trace_ids) == len(set(trace_ids))


class TestCustomEntryCount:

    @pytest.mark.parametrize("scenario", list(GENERATORS.keys()))
    def test_custom_num_entries(self, scenario):
        gen_fn = GENERATORS[scenario]
        rng = random.Random(42)
        entries = gen_fn(rng, num_entries=5)
        assert len(entries) == 5

    @pytest.mark.parametrize("scenario", list(GENERATORS.keys()))
    def test_single_entry(self, scenario):
        gen_fn = GENERATORS[scenario]
        rng = random.Random(42)
        entries = gen_fn(rng, num_entries=1)
        assert len(entries) == 1
        TraceEntry(**entries[0])


class TestScenarioSpecifics:

    def test_prefix_reuse_has_shared_prefixes(self):
        rng = random.Random(42)
        entries = generate_prefix_reuse_sweep(rng)
        prefix_ids = {e.get("system_prefix_id") for e in entries}
        prefix_ids.discard(None)
        assert len(prefix_ids) >= 2

    def test_offload_cliff_has_tiers(self):
        rng = random.Random(42)
        entries = generate_kv_offload_cliff(rng)
        tiers = {e["residency_tier_hint"] for e in entries}
        assert "gpu" in tiers
        assert "cpu" in tiers

    def test_reasoning_growth_has_quality_eval(self):
        rng = random.Random(42)
        entries = generate_long_reasoning_growth(rng)
        for e in entries:
            assert e["requires_quality_eval"] is True
            assert e["quality_task"] is not None
