import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import TraceEntry, KVResultExtension, TraceFile, Scenario, ResidencyTier  # noqa: E402

TRACES_DIR = Path(__file__).parent.parent / "traces"


@pytest.fixture
def traces_dir():
    return TRACES_DIR


@pytest.fixture
def sample_trace_entry():
    """Minimal valid trace entry dict."""
    return {
        "trace_id": "test-001",
        "session_id": "sess-001",
        "turn_id": 0,
        "workload_family": "kv_cache_stress",
        "scenario": "prefix_reuse_sweep",
        "input_tokens_total": 1024,
        "expected_output_tokens": 128,
    }


@pytest.fixture
def sample_prefix_reuse_entry():
    """Trace entry for prefix reuse scenario with shared prefix."""
    return {
        "trace_id": "prefix-001",
        "session_id": "sess-prefix-001",
        "turn_id": 0,
        "workload_family": "kv_cache_stress",
        "scenario": "prefix_reuse_sweep",
        "system_prefix_id": "sys_prefix_001",
        "input_tokens_total": 4096,
        "shared_prefix_tokens": 3072,
        "unique_suffix_tokens": 1024,
        "expected_output_tokens": 512,
        "expected_prefix_hit_rate": 0.75,
    }


@pytest.fixture
def sample_offload_entry():
    """Trace entry for offload cliff scenario."""
    return {
        "trace_id": "offload-001",
        "session_id": "sess-offload-001",
        "turn_id": 1,
        "workload_family": "kv_cache_stress",
        "scenario": "kv_offload_cliff",
        "parent_turn_id": 0,
        "input_tokens_total": 8192,
        "unique_suffix_tokens": 8192,
        "expected_output_tokens": 256,
        "idle_ms_since_last_turn": 60000.0,
        "residency_tier_hint": "cpu",
    }


@pytest.fixture
def sample_reasoning_entry():
    """Trace entry for long reasoning growth scenario."""
    return {
        "trace_id": "reasoning-001",
        "session_id": "sess-reasoning-001",
        "turn_id": 0,
        "workload_family": "kv_cache_stress",
        "scenario": "long_reasoning_growth",
        "input_tokens_total": 32768,
        "unique_suffix_tokens": 32768,
        "expected_output_tokens": 16384,
        "requires_quality_eval": True,
        "quality_task": "aime25",
        "gold_answer": "42",
    }
