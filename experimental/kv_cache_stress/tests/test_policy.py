"""Tests for KVPolicy protocol and implementations."""
import pytest

from policy import (
    KVPolicy,
    FullBF16KVPolicy,
    UniformINT8KVPolicy,
    UniformINT4KVPolicy,
    KeepRecentWindowKVPolicy,
    OracleTopAttentionKVPolicy,
    BUILTIN_POLICIES,
)


class TestKVPolicyProtocol:

    def test_full_bf16_satisfies_protocol(self):
        policy = FullBF16KVPolicy()
        assert isinstance(policy, KVPolicy)

    def test_full_bf16_name(self):
        policy = FullBF16KVPolicy()
        assert policy.name == "full_bf16"
        assert policy.family == "full_cache"

    def test_full_bf16_properties(self):
        policy = FullBF16KVPolicy()
        assert policy.requires_calibration is False
        assert policy.supports_fused_attention is False
        assert policy.supports_reactivation is False

    def test_full_bf16_calibrate_is_noop(self):
        policy = FullBF16KVPolicy()
        result = policy.calibrate(None, None)
        assert result == {}

    def test_full_bf16_encode_decode_roundtrip(self):
        policy = FullBF16KVPolicy()
        k, v = [1, 2, 3], [4, 5, 6]
        encoded = policy.encode(layer_id=0, k=k, v=v, metadata={})
        k_out, v_out = policy.decode(layer_id=0, compressed_kv=encoded, metadata={})
        assert k_out == k
        assert v_out == v

    def test_full_bf16_select_passthrough(self):
        policy = FullBF16KVPolicy()
        kv = ([1, 2, 3], [4, 5, 6])
        result = policy.select(layer_id=0, q=None, compressed_or_raw_kv=kv, metadata={})
        assert result == kv

    def test_full_bf16_report_stats(self):
        policy = FullBF16KVPolicy()
        stats = policy.report_stats()
        assert stats["kv_policy_name"] == "full_bf16"
        assert stats["kv_policy_family"] == "full_cache"
        assert stats["effective_bits_per_element"] == 16.0
        assert stats["compression_ratio_vs_bf16"] == 1.0
        assert stats["requires_calibration"] is False


class TestStubPolicies:

    @pytest.mark.parametrize("cls,name,family", [
        (UniformINT8KVPolicy, "uniform_int8", "quantization"),
        (UniformINT4KVPolicy, "uniform_int4", "quantization"),
        (KeepRecentWindowKVPolicy, "keep_recent_window", "pruning_eviction"),
        (OracleTopAttentionKVPolicy, "oracle_top_attention", "pruning_eviction"),
    ])
    def test_stub_has_name_and_family(self, cls, name, family):
        policy = cls()
        assert policy.name == name
        assert policy.family == family

    @pytest.mark.parametrize("cls", [
        UniformINT8KVPolicy,
        UniformINT4KVPolicy,
        KeepRecentWindowKVPolicy,
        OracleTopAttentionKVPolicy,
    ])
    def test_stub_encode_raises(self, cls):
        policy = cls()
        with pytest.raises(NotImplementedError, match="phase 2"):
            policy.encode(layer_id=0, k=None, v=None, metadata={})

    @pytest.mark.parametrize("cls", [
        UniformINT8KVPolicy,
        UniformINT4KVPolicy,
        KeepRecentWindowKVPolicy,
        OracleTopAttentionKVPolicy,
    ])
    def test_stub_decode_raises(self, cls):
        policy = cls()
        with pytest.raises(NotImplementedError, match="phase 2"):
            policy.decode(layer_id=0, compressed_kv=None, metadata={})

    @pytest.mark.parametrize("cls", [
        UniformINT8KVPolicy,
        UniformINT4KVPolicy,
        KeepRecentWindowKVPolicy,
        OracleTopAttentionKVPolicy,
    ])
    def test_stub_report_stats_works(self, cls):
        policy = cls()
        stats = policy.report_stats()
        assert "kv_policy_name" in stats
        assert "kv_policy_family" in stats

    def test_stubs_satisfy_protocol(self):
        for cls in [UniformINT8KVPolicy, UniformINT4KVPolicy,
                    KeepRecentWindowKVPolicy, OracleTopAttentionKVPolicy]:
            policy = cls()
            assert isinstance(policy, KVPolicy)


class TestBuiltinRegistry:

    def test_all_policies_registered(self):
        assert "full_bf16" in BUILTIN_POLICIES
        assert "uniform_int8" in BUILTIN_POLICIES
        assert "uniform_int4" in BUILTIN_POLICIES
        assert "keep_recent_window" in BUILTIN_POLICIES
        assert "oracle_top_attention" in BUILTIN_POLICIES

    def test_registry_instantiation(self):
        for name, cls in BUILTIN_POLICIES.items():
            policy = cls()
            assert policy.name == name
