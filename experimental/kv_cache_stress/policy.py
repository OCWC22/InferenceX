"""KV cache policy interface and baseline implementations.

Defines a Protocol that all KV cache management policies must satisfy.
Phase 1 provides:
  - FullBF16KVPolicy: reference baseline (no compression, no eviction)
  - Stubs for INT8, INT4, keep-recent-window, oracle-top-attention

Phase 2+ will add real implementations backed by tensor operations.
Fused CUDA/Triton kernels are explicitly out of scope for phase 1.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KVPolicy(Protocol):
    """Interface for KV cache management policies.

    A policy controls how key-value tensors are stored, compressed,
    selected, and used during attention. Policies fall into families:

      - full_cache: BF16/FP16 baseline, no transformation
      - quantization: uniform INT8/INT4, structured (TurboQuant), adaptive (AQUA-KV)
      - pruning_eviction: SnapKV, H2O, TriAttention
      - structured_compression: KVTC (PCA + quantize + entropy code)
      - offload_prefetch: SpeCache-style GPU/CPU/disk tiering

    All methods accept Any types in phase 1 to avoid mandatory PyTorch
    dependency. Phase 2 will tighten signatures to Optional[torch.Tensor].
    """

    @property
    def name(self) -> str:
        """Short unique identifier, e.g. 'full_bf16', 'turbo_quant_3bit'."""
        ...

    @property
    def family(self) -> str:
        """Policy family: full_cache | quantization | pruning_eviction | ..."""
        ...

    @property
    def requires_calibration(self) -> bool:
        """Whether calibrate() must be called before encode/decode."""
        ...

    @property
    def supports_fused_attention(self) -> bool:
        """Whether attention() operates directly on compressed KV."""
        ...

    @property
    def supports_reactivation(self) -> bool:
        """Whether the policy handles offload/restore from CPU/disk."""
        ...

    def calibrate(
        self,
        model: Any,
        calibration_loader: Any,
        *,
        seed: int = 42,
        max_samples: int = 128,
    ) -> dict:
        """One-time calibration (e.g. quantization range estimation).

        Returns a metadata dict that should be persisted for reproducibility.
        Policies that don't need calibration should return an empty dict.
        """
        ...

    def encode(self, layer_id: int, k: Any, v: Any, metadata: dict) -> Any:
        """Compress/transform KV tensors for storage.

        Returns an opaque encoded object (policy-specific format).
        """
        ...

    def decode(self, layer_id: int, compressed_kv: Any, metadata: dict) -> tuple[Any, Any]:
        """Decompress encoded KV back to (keys, values)."""
        ...

    def select(
        self, layer_id: int, q: Any, compressed_or_raw_kv: Any, metadata: dict
    ) -> Any:
        """Select/prune KV entries based on query attention patterns.

        Returns selected KV (format is policy-specific).
        For non-pruning policies, this is a pass-through.
        """
        ...

    def attention(self, q: Any, compressed_kv: Any, metadata: dict) -> Any:
        """Compute attention output, possibly fused with decompression.

        Policies without fused attention should decompress then do standard attn.
        """
        ...

    def report_stats(self) -> dict:
        """Return a dict of stats suitable for KVResultExtension fields."""
        ...


# ---------------------------------------------------------------------------
# Baseline: Full BF16 (reference)
# ---------------------------------------------------------------------------

class FullBF16KVPolicy:
    """Reference baseline: store full BF16 KV with no compression or eviction."""

    @property
    def name(self) -> str:
        return "full_bf16"

    @property
    def family(self) -> str:
        return "full_cache"

    @property
    def requires_calibration(self) -> bool:
        return False

    @property
    def supports_fused_attention(self) -> bool:
        return False

    @property
    def supports_reactivation(self) -> bool:
        return False

    def calibrate(
        self,
        model: Any,
        calibration_loader: Any,
        *,
        seed: int = 42,
        max_samples: int = 128,
    ) -> dict:
        return {}

    def encode(self, layer_id: int, k: Any, v: Any, metadata: dict) -> Any:
        return (k, v)

    def decode(self, layer_id: int, compressed_kv: Any, metadata: dict) -> tuple[Any, Any]:
        k, v = compressed_kv
        return k, v

    def select(
        self, layer_id: int, q: Any, compressed_or_raw_kv: Any, metadata: dict
    ) -> Any:
        return compressed_or_raw_kv

    def attention(self, q: Any, compressed_kv: Any, metadata: dict) -> Any:
        # Phase 1: no actual attention computation
        return None

    def report_stats(self) -> dict:
        return {
            "kv_policy_name": "full_bf16",
            "kv_policy_family": "full_cache",
            "effective_bits_per_element": 16.0,
            "compression_ratio_vs_bf16": 1.0,
            "requires_calibration": False,
        }


# ---------------------------------------------------------------------------
# Stubs: these define the interface contract for phase 2+ implementations
# ---------------------------------------------------------------------------

class _StubPolicy:
    """Base for stub policies that raise NotImplementedError on all ops."""

    _name: str = "stub"
    _family: str = "stub"
    _requires_calibration: bool = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> str:
        return self._family

    @property
    def requires_calibration(self) -> bool:
        return self._requires_calibration

    @property
    def supports_fused_attention(self) -> bool:
        return False

    @property
    def supports_reactivation(self) -> bool:
        return False

    def calibrate(self, model: Any, calibration_loader: Any, *, seed: int = 42, max_samples: int = 128) -> dict:
        raise NotImplementedError(f"{self._name}: phase 2+ implementation")

    def encode(self, layer_id: int, k: Any, v: Any, metadata: dict) -> Any:
        raise NotImplementedError(f"{self._name}: phase 2+ implementation")

    def decode(self, layer_id: int, compressed_kv: Any, metadata: dict) -> tuple[Any, Any]:
        raise NotImplementedError(f"{self._name}: phase 2+ implementation")

    def select(self, layer_id: int, q: Any, compressed_or_raw_kv: Any, metadata: dict) -> Any:
        raise NotImplementedError(f"{self._name}: phase 2+ implementation")

    def attention(self, q: Any, compressed_kv: Any, metadata: dict) -> Any:
        raise NotImplementedError(f"{self._name}: phase 2+ implementation")

    def report_stats(self) -> dict:
        return {"kv_policy_name": self._name, "kv_policy_family": self._family}


class UniformINT8KVPolicy(_StubPolicy):
    """TODO(phase2): Uniform per-tensor INT8 quantization baseline."""
    _name = "uniform_int8"
    _family = "quantization"
    _requires_calibration = False


class UniformINT4KVPolicy(_StubPolicy):
    """TODO(phase2): Uniform group-wise INT4 quantization baseline."""
    _name = "uniform_int4"
    _family = "quantization"
    _requires_calibration = True


class KeepRecentWindowKVPolicy(_StubPolicy):
    """TODO(phase2): Keep only the most recent N tokens, evict the rest."""
    _name = "keep_recent_window"
    _family = "pruning_eviction"


class OracleTopAttentionKVPolicy(_StubPolicy):
    """TODO(phase2): Offline oracle that keeps top-k by attention score. Diagnostic only."""
    _name = "oracle_top_attention"
    _family = "pruning_eviction"


# Registry for discovery
BUILTIN_POLICIES: dict[str, type] = {
    "full_bf16": FullBF16KVPolicy,
    "uniform_int8": UniformINT8KVPolicy,
    "uniform_int4": UniformINT4KVPolicy,
    "keep_recent_window": KeepRecentWindowKVPolicy,
    "oracle_top_attention": OracleTopAttentionKVPolicy,
}
