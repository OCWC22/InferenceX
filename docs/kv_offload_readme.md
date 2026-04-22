# KV Offload Readme

This PR adds three operator-facing KV-pressure controls without touching the
existing replay harness scripts:

1. **Granular CPU offload sweep**
   - file: `.github/configs/multiturn-agentic-trace-isb1-offload-sweep.yaml`
   - values: `0`, `20`, `40`, `80`, `noprefix`

2. **Live vLLM metrics probe**
   - file: `tools/kv_offload_probe.py`
   - output: JSONL side-car from `/metrics`

3. **LMCache NVMe cold-tier recipe**
   - file: `docs/lmcache_nvme_recipe.md`
   - consumed through `LMCACHE_EXTRA_CONFIG_FILE`

Supporting pieces:

- curated reference subset: `datasets/isb1/kv_pressure/manifest.json`
- validator extension: `tools/validate_kvcache_tester_trace.py --pressure-manifest`
- operator walkthrough: `docs/kv_offload_playbook.md`

Suggested first pass:

1. validate the trace tree + pressure manifest
2. compare `0` vs `noprefix`
3. sweep `20/40/80`
4. add the probe once the cliff reproduces
5. only then try LMCache NVMe
