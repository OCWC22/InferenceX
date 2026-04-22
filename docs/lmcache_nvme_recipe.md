# LMCache NVMe Recipe

This is a paste-able NVMe cold-tier recipe for operators already using Cam's
existing LMCache replay lane.

Do not edit the wrapper script.
Instead, point the existing flow at an extra config file via
`LMCACHE_EXTRA_CONFIG_FILE`.

---

## 1. Intended use

Use this when explicit `--cpu-offload-gb` values are not enough, or when you do
not want to keep pushing more working-set pressure into host RAM.

Best fit:

- very large cold turns
- replay lanes with bursty revisits
- operators who already trust the LMCache path

---

## 2. Invocation pattern

Cam's existing H100 LMCache lane is the consumer:

- `experimental/multiturn/benchmarks/single_node/multiturn_fp8_h100_lmcache_aiperf.sh`

You do **not** modify that script.
You only export one extra environment variable before invoking it.

```bash
cat > /tmp/lmcache_nvme.yaml <<'YAML'
chunk_size: 256
local_cpu:
  enabled: true
  max_size: 40GB
local_disk:
  enabled: true
  path: /local_nvme/lmcache
  max_local_disk_size: 50GB
  file_rotation:
    enabled: true
    max_file_size: 4GB
    max_files: 16
    policy: lru
prefetch:
  enabled: false
metrics:
  enabled: true
YAML

export LMCACHE_EXTRA_CONFIG_FILE=/tmp/lmcache_nvme.yaml
```

Then run the existing wrapper exactly as usual.

---

## 3. Recommended file contents

```yaml
chunk_size: 256
local_cpu:
  enabled: true
  max_size: 40GB
local_disk:
  enabled: true
  path: /local_nvme/lmcache
  max_local_disk_size: 50GB
  file_rotation:
    enabled: true
    max_file_size: 4GB
    max_files: 16
    policy: lru
prefetch:
  enabled: false
metrics:
  enabled: true
```

Why these defaults:

- `chunk_size: 256` keeps chunking coarse enough for large replay assets
- `max_local_disk_size: 50GB` is large enough to matter but small enough to be a
  deliberate cold tier, not an unbounded scratch dump
- rotation prevents silent NVMe growth during long aiperf loops
- disabling prefetch keeps the first experiment easier to reason about

---

## 4. Operator notes

- prefer a local NVMe mount, not network storage
- keep the LMCache directory isolated per run when debugging corruption or stale
  cold blocks
- if disk latency is unstable, compare against the explicit CPU offload sweep
  before drawing conclusions
- if `80` GiB CPU spill already works cleanly, LMCache may be unnecessary

---

## 5. Minimal runbook snippet

```bash
export LMCACHE_EXTRA_CONFIG_FILE=/tmp/lmcache_nvme.yaml
bash experimental/multiturn/benchmarks/single_node/multiturn_fp8_h100_lmcache_aiperf.sh
```

This recipe is intentionally config-only so the entire PR stays additive and
avoids any `experimental/**` code changes.
