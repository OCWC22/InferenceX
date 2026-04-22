# KV Offload Playbook
This playbook is the operator companion to
`.github/configs/multiturn-agentic-trace-isb1-offload-sweep.yaml`.
It adds three opt-in knobs on top of Cam's existing replay + aiperf flow:
1. explicit `--cpu-offload-gb` sweep values (`0`, `20`, `40`, `80`)
2. a side-car `/metrics` probe (`tools/kv_offload_probe.py`)
3. an LMCache NVMe cold-tier recipe (`docs/lmcache_nvme_recipe.md`)
Nothing here requires a harness edit.
You keep using the same replay wrappers and only swap:
- the sweep YAML you hand the generator, and/or
- the extra config you point LMCache at, and/or
- the probe you launch in parallel.
---
## 1. What problem this solves
The original ISB1 replay lane exposed a coarse `offload ∈ {on, off, noprefix}`.
That is enough to show whether offload helps at all, but not enough to answer:
- how much CPU spill is needed before a cliff softens
- when prefix caching is better than spill
- when LMCache cold-tiering is cleaner than pushing more host RAM
- how preemption/swapping evolves during a real replay run
The new sweep file turns that binary switch into an operator gradient.
---
## 2. The three knobs, in one table
| Knob | What it changes | Best first use | Main failure mode |
|---|---|---|---|
| `offload=0/20/40/80` | passes `--cpu-offload-gb N` to vLLM | HBM is close but not enough | host-memory bandwidth / latency tax |
| `offload=noprefix` | disables prefix caching with no CPU spill | establish the clean-cache floor | looks worse than production, by design |
| LMCache NVMe recipe | adds a colder disk tier via `LMCACHE_EXTRA_CONFIG_FILE` | working set is bigger than host RAM comfort zone | reclaim/readback jitter if disk is weak |
A practical order:
1. baseline with `0`
2. measure `noprefix`
3. sweep `20 → 40 → 80`
4. only then try LMCache NVMe if the cliff still arrives too early
---
## 3. Interpreting the sweep values
### `offload=0`
This is the cleanest HBM-only comparison point.
The wrapper should omit `--cpu-offload-gb` entirely.
Use it to answer: “what happens if I rely only on HBM + prefix reuse?”
### `offload=20`
Small host spill budget.
Good first stop on H100-like rigs where you want to test whether the cliff is
caused by a modest overflow rather than a fundamentally oversized trace.
### `offload=40`
Middle ground.
Usually the best first operator setting on H200 if the workload just spills past
HBM during later turns or synchronized fan-out bursts.
### `offload=80`
Large spill budget.
Use this when the trace is intentionally pressure-heavy and you want to see if
more host residency flattens the cliff or just moves the pain into swap churn.
### `offload=noprefix`
No CPU spill, no prefix caching.
This is not meant to win.
It exists to expose the lower bound: what the lane looks like when every replay
turn pays the full prompt rebuild cost.
If `noprefix` and `0` are nearly identical, prefix caching is not buying much.
If `0` is much better than `noprefix`, preserve prefix reuse before reaching for
larger offload budgets.
---
## 4. When to use CPU offload vs LMCache NVMe vs noprefix
## Use CPU offload when
- the cliff arrives late in the replay, not immediately
- preemption exists but does not explode
- you need a minimal, no-script-change test
- the workload fits in host RAM with reasonable margin
CPU offload is the lowest-friction intervention because it stays inside the
existing vLLM process model.
## Use LMCache NVMe when
- `80` GiB of CPU spill still leaves the run unstable
- host RAM is precious or shared with many worker processes
- you want a colder overflow tier without touching the existing shell wrappers
- the replay contains very large cold segments that are revisited less often
LMCache adds more moving parts, but it gives you a larger cold tier than pure
CPU spill.
## Use `noprefix` when
- you want the “no cache help at all” floor
- you suspect cache sharing, not spill size, is dominating results
- you want to quantify how much reuse the trace actually provides
Do not treat `noprefix` as a production recommendation.
Treat it as a measurement control.
---
## 5. The curated KV-pressure subset
`datasets/isb1/kv_pressure/manifest.json` is intentionally small.
It points at existing converted traces with large replay inputs so you can force
pressure quickly without materializing new bundles.
Current entries:
| File | Cumulative ISL tokens | Peak per-turn ISL tokens | Why it matters |
|---|---:|---:|---|
| `preview/long_context_1m/.../isb1_hb_depth_cache_ulc2_offload_cliff_01.json` | 9,754,210 | 1,626,197 | the full long-context offload cliff |
| `preview/long_context_500k/.../isb1_hb_depth_cache_xlc2_hot_cold_session_mix_01.json` | 2,688,641 | 672,449 | intermediate hot/cold residency churn |
| `extension_131k/code_131k1k_qwen3.5/isb1_hb_depth_cache_xlc1_text_shared_prefix_swarm_01.json` | 1,214,768 | 304,114 | shared-prefix pressure without preview-only setup |
Use the validator flag to guarantee these numbers stay honest:
```bash
python3 tools/validate_kvcache_tester_trace.py \
  datasets/isb1/converted/ \
  --pressure-manifest datasets/isb1/kv_pressure/manifest.json
```
---
## 6. Running the side-car probe
The probe polls the vLLM Prometheus endpoint once per interval and appends JSONL.
Default interval is `1s`.
### Minimal launch
```bash
python3 tools/kv_offload_probe.py \
  --url http://127.0.0.1:8000 \
  --output /tmp/aiperf_run/probe.jsonl
```
### Launch beside a replay/aiperf run
```bash
python3 tools/kv_offload_probe.py \
  --url http://127.0.0.1:8000 \
  --output /tmp/aiperf_run/probe.jsonl \
  --interval 1 \
  > /tmp/aiperf_run/probe.stdout.log \
  2> /tmp/aiperf_run/probe.stderr.log &
PROBE_PID=$!
# run the existing replay / aiperf wrapper here
wait "$PROBE_PID"
```
If you want the probe to stop automatically in a scripted smoke test, pass the
hidden `--max-samples N` flag. The production path does not need it.
---
## 7. Probe fields and how to read them
Each JSONL row contains:
| Field | Meaning |
|---|---|
| `ts` | sample timestamp (`time.time()`) |
| `kv_cache_usage_perc` | best available KV occupancy gauge |
| `gpu_cache_usage_perc` | GPU-specific occupancy gauge when exposed |
| `cpu_offload_queue_depth` | optional queue depth metric if the endpoint exposes one |
| `num_preempted` | cumulative preemption counter |
| `num_swapped` | swap-related counter/gauge when exposed |
The probe accepts both older and newer vLLM metric names.
That matters because current stable docs expose `vllm:kv_cache_usage_perc`, while
older vLLM series exposed `vllm:gpu_cache_usage_perc`,
`vllm:num_requests_swapped`, and `vllm:num_preemptions_total`.
If a metric is missing, the field is written as `null`.
That is expected for version-skewed metrics like swap-related counters.
---
## 8. Reading `preempted` and `swapped`
### `num_preempted`
This tells you the scheduler had to evict or reshuffle work to make progress.
A slow increase can be acceptable in a pressure test.
A rapid increase while throughput collapses is the classic cliff signal.
### `num_swapped`
This is the stronger warning sign.
If swap-related counters begin moving while TTFT and end-to-end latency spike,
you are no longer in a “small spill tax” regime — you are in an offload-bound
regime.
### Rule of thumb
| Pattern | Likely interpretation |
|---|---|
| KV usage high, no preemption, no swap | close to limit but still healthy |
| preemption rises, swap flat | soft pressure; try a modest spill budget |
| preemption and swap both rise | cliff onset; compare `40` vs `80` vs LMCache |
| `noprefix` much worse than `0` | prefix reuse is doing real work |
| `80` barely better than `40` | host spill is no longer the right lever |
---
## 9. What the cliff looks like in the new YAML
The offload sweep is useful because it makes the cliff legible.
Typical pattern:
1. `0`: sharp collapse once the replay exceeds practical HBM residency
2. `20`: cliff shifts right slightly
3. `40`: cliff softens for medium pressure, still fails on the heaviest traces
4. `80`: either meaningfully extends stability or proves the run is now host-bound
5. `noprefix`: worst-case floor with cache reuse removed
You are looking for the smallest spill value that materially improves the curve.
If `80` is required just to survive, the lane may be a better fit for LMCache or
for a smaller working-set slice.
---
## 10. Suggested operating sequence
### Fast triage
1. run `0`
2. run `noprefix`
3. if `0` already fails, jump to `40`
4. attach the probe once a failure mode reproduces
### Detailed comparison
1. select one pressure trace from `datasets/isb1/kv_pressure/manifest.json`
2. run `0`, `20`, `40`, `80`, `noprefix`
3. store `probe.jsonl` beside each artifact directory
4. compare TTFT / throughput alongside `num_preempted` and `num_swapped`
### Escalation path
- if `20` helps: keep the lane simple and stay on CPU spill
- if `40` or `80` helps but swap churn remains: consider LMCache NVMe
- if none help and `noprefix` is similar to `0`: the real issue may be poor reuse,
  not insufficient spill
---
## 11. Paste-ready command checklist
### Validate the subset + traces
```bash
python3 tools/validate_kvcache_tester_trace.py \
  datasets/isb1/converted/ \
  --pressure-manifest datasets/isb1/kv_pressure/manifest.json
```
### Validate the sweep YAML parses
```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -c \
  "import yaml; yaml.safe_load(open('.github/configs/multiturn-agentic-trace-isb1-offload-sweep.yaml'))"
```
### Probe smoke test
```bash
/opt/homebrew/opt/python@3.13/bin/python3.13 -m pytest tools/test_kv_offload_probe.py -q
```
---
## 12. Guardrails
- this playbook is additive and opt-in
- nothing here edits `experimental/**`
- nothing here requires changing Cam's `*_lmcache_aiperf.sh` wrappers
- `datasets/isb1/kv_pressure/manifest.json` is reference-only and points at
  existing converted traces
- if you only need one answer, prefer the smallest pressure trace that still
  reproduces the failure
That is the entire point of this PR: give operators more precise offload
controls without forcing a harness fork.
