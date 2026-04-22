# Operator recipe for `MOONCAKE_INPUT`

> **Status: speculative / non-blocking ops-note. Not a PR ask.** This file documents *one possible* way upstream operators could wire the ISB-1 mooncake corpus into Cam's aiperf scripts. It is NOT the canonical PR framing. The canonical framing is the "opt-in second mooncake corpus" described in [fork PR #2](https://github.com/OCWC22/InferenceX/pull/2) — operators run `--custom-dataset-type mooncake_trace` against `datasets/isb1/mooncake/` with their existing `MOONCAKE_INPUT` plumbing; no upstream patches required.
>
> **These diffs target upstream harness files (`benchmarks/single_node/multiturn_fp8_*_lmcache_aiperf.sh`, `.github/workflows/benchmark-multiturn-tmpl.yml`, `.github/workflows/multiturn-sweep.yml`). Do NOT apply locally — they are for operators pulling this data into the upstream harness.**

## MOONCAKE_INPUT semantics

Recommended `MOONCAKE_INPUT` support covers three cases:

1. **Local dir**  
   `MOONCAKE_INPUT=datasets/isb1/mooncake/core/code_8k1k`  
   Concatenate local `*.jsonl` files into the single `$TRACE_FILE` the harness already passes to `aiperf profile --custom-dataset-type mooncake_trace`.

2. **HF repo or repo subdir**  
   `MOONCAKE_INPUT=hf_semianalysisai--isb1-cc-traces/mooncake/core/code_8k1k`  
   Download the dataset repo, optionally target a subdirectory, then concatenate matched cached `*.jsonl` files into `$TRACE_FILE`.

3. **Unset**  
   Preserve the current `sammshen/lmcache-agentic-traces` download path and parquet/jsonl fallback so existing sweeps continue to work when `MOONCAKE_INPUT` is not provided.

## Shell script hunk

Apply this hunk to the H100 shell script. **H200 and B200 use the same dataset-materialization hunk.**

```diff
--- a/benchmarks/single_node/multiturn_fp8_h100_lmcache_aiperf.sh
+++ b/benchmarks/single_node/multiturn_fp8_h100_lmcache_aiperf.sh
@@ -9,6 +9,9 @@
 # Required env vars:
 #   MODEL, TP, USERS, OFFLOAD_MODE, TOTAL_CPU_DRAM_GB, RESULT_DIR
 # Optional:
+#   MOONCAKE_INPUT:
+#     - /path/to/mooncake/subdir
+#     - hf_<org>--<repo>[/optional/subdir]
 #   PORT (default 8888), REQUEST_TIMEOUT (default 3600)
 #   DURATION (if set, runs for this many seconds; otherwise runs to completion)
@@ -60,16 +63,37 @@ mkdir -p "$RESULT_DIR"
-# ---- Download and convert LMCache traces to mooncake format ----------------
-echo "Downloading LMCache traces..."
-hf download sammshen/lmcache-agentic-traces --repo-type dataset
-
-echo "Converting LMCache traces to mooncake format..."
+# ---- Resolve and materialize mooncake traces --------------------------------
+if [[ -n "${MOONCAKE_INPUT:-}" && "${MOONCAKE_INPUT}" == hf_* ]]; then
+    HF_SPEC="${MOONCAKE_INPUT#hf_}"
+    HF_REPO_SPEC="${HF_SPEC%%/*}"
+    HF_REPO="${HF_REPO_SPEC/--//}"
+    hf download "$HF_REPO" --repo-type dataset
+fi
+
+echo "Materializing mooncake traces..."
 python3 -c "
 import json, glob, os
 hf_cache = os.environ.get('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
-# Find the downloaded parquet/jsonl files in the HF cache
-candidates = glob.glob(os.path.join(hf_cache, 'datasets--sammshen--lmcache-agentic-traces', '**', '*.parquet'), recursive=True)
-if not candidates:
-    candidates = glob.glob(os.path.join(hf_cache, 'datasets--sammshen--lmcache-agentic-traces', '**', '*.jsonl'), recursive=True)
-if not candidates:
-    # Fallback: use datasets library to load from cache
+
+src = os.environ.get('MOONCAKE_INPUT', '').strip()
+jsonl_candidates = []
+if src.startswith('hf_'):
+    hf_spec = src[3:]
+    repo_spec, _, subdir = hf_spec.partition('/')
+    dataset = repo_spec.replace('--', '/', 1)
+    repo_cache = f'datasets--{dataset.replace('/', '--')}'
+    pattern = os.path.join(hf_cache, repo_cache, '**', subdir, '*.jsonl') if subdir else os.path.join(hf_cache, repo_cache, '**', '*.jsonl')
+    jsonl_candidates = sorted(glob.glob(pattern, recursive=True))
+elif src:
+    jsonl_candidates = sorted(glob.glob(os.path.join(src, '*.jsonl')))
+
+out_path = '$TRACE_FILE'
+if jsonl_candidates:
+    with open(out_path, 'w', encoding='utf-8') as out:
+        for path in jsonl_candidates:
+            out.write(open(path, encoding='utf-8').read())
+    print(f'Concatenated {len(jsonl_candidates)} JSONL file(s) into {out_path}')
+    raise SystemExit(0)
+
+candidates = glob.glob(os.path.join(hf_cache, 'datasets--sammshen--lmcache-agentic-traces', '**', '*.parquet'), recursive=True)
+if not candidates:
+    candidates = glob.glob(os.path.join(hf_cache, 'datasets--sammshen--lmcache-agentic-traces', '**', '*.jsonl'), recursive=True)
+if not candidates:
     from datasets import load_dataset
     ds = load_dataset('sammshen/lmcache-agentic-traces', split='train')
     rows = list(ds)
@@ -85,7 +109,6 @@ else:
- out_path = '$TRACE_FILE'
  sessions = set()
  skipped = 0
  with open(out_path, 'w') as f:
```

## Workflow YAML hunks

### `benchmark-multiturn-tmpl.yml`

```diff
--- a/.github/workflows/benchmark-multiturn-tmpl.yml
+++ b/.github/workflows/benchmark-multiturn-tmpl.yml
@@ -72,6 +72,11 @@ on:
       trace-dir:
         description: "Override trace directory (relative to kv-cache-tester dir)"
         required: false
         type: string
         default: ''
+      mooncake-input:
+        description: "Override mooncake JSONL source (local dir or hf_<org>--<repo>[/subdir])"
+        required: false
+        type: string
+        default: ''
@@ -101,6 +106,7 @@ env:
   HASH_BLOCK_MODE: ${{ inputs.hash-block-mode }}
   TRACE_DIR: ${{ inputs.trace-dir }}
+  MOONCAKE_INPUT: ${{ inputs.mooncake-input }}
   DEBUG_TRACE: ${{ inputs.debug-trace }}
   NO_MAX_TOKENS: ${{ inputs.no-max-tokens }}
```

### `multiturn-sweep.yml`

```diff
--- a/.github/workflows/multiturn-sweep.yml
+++ b/.github/workflows/multiturn-sweep.yml
@@ -105,6 +105,11 @@ on:
       trace_dir:
         description: 'Override trace directory (e.g. traces, traces_neon). Relative to kv-cache-tester dir.'
         required: false
         default: ''
         type: string
+      mooncake_input:
+        description: 'Override mooncake JSONL source (local dir or hf_<org>--<repo>[/subdir]).'
+        required: false
+        default: ''
+        type: string
@@ -219,6 +224,7 @@ jobs:
       ignore-eos: ${{ inputs.ignore_eos }}
       hash-block-mode: ${{ inputs.hash_block_mode }}
       trace-dir: ${{ inputs.trace_dir }}
+      mooncake-input: ${{ inputs.mooncake_input }}
       debug-trace: ${{ inputs.debug_trace }}
       no-max-tokens: ${{ inputs.no_max_tokens }}
```

## How to consume

- Pull replay inputs from `datasets/isb1/mooncake/`.
- Use the first-wave sweep cells in `.github/configs/multiturn-agentic-trace-isb1-mooncake.yaml`.
- Point `MOONCAKE_INPUT` at either a local subtree or an `hf_<org>--<repo>[/subdir]` dataset path.
- Leave `MOONCAKE_INPUT` unset to preserve the legacy `sammshen/lmcache-agentic-traces` fallback.
