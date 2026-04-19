#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  gmi_test_matrix.sh --gpu-type <h100|h200|b200|b300|gb200|gb300> [--cloud <gmi|aws>]

Runs a curated Lane B bare-metal matrix:
  - Qwen3.5 × vllm × 8k, 131k, 500k, 1m
  - Qwen3.5 × sglang × 500k
  - GPT-OSS × vllm × 131k chat+code, 500k chat
  - DSR1 × sglang × 131k chat+code
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_gmi_common.sh
source "$SCRIPT_DIR/_gmi_common.sh"

GPU_TYPE=""
CLOUD="gmi"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --cloud)
      CLOUD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

[[ -n "$GPU_TYPE" ]] || {
  usage >&2
  exit 1
}

validate_gpu_type "$GPU_TYPE" || exit 1
validate_cloud "$CLOUD" || exit 1

PORTABLE_SCRIPT="$SCRIPT_DIR/gmi_portable_benchmark.sh"
[[ -x "$PORTABLE_SCRIPT" ]] || {
  echo "Expected executable helper at $PORTABLE_SCRIPT" >&2
  exit 1
}

run_case() {
  local model="$1"
  local engine="$2"
  local context_band="$3"
  local workload="${4:-code}"

  echo
  echo "============================================================"
  echo "Running: cloud=${CLOUD} gpu=${GPU_TYPE} model=${model} engine=${engine} context=${context_band} workload=${workload}"
  echo "============================================================"

  "$PORTABLE_SCRIPT" \
    --gpu-type "$GPU_TYPE" \
    --cloud "$CLOUD" \
    --model "$model" \
    --engine "$engine" \
    --context-band "$context_band" \
    --workload "$workload"
}

run_case qwen3.5 vllm 8k chat
run_case qwen3.5 vllm 131k code
run_case qwen3.5 vllm 500k code
run_case qwen3.5 sglang 500k chat
run_case gptoss vllm 131k code
run_case gptoss vllm 131k chat
run_case gptoss vllm 500k chat
run_case dsr1 sglang 131k code
run_case dsr1 sglang 131k chat

# 1M preview — all Blackwell SKUs have enough HBM for Qwen 1M.
case "$GPU_TYPE" in
  b200|b300|gb200|gb300)
    run_case qwen3.5 vllm 1m code
    ;;
  *)
    echo "Skipping Qwen3.5 1M preview on $GPU_TYPE (requires blackwell: b200/b300/gb200/gb300)"
    ;;
esac

echo
echo "Curated Lane B test matrix completed successfully."
