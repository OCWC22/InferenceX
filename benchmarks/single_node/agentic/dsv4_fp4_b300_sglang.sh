#!/usr/bin/env bash
set -euo pipefail

# Agentic trace replay wrapper for DeepSeek-V4-Pro FP4 on B300 with SGLang.
# The server recipe lives in ../dsv4_fp4_b300_sglang.sh; AGENTIC_MODE switches
# the post-ready client from fixed random prompts to WEKA trace replay.

export AGENTIC_MODE=1
export ISL="${ISL:-8192}"
export OSL="${OSL:-1024}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"
export RESULT_FILENAME="${RESULT_FILENAME:-agentic_dsv4_fp4_b300_sglang}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-$REPO_ROOT}"

exec "$REPO_ROOT/benchmarks/single_node/dsv4_fp4_b300_sglang.sh"
