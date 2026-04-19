# shellcheck shell=bash
# Lane B bare-metal shared helpers.
#
# Source me — do not execute. All public functions are prefixed `laneb_*` or
# `validate_*` and print their single-line output to stdout (caller captures
# via $(...)). Failure paths use a non-zero exit status; callers decide
# whether to `die` with a user-facing message.
#
# Scope:
#   * Accepted GPUs: h100, h200, b200, b300, gb200, gb300
#     (Hopper + full Blackwell family — standard B200/GB200 and Ultra
#     B300/GB300. B300 is x86 SXM; GB300 is Grace-attached aarch64.)
#   * Accepted clouds: gmi, aws
#   * Runner labels: <gpu>-<cloud>-baremetal (stable identifier, not a
#     GitHub runner tag — Lane B is not Slurm/GHA)
#   * Hardware profile IDs: canonical `nvidia:<gpu>_<form>_<hbm>` strings
#     that match the values baked into ISB1 exports. GB200 uses the
#     NVL72-style profile that already appears in committed exports; the
#     Ultra SKUs (b300, gb300) follow the same pattern.
#   * Image selection: cloud-agnostic defaults per (engine, gpu). GB200,
#     B300, and GB300 have no pinned public tag yet, so their defaults
#     are empty and the caller MUST provide LANEB_IMAGE_<CLOUD>_<GPU>_<ENGINE>.
#
# NOTE: We deliberately do NOT `set -Eeuo pipefail` here — sourced
# libraries must not mutate the caller's shell options. Every wrapper in
# this directory sets its own strict mode.

# Accepted values. Kept as bash arrays so the helpers can also expose them
# to `--help` output if a wrapper wants to list what's allowed.
LANEB_SUPPORTED_GPUS=("h100" "h200" "b200" "b300" "gb200" "gb300")
LANEB_SUPPORTED_CLOUDS=("gmi" "aws")

_laneb_in_list() {
  # $1 = needle, remaining args = haystack
  local needle="$1"
  shift
  local element
  for element in "$@"; do
    [[ "$element" == "$needle" ]] && return 0
  done
  return 1
}

validate_gpu_type() {
  # usage: validate_gpu_type <gpu>
  # prints nothing; rc 0 on valid, rc 1 with an error to stderr on invalid.
  local gpu="${1:-}"
  if [[ -z "$gpu" ]]; then
    echo "validate_gpu_type: missing argument" >&2
    return 1
  fi
  if ! _laneb_in_list "$gpu" "${LANEB_SUPPORTED_GPUS[@]}"; then
    echo "validate_gpu_type: unsupported GPU '$gpu' (allowed: ${LANEB_SUPPORTED_GPUS[*]})" >&2
    return 1
  fi
  return 0
}

validate_cloud() {
  # usage: validate_cloud <cloud>
  local cloud="${1:-}"
  if [[ -z "$cloud" ]]; then
    echo "validate_cloud: missing argument" >&2
    return 1
  fi
  if ! _laneb_in_list "$cloud" "${LANEB_SUPPORTED_CLOUDS[@]}"; then
    echo "validate_cloud: unsupported cloud '$cloud' (allowed: ${LANEB_SUPPORTED_CLOUDS[*]})" >&2
    return 1
  fi
  return 0
}

laneb_runner_type() {
  # usage: laneb_runner_type <gpu> <cloud>
  # prints "<gpu>-<cloud>-baremetal" (stable Lane B runner label).
  local gpu="${1:-}" cloud="${2:-}"
  validate_gpu_type "$gpu" || return 1
  validate_cloud "$cloud" || return 1
  printf '%s-%s-baremetal\n' "$gpu" "$cloud"
}

laneb_hardware_profile_id() {
  # usage: laneb_hardware_profile_id <gpu>
  # prints the canonical hardware_profile_id used in ISB1 exports.
  local gpu="${1:-}"
  validate_gpu_type "$gpu" || return 1
  case "$gpu" in
    h100)  printf 'nvidia:h100_sxm_80gb\n' ;;
    h200)  printf 'nvidia:h200_sxm_141gb\n' ;;
    b200)  printf 'nvidia:b200_sxm_180gb\n' ;;
    b300)  printf 'nvidia:b300_sxm_288gb\n' ;;
    gb200) printf 'nvidia:gb200_nvl72_192gb\n' ;;
    gb300) printf 'nvidia:gb300_nvl72_288gb\n' ;;
  esac
}

laneb_gpu_aarch64() {
  # usage: laneb_gpu_aarch64 <gpu>
  # prints "1" for Grace-attached GPUs (GB200/GB300 NVL72), "0" otherwise.
  # Used so the caller can surface a clear message when a default x86
  # image is requested on aarch64 hardware.
  local gpu="${1:-}"
  validate_gpu_type "$gpu" || return 1
  case "$gpu" in
    gb200) printf '1\n' ;;
    gb300) printf '1\n' ;;
    *)     printf '0\n' ;;
  esac
}

laneb_default_image() {
  # usage: laneb_default_image <cloud> <gpu> <engine>
  # prints the default container image for (cloud, gpu, engine), or an
  # empty string when the caller must supply an env override. GB200 has
  # no pinned public tag yet and therefore returns empty on purpose.
  #
  # Override precedence (evaluated in the caller):
  #   1. LANEB_IMAGE_<CLOUD>_<GPU>_<ENGINE>   — cloud+gpu+engine specific
  #   2. LANEB_IMAGE_<GPU>_<ENGINE>           — gpu+engine
  #   3. laneb_default_image (this function)  — baseline defaults
  local cloud="${1:-}" gpu="${2:-}" engine="${3:-}"
  validate_cloud "$cloud" || return 1
  validate_gpu_type "$gpu" || return 1
  case "$engine" in
    vllm|sglang) ;;
    *)
      echo "laneb_default_image: unsupported engine '$engine' (allowed: vllm sglang)" >&2
      return 1
      ;;
  esac
  # Explicit table. Rows that deliberately emit empty stdout with rc=0
  # (GB200, B300, GB300) signal "no pinned public tag" so the caller
  # surfaces a precise "set LANEB_IMAGE_* override" message. Do NOT
  # replace those empties with an inherited b200/h200 tag: B300 is a
  # distinct Blackwell Ultra (sm_103) and GB300 is aarch64 Ultra — a
  # silent inherit would fail later inside the container with a much
  # less obvious error. The fall-through `*` branch errors on unknown
  # combos so a future GPU added above (e.g. mi300x) without a default
  # here fails loudly instead of silently returning empty.
  case "$engine:$gpu" in
    vllm:h100)    printf 'vllm/vllm-openai:v0.18.0\n' ;;
    vllm:h200)    printf 'vllm/vllm-openai:v0.18.0\n' ;;
    vllm:b200)    printf 'vllm/vllm-openai:v0.19.0-cu130\n' ;;
    vllm:b300)    printf '\n' ;;   # Blackwell Ultra x86; override required
    vllm:gb200)   printf '\n' ;;   # aarch64; override required
    vllm:gb300)   printf '\n' ;;   # aarch64 Ultra; override required
    sglang:h100)  printf 'lmsysorg/sglang:v0.5.9-cu130\n' ;;
    sglang:h200)  printf 'lmsysorg/sglang:v0.5.9-cu130\n' ;;
    sglang:b200)  printf 'lmsysorg/sglang:v0.5.9-cu130\n' ;;
    sglang:b300)  printf '\n' ;;   # Blackwell Ultra x86; override required
    sglang:gb200) printf '\n' ;;   # aarch64; override required
    sglang:gb300) printf '\n' ;;   # aarch64 Ultra; override required
    *)
      echo "laneb_default_image: no image table entry for engine=$engine gpu=$gpu" >&2
      return 1
      ;;
  esac
}

laneb_resolve_image() {
  # usage: laneb_resolve_image <cloud> <gpu> <engine>
  # prints the resolved image after applying env overrides, or empty if
  # neither an override nor a default exists (caller must `die`).
  local cloud="${1:-}" gpu="${2:-}" engine="${3:-}"
  validate_cloud "$cloud" || return 1
  validate_gpu_type "$gpu" || return 1

  # Uppercased lookup keys: LANEB_IMAGE_<CLOUD>_<GPU>_<ENGINE>
  local cloud_up gpu_up engine_up
  cloud_up=$(printf '%s' "$cloud" | tr '[:lower:]' '[:upper:]')
  gpu_up=$(printf '%s' "$gpu" | tr '[:lower:]' '[:upper:]')
  engine_up=$(printf '%s' "$engine" | tr '[:lower:]' '[:upper:]')

  local specific_key="LANEB_IMAGE_${cloud_up}_${gpu_up}_${engine_up}"
  local generic_key="LANEB_IMAGE_${gpu_up}_${engine_up}"

  if [[ -n "${!specific_key:-}" ]]; then
    printf '%s\n' "${!specific_key}"
    return 0
  fi
  if [[ -n "${!generic_key:-}" ]]; then
    printf '%s\n' "${!generic_key}"
    return 0
  fi
  laneb_default_image "$cloud" "$gpu" "$engine"
}
