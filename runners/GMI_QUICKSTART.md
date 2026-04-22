# GMI Cloud H200 ISB-1 quickstart
> Manual GMI Cloud path only. No Slurm, enroot, squash files, or GitHub runner labels.
## What you need
- GMI Cloud H200: 1x node is enough; 2x lets you run code-8k/chat-32k on one node and code-131k on the other
- SSH access, read-only HF token, ~30 minutes
- This branch checked out on the box
## Provision + SSH
Create 1-2 H200 instances in GMI Cloud, note the public IP(s), then:
```bash
ssh ubuntu@<gmi-ip>
```
## Install once
```bash
sudo apt-get update
sudo apt-get install -y git docker.io python3-pip curl jq
sudo usermod -aG docker $USER
exit
```
Re-ssh once so the docker group applies, then continue:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install 'aiperf==0.5.0' 'huggingface_hub>=0.23'
docker pull vllm/vllm-openai:v0.18.0
git clone https://github.com/OCWC22/InferenceX.git
cd InferenceX
git checkout isb1/gmi-h200-runbook
export HF_TOKEN=hf_xxx
nvidia-smi
```
## Fixed values for this lane
- Model from shipped H200 Qwen fp8 config: `Qwen/Qwen3.5-397B-A17B-FP8`
- Pinned vLLM image: `vllm/vllm-openai:v0.18.0`
- Corpus alias: `hf_wchen22--isb1-cc-traces`
- Manual offload mapping here: `on` = `--cpu-offload-gb 40`; `off` = no offload flag; `noprefix` = offload off + `--no-enable-prefix-caching`
## Prep once
```bash
export MODEL='Qwen/Qwen3.5-397B-A17B-FP8'
export IMAGE='vllm/vllm-openai:v0.18.0'
export HF_ALIAS='hf_wchen22--isb1-cc-traces'
export HF_REPO="${HF_ALIAS#hf_}"; export HF_REPO="${HF_REPO/--//}"
export PORT=8000
export ARTIFACT_ROOT="$PWD/gmi_runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ARTIFACT_ROOT"
fetch_trace () {
  local rel="$1" out="$2"
  curl -LfsS -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/datasets/${HF_REPO}/resolve/main/${rel}" \
    -o "$out" && test -s "$out"
}
start_server () {
  local tp="$1" max_model_len="$2" offload="$3"
  local prefix_flag='--enable-prefix-caching'
  local offload_flag=''
  [[ "$offload" == 'noprefix' ]] && prefix_flag='--no-enable-prefix-caching'
  [[ "$offload" == 'on' ]] && offload_flag='--cpu-offload-gb 40'
  docker rm -f qwen35-h200 >/dev/null 2>&1 || true
  docker run -d --name qwen35-h200 --gpus all --ipc=host --network host \
    -e HF_TOKEN -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$tp" \
    --enable-expert-parallel \
    --reasoning-parser qwen3 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len "$max_model_len" \
    $prefix_flag $offload_flag
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null && return 0
    sleep 5
  done
  docker logs --tail=200 qwen35-h200 || true
  return 1
}
stop_server () { docker rm -f qwen35-h200 >/dev/null 2>&1 || true; }
```
## Per-cell settings
```bash
# code-8k   -> tp4, max-model-len 9472
CELL=code-8k; TP=4; MAX_MODEL_LEN=9472
TRACE_REL='mooncake/core/code_8k1k/isb1_core_code_8k1k.jsonl'
# chat-32k  -> tp4, max-model-len 34048
CELL=chat-32k; TP=4; MAX_MODEL_LEN=34048
TRACE_REL='mooncake/extension_32k/chat_32k1k_qwen3.5/isb1_extension_32k_chat_32k1k_qwen3_5.jsonl'
# code-131k -> tp8, max-model-len 132352
CELL=code-131k; TP=8; MAX_MODEL_LEN=132352
TRACE_REL='mooncake/extension_131k/code_131k1k_qwen3.5/isb1_extension_131k_vllm_code_131k1k_qwen3_5.jsonl'
```
## One run
```bash
set -euo pipefail
OFFLOAD=off; USERS=8
RUN_DIR="$ARTIFACT_ROOT/${CELL}/users${USERS}-${OFFLOAD}"
mkdir -p "$RUN_DIR"
fetch_trace "$TRACE_REL" "$RUN_DIR/input.jsonl"
start_server "$TP" "$MAX_MODEL_LEN" "$OFFLOAD"
aiperf profile \
  --model "$MODEL" \
  --url "http://127.0.0.1:${PORT}" \
  --endpoint-type chat \
  --streaming \
  --input-file "$RUN_DIR/input.jsonl" \
  --custom-dataset-type mooncake_trace \
  --fixed-schedule \
  --concurrency "$USERS" \
  --artifact-dir "$RUN_DIR" \
  --export-level raw \
  --ui simple \
  --tokenizer "$MODEL"
stop_server
```
## Full sweep
```bash
set -euo pipefail
for spec in \
  "code-8k 4 9472 mooncake/core/code_8k1k/isb1_core_code_8k1k.jsonl" \
  "chat-32k 4 34048 mooncake/extension_32k/chat_32k1k_qwen3.5/isb1_extension_32k_chat_32k1k_qwen3_5.jsonl" \
  "code-131k 8 132352 mooncake/extension_131k/code_131k1k_qwen3.5/isb1_extension_131k_vllm_code_131k1k_qwen3_5.jsonl"
do
  read -r CELL TP MAX_MODEL_LEN TRACE_REL <<<"$spec"
  fetch_trace "$TRACE_REL" "$ARTIFACT_ROOT/${CELL}.jsonl"
  for OFFLOAD in on off noprefix; do
    start_server "$TP" "$MAX_MODEL_LEN" "$OFFLOAD"
    for USERS in 1 2 4 8 16; do
      RUN_DIR="$ARTIFACT_ROOT/${CELL}/users${USERS}-${OFFLOAD}"
      mkdir -p "$RUN_DIR"
      cp "$ARTIFACT_ROOT/${CELL}.jsonl" "$RUN_DIR/input.jsonl"
      aiperf profile \
        --model "$MODEL" --url "http://127.0.0.1:${PORT}" \
        --endpoint-type chat --streaming \
        --input-file "$RUN_DIR/input.jsonl" \
        --custom-dataset-type mooncake_trace --fixed-schedule \
        --concurrency "$USERS" \
        --artifact-dir "$RUN_DIR" --export-level raw --ui simple \
        --tokenizer "$MODEL"
    done
    stop_server
  done
done
```
## Results + post back
- Per-run artifacts: `$ARTIFACT_ROOT/<cell>/users<users>-<offload>/`
- Main files: `profile_export*.json`, `profile_export*.jsonl`, `logs/`
- Tarball:
```bash
tar -C "$ARTIFACT_ROOT" -czf "$PWD/isb1-gmi-h200-$(basename "$ARTIFACT_ROOT").tgz" .
```
Post back the tarball plus instance shape, driver/CUDA version, docker image tag, model, and any failed `(cell, users, offload)` tuples. If you reconnect, rerun the full `Prep once` block before resuming.
