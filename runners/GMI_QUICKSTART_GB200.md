# GMI Cloud GB200 ISB-1 quickstart
> Manual GMI Cloud path only. No Slurm, enroot, squash files, or GitHub runner labels.
## What you need
- GMI Cloud GB200: 1x node is enough; 2x lets you split code-8k/chat-32k from code-131k
- SSH access, read-only HF token, ~30 minutes
- This branch checked out on the box
## Provision + SSH
Create 1-2 GB200 instances in GMI Cloud, note the public IP(s), then:
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
- Model from shipped B200 FP4 DSR1 configs: `nvidia/DeepSeek-R1-0528-FP4-V2`
- Pinned vLLM image: `vllm/vllm-openai:v0.18.0`
- Corpus alias: `hf_wchen22--isb1-cc-traces`
- Manual offload mapping here: `on` = `--cpu-offload-gb 60`; `off` = no offload flag; `noprefix` = offload off + `--no-enable-prefix-caching`
- Default TP picks below are conservative for one-node manual runs; the higher-width alternates mirror the B200 FP4 ISB-1 cells
## Prep once
```bash
export MODEL='nvidia/DeepSeek-R1-0528-FP4-V2'
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
  [[ "$offload" == 'on' ]] && offload_flag='--cpu-offload-gb 60'
  docker rm -f dsr1-gb200 >/dev/null 2>&1 || true
  docker run -d --name dsr1-gb200 --gpus all --ipc=host --network host \
    -e HF_TOKEN -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$IMAGE" "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size "$tp" \
    --enable-expert-parallel \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --max-model-len "$max_model_len" \
    $prefix_flag $offload_flag
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null && return 0
    sleep 5
  done
  docker logs --tail=200 dsr1-gb200 || true
  return 1
}
stop_server () { docker rm -f dsr1-gb200 >/dev/null 2>&1 || true; }
```
## Per-cell settings
```bash
# code-8k   -> default tp2, alternate tp4, max-model-len 9472
CELL=code-8k; TP=2; MAX_MODEL_LEN=9472
TRACE_REL='mooncake/core/code_8k1k/isb1_core_code_8k1k.jsonl'
# chat-32k  -> default tp2, alternate tp4, max-model-len 34048
CELL=chat-32k; TP=2; MAX_MODEL_LEN=34048
TRACE_REL='mooncake/extension_32k/chat_32k1k/isb1_extension_32k_chat_32k1k.jsonl'
# code-131k -> default tp4, alternate tp8, max-model-len 132352
CELL=code-131k; TP=4; MAX_MODEL_LEN=132352
TRACE_REL='mooncake/extension_131k/code_131k1k/isb1_extension_131k_vllm_code_131k1k.jsonl'
```
If you want the wider topology from the shipped B200 FP4 reference cells, rerun with `TP=4` for code-8k/chat-32k and `TP=8` for code-131k.
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
  "code-8k 2 9472 mooncake/core/code_8k1k/isb1_core_code_8k1k.jsonl" \
  "chat-32k 2 34048 mooncake/extension_32k/chat_32k1k/isb1_extension_32k_chat_32k1k.jsonl" \
  "code-131k 4 132352 mooncake/extension_131k/code_131k1k/isb1_extension_131k_vllm_code_131k1k.jsonl"
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
tar -C "$ARTIFACT_ROOT" -czf "$PWD/isb1-gmi-gb200-$(basename "$ARTIFACT_ROOT").tgz" .
```
Post back the tarball plus instance shape, driver/CUDA version, docker image tag, model, chosen TP layout, and any failed `(cell, users, offload)` tuples. If you reconnect, rerun the full `Prep once` block before resuming.
