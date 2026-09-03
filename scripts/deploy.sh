#!/usr/bin/env bash
# deploy.sh — Qwen3.8-27B (Unsloth NVFP4) single-machine one-command deploy on Dell Pro Max with GB10
# Usage: ./deploy.sh [--port 8000] [--models-dir ~/models]
# Does four things: preflight -> pull (image + weights, fully pinned) -> start -> liveness check. Idempotent: if already running, skips straight to the liveness check.
set -euo pipefail

PORT=8000; MODELS_DIR="$HOME/models"
while [ $# -gt 0 ]; do case "$1" in
  --port) PORT="$2"; shift 2;;
  --models-dir) MODELS_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

IMAGE="vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404"
HF_REPO="unsloth/Qwen3.8-27B-NVFP4"          # Unsloth Dynamic V3.0 public quantized release
MODEL_DIR="$MODELS_DIR/Qwen3.8-27B-NVFP4"
NAME="qwen38-27b"

say() { printf '\033[1m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. Preflight (this book's Pitfalls turned into machine checks) ──
[ "$(uname -m)" = "aarch64" ] || die "This recipe targets aarch64 (Dell Pro Max with GB10); current arch is $(uname -m). On x86 use the official default image line."
command -v docker >/dev/null || die "docker is required"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  say "Container $NAME already running, skipping to liveness check (idempotent)"; SKIP_START=1
elif docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  die "A stopped $NAME container exists. Reuse it with: docker start $NAME, or docker rm $NAME and rerun."
else
  SKIP_START=0
  lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "Port $PORT is taken (pitfall: a port conflict looks like 'the service never came up'). Run lsof -iTCP:$PORT to see who has it."
  avail_gb=$(free -g | awk '/^Mem:/{print $7}')
  [ "${avail_gb:-0}" -ge 30 ] || die "Available memory ${avail_gb}GB < 30GB. NVFP4 weights are ~20GB plus KV headroom; free memory first (pitfall: filling unified memory = whole-machine thrash)."
  df -BG "$MODELS_DIR" 2>/dev/null | awk 'NR==2{gsub("G","",$4); if($4<30) exit 1}' || die "Disk free space <30GB ($MODELS_DIR)"
fi

# ── 2. Pull (fully pinned) ──
if [ "$SKIP_START" = 0 ]; then
  say "Pulling image $IMAGE (note: must be the aarch64-suffixed line, pitfall #1)"
  docker pull "$IMAGE"
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    say "Downloading weights $HF_REPO -> $MODEL_DIR (~20GB, be patient)"
    command -v hf >/dev/null || die "Missing hf CLI. Install (ideally in an isolated environment): pip install -U huggingface_hub"
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  else
    say "Weights already at $MODEL_DIR, skipping download"
  fi

  # ── 3. Start (same params as production; --memory hard cap = insurance for pitfall #2) ──
  say "Starting $NAME @ :$PORT"
  docker run -d --name "$NAME" --gpus all --memory 90g \
    -p "$PORT:$PORT" -v "$MODEL_DIR:/models/Qwen3.8-27B-NVFP4" \
    "$IMAGE" \
    --model /models/Qwen3.8-27B-NVFP4 --served-model-name qwen38-27b \
    --port "$PORT" --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.70 --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
fi

# ── 4. Liveness check (a 200 doesn't count; only real inference + a thinking-off assertion counts) ──
say "Waiting for the service to become ready (cold start + compilation can take minutes)..."
for i in $(seq 1 120); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
  [ "$i" = 120 ] && die "Not ready after 20 minutes, check logs: docker logs $NAME"
done
say "Real inference assertion (includes thinking-off verification, pitfall #3)..."
RESP=$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"qwen38-27b","max_tokens":50,
  "chat_template_kwargs":{"enable_thinking":false},
  "messages":[{"role":"user","content":"Reply with one word: DEPLOY_OK"}]}')
echo "$RESP" | grep -q "DEPLOY_OK" || die "Inference assertion failed, response: $(echo "$RESP" | head -c 300)"
RT=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0))" 2>/dev/null || echo "?")
say "✅ Deploy complete: http://127.0.0.1:$PORT/v1 (thinking-off check reasoning_tokens=$RT, should be ≈0)"
say "Optional: 3-tier thinking proxy at scripts/thinking_proxy_3tier.py"
