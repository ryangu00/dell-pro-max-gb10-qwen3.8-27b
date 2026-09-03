#!/usr/bin/env bash
# deploy.sh — Qwen3.8-27B (Unsloth NVFP4) 单机一键部署 on Dell Pro Max with GB10
# 用法: ./deploy.sh [--port 8000] [--models-dir ~/models]
# 做四件事:预检 → 拉取(镜像+权重,全 pin) → 启动 → 验活。幂等:已在跑则直接验活。
set -euo pipefail

PORT=8000; MODELS_DIR="$HOME/models"
while [ $# -gt 0 ]; do case "$1" in
  --port) PORT="$2"; shift 2;;
  --models-dir) MODELS_DIR="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

IMAGE="vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404"
HF_REPO="unsloth/Qwen3.8-27B-NVFP4"          # Unsloth Dynamic V3.0 公开量化版
MODEL_DIR="$MODELS_DIR/Qwen3.8-27B-NVFP4"
NAME="qwen38-27b"

say() { printf '\033[1m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[deploy] FAIL:\033[0m %s\n' "$*" >&2; exit 1; }

# ── 1. 预检(把本书避坑清单变成机器检查) ──
[ "$(uname -m)" = "aarch64" ] || die "本配方面向 aarch64(Dell Pro Max with GB10);当前 $(uname -m)。x86 请换官方默认镜像线。"
command -v docker >/dev/null || die "需要 docker"
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  say "容器 $NAME 已在跑,跳到验活(幂等)"; SKIP_START=1
elif docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  die "存在已停止的 $NAME 容器。docker start $NAME 复用,或 docker rm $NAME 后重跑。"
else
  SKIP_START=0
  lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && die "端口 $PORT 被占(坑:端口冲突像'服务没起来')。lsof -iTCP:$PORT 看是谁。"
  avail_gb=$(free -g | awk '/^Mem:/{print $7}')
  [ "${avail_gb:-0}" -ge 30 ] || die "可用内存 ${avail_gb}GB < 30GB。NVFP4 权重 ~20GB+KV 需余量;先清场(坑:统一内存吃满=整机 thrash)。"
  df -BG "$MODELS_DIR" 2>/dev/null | awk 'NR==2{gsub("G","",$4); if($4<30) exit 1}' || die "磁盘余量 <30GB($MODELS_DIR)"
fi

# ── 2. 拉取(全 pin) ──
if [ "$SKIP_START" = 0 ]; then
  say "拉镜像 $IMAGE(注意:必须 aarch64 后缀线,坑 #1)"
  docker pull "$IMAGE"
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    say "下载权重 $HF_REPO → $MODEL_DIR(~20GB,耐心)"
    command -v hf >/dev/null || die "缺 hf CLI。安装(建议独立环境): pip install -U huggingface_hub"
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  else
    say "权重已在 $MODEL_DIR,跳过下载"
  fi

  # ── 3. 启动(生产同款参数;--memory 硬 cap=坑 #2 的保险) ──
  say "启动 $NAME @ :$PORT"
  docker run -d --name "$NAME" --gpus all --memory 90g \
    -p "$PORT:$PORT" -v "$MODEL_DIR:/models/Qwen3.8-27B-NVFP4" \
    "$IMAGE" \
    --model /models/Qwen3.8-27B-NVFP4 --served-model-name qwen38-27b \
    --port "$PORT" --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.70 --enable-prefix-caching \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
fi

# ── 4. 验活(200 不算数,真实推理+关思考断言才算) ──
say "等服务就绪(冷启动+编译可达数分钟)..."
for i in $(seq 1 120); do
  curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 10
  [ "$i" = 120 ] && die "20 分钟未就绪,看日志: docker logs $NAME"
done
say "真实推理断言(含关思考验证,坑 #3)..."
RESP=$(curl -s -m 60 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
  "model":"qwen38-27b","max_tokens":50,
  "chat_template_kwargs":{"enable_thinking":false},
  "messages":[{"role":"user","content":"回复一个词:DEPLOY_OK"}]}')
echo "$RESP" | grep -q "DEPLOY_OK" || die "推理断言失败,响应: $(echo "$RESP" | head -c 300)"
RT=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens',0))" 2>/dev/null || echo "?")
say "✅ 部署完成: http://127.0.0.1:$PORT/v1 (关思考验证 reasoning_tokens=$RT,应≈0)"
say "可选:三档 thinking proxy 见 scripts/thinking_proxy_3tier.py"
