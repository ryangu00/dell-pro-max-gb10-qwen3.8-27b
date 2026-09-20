# Qwen3.8-27B on a single Dell Pro Max with GB10 (vLLM NVFP4 + MTP + vision tower)

> Single-machine deployment of Qwen3.8-27B: NVFP4 quantization + MTP speculative decoding (decode +45~50%) + hot-plugged vision tower on the same endpoint + thinking split across three ports.
> This is our longest-running single-machine workhorse config (long in production), and the book with the broadest audience: one Dell Pro Max with GB10 is all you need.

## Hardware and versions

| Item | Spec/version |
|---|---|
| Machine | Dell Pro Max with GB10 ×1 (GB10 chip, 128GB unified memory, sm_121/aarch64) |
| Image | `vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404` (**official aarch64 line**; use whatever tag is pullable at release time, see pitfall #1) |
| Weights | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) (**Dynamic V3.0 quantization**, base=`Qwen/Qwen3.8-27B`, Apache-2.0) — no need to quantize yourself, pull and go |
| Vision | The base model's native VL tower; one endpoint serves both text and images (no second model needed, see pitfall #5) |

## Results at a glance (measured)

| Metric | Value |
|---|---|
| MTP speculative decoding | decode throughput +45~50% (MTP-3; short-output/code-heavy workloads, single-seat, temperature 0.3 — gains vary with workload shape, see benchmark notes) |
| vs DeepSeek V4 family on the same machine (our faceoff sample set) | code decode +37%; cold prefill 4.5×; honest refusals (says "I don't know" when it doesn't) markedly better on our samples |
| Thinking tiers | three ports: :8001 thinking off / :8002 low / :8003 xhigh (proxy-based tiering, zero client changes) |

## Quick start

**One-command deploy**: `scripts/deploy.sh` (preflight → pull image + Unsloth weights → start with production params → thinking-off inference assertion), or walk through it manually:

```bash
# 1. Image: must be on the aarch64 line (pitfall #1)
docker pull vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404

# 2. Full launch params from our production instance (taken from the container, copy verbatim):
#    vllm serve /models/Qwen3.8-27B-NVFP4 \
#      --port 8000 --served-model-name qwen38-27b \
#      --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096 \
#      --gpu-memory-utilization 0.40 \
#      --enable-prefix-caching \
#      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
#      --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
#      --enable-lora --lora-modules vl38=/adapter
#    Note: gpu-memory-utilization 0.40 is our conservative value for coexisting with other services;
#    on a dedicated machine you can raise it, but on GB10 unified memory we recommend ≤0.70 (pitfall #2);
#    add a --memory hard cap on the container
# 3. Thinking tiers: the 3-tier proxy source lives in this repo at scripts/thinking_proxy_3tier.py
#    :8001=off / :8002=low / :8003=xhigh, pure stdlib, run directly with python3
```

## Pitfalls (6, all first-hand incidents)

1. **First thing when switching vLLM versions: confirm the image tag is on the aarch64 line.** `vllm/vllm-openai:vX.Y.Z` defaults to x86; GB10 requires the `-aarch64-ubuntu2404` suffixed tag. Pull the wrong one and the symptoms are all over the map — it will burn half your day.
2. **Never serve a large model in bare BF16**: we served a 67GB BF16 weight directly and unified-memory thrash froze the whole machine until we had to pull the power cord. Iron rule: quantize first (~20GB-class after NVFP4), add a `--memory` hard cap on the container, and keep `gpu_memory_utilization` ≤0.70 on GB10.
3. **The ground truth for turning thinking off is the chat template**: for the Qwen family, `chat_template_kwargs: {"enable_thinking": false}` is the official template key and actually takes effect; per-client fields (`reasoning`/`thinking`) are not necessarily honored by this stack. To verify: check that `usage.completion_tokens_details.reasoning_tokens` is 0 — don't just trust the response getting faster.
4. **Be careful with SGLang's model-specific tags**: we tried SGLang's Qwen-specific image line, hit three compatibility pitfalls in a row, and went back to vLLM. Not saying SGLang is bad — "model-specific tags" just can't keep maintenance pace with model iteration.
5. **vLLM silently ignores LoRA on the vision tower** (observed on the v0.24 line; verify yourself on newer versions): attach a LoRA to the vision tower and vLLM neither errors nor applies it — silently swallowed; text LoRA works fine. To verify: same image, same question, one request with the LoRA and one without, then `diff` the outputs — identical output means it was swallowed.
6. **Vision benchmarks across measurement setups are not directly comparable**: different stacks preprocess images differently (aspect-ratio-preserving vs square resize), so the same model's vision scores on two stacks are not comparable. Comparative tests must pin the preprocessing methodology.

## Benchmark notes

- The Qwen family defaults to thinking on with high effort; for quality comparisons, measure with thinking on and off separately (standard methodology for this series, see the DSV4F book's BENCHMARKS).
- MTP gains depend on workload shape: short-output/code-completion workloads gain the most; long reasoning outputs gain less. Measure with your real workload — don't just trust our +45~50%.

## When to pick this setup

✅ You have one Dell Pro Max with GB10 and want an all-round workhorse (text + vision + fast); ✅ agent backend (the honest-refusal trait suits toolchains)
❌ Need 1M context / multi-machine concurrency → the DSV4F dual-machine book in this series; ❌ chasing peak single-shot quality → go to a larger-parameter model

---
*RyanAI Lab · All numbers measured on our resident environment. Updated 2026-09. Issues welcome.*

## Update 2026-09-20: serving-tuning rounds and what replaced this setup

Four serving-side tuning rounds on this exact configuration were each measured and rolled back: fp8 KV cache with the `triton_attn` backend plus a raised `--max-num-batched-tokens` budget took cold prefill on a 28K-token prompt from 35.6 s to 61.1 s (+72% time); the raised batch budget alone took it to 47.5 s (+33% time); MTP k=5 traded +9.8% code decode for −9.7% Chinese prose; MTP k=7 cost −27% on Chinese prose and −11% KV pool. The numbers and the rollback reasoning are in `dell-pro-max-gb10-qwen3.8-27b-mtp-k-sweep` (github.com/ryangu00/). This single-node setup was our workhorse until early September 2026; production then moved to a two-node engine (documented in `dell-pro-max-gb10-deepseek-v4-flash-vision-exp` and, after the later switches, `dell-pro-max-gb10-vllm-stack-ab` and `dell-pro-max-gb10-qwen3.8-flash-next-1m-context`).
