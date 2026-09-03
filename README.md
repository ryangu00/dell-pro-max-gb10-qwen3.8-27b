# Qwen3.8-27B on Dell Pro Max with GB10 单机 (vLLM NVFP4 + MTP + 视觉塔)

> 单机部署 Qwen3.8-27B:NVFP4 量化 + MTP 投机解码(decode +45~50%) + 视觉塔同端点热插 + thinking 三档分端口。
> 这是我们跑了最久的单机主力配置(曾长期生产),受众最广的一本:一台Dell Pro Max with GB10 就够。

## 硬件与版本

| 项 | 规格/版本 |
|---|---|
| 机器 | Dell Pro Max with GB10 ×1(GB10 芯片,128GB 统一内存,sm_121/aarch64) |
| 镜像 | `vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404`(**官方 aarch64 线**,以发布时可拉 tag 为准,见坑 #1) |
| 权重 | [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)(**Dynamic V3.0 量化**,base=`Qwen/Qwen3.8-27B`,Apache-2.0)——无需自己量化,拉下来就能用 |
| 视觉 | 同 base 的原生 VL 塔,一个端点同时吃文本+图(无需第二模型,见坑 #5) |

## 结果速览(实测)

| 指标 | 数值 |
|---|---|
| MTP 投机解码 | decode 吞吐 +45~50%(MTP-3;短输出/代码类负载,单座,temperature 0.3——收益随负载形状浮动,见评测提示) |
| 对比同机 DeepSeek V4 系(我们的 faceoff 样本集) | code decode +37%;冷 prefill 4.5×;诚实拒答(不知道就说不知道)在我们的样本中显著更好 |
| thinking 分档 | 三端口:8001 关思考 / :8002 low / :8003 xhigh(代理分档,零客户端改造) |

## 快速开始

**一键部署**:`scripts/deploy.sh`(预检→拉镜像+Unsloth 权重→生产参数启动→关思考推理断言),或按下面手动走:

```bash
# 1. 镜像:必须选 aarch64 线(坑 #1)
docker pull vllm/vllm-openai:v0.27.1-aarch64-ubuntu2404

# 2. 我们生产实例的完整启动参数(取自容器,可整体照抄):
#    vllm serve /models/Qwen3.8-27B-NVFP4 \
#      --port 8000 --served-model-name qwen38-27b \
#      --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096 \
#      --gpu-memory-utilization 0.40 \
#      --enable-prefix-caching \
#      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
#      --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
#      --enable-lora --lora-modules vl38=/adapter
#    说明:gpu-memory-utilization 0.40 是我们与其他服务共存的保守值;
#    独占机器可提高,但 GB10 统一内存建议 ≤0.70(坑 #2);容器加 --memory 硬 cap

# 3. thinking 分档:三档 proxy 源码就在本 repo scripts/thinking_proxy_3tier.py
#    :8001=off / :8002=low / :8003=xhigh,纯 stdlib,python3 直接跑
```

## 避坑清单(6 个,均为一手事故)

1. **换 vLLM 版本第一件事:确认镜像 tag 在 aarch64 线**。`vllm/vllm-openai:vX.Y.Z` 默认是 x86;GB10 必须用 `-aarch64-ubuntu2404` 后缀 tag。拉错了症状五花八门,浪费你半天。
2. **永远不要裸 BF16 serve 大模型**:我们把一个 67GB BF16 权重直接 serve,统一内存 thrash 整机僵死到只能拔电。铁律:先量化(NVFP4 后 ~20GB 级),容器加 `--memory` 硬 cap,`gpu_memory_utilization` 在 GB10 上建议 ≤0.70。
3. **关思考的 ground truth 是 chat template**:对 Qwen 系,`chat_template_kwargs: {"enable_thinking": false}` 是官方模板键,真实生效;各家客户端字段(`reasoning`/`thinking`)不一定被这个栈认。验证方法:看 `usage.completion_tokens_details.reasoning_tokens` 是否为 0,别只看响应变快。
4. **SGLang 的模型专用 tag 慎用**:我们试过 SGLang 的 Qwen 专用镜像线,连续踩三个兼容坑后放弃回 vLLM。不是说 SGLang 不好,是"模型专用 tag"维护节奏跟不上模型迭代。
5. **视觉塔 LoRA 会被 vLLM 静默 ignore**(我们在 v0.24 线观测,新版本请自验):给视觉塔挂 LoRA,vLLM 不报错、也不生效——静默吞掉;文本 LoRA 正常。验证命令:同图同题、挂/不挂 LoRA 各打一发,`diff` 输出——完全一致即被吞。
6. **跨口径视觉评测不可直比**:不同栈对图片的预处理不同(保长宽比 vs 方形 resize),同一模型在两个栈上的视觉分数没有可比性。对比测试必须固定预处理口径。

## 评测提示

- Qwen 系默认思考开且 effort 高;做质量对比时开/关思考各测一档(本系列通用方法论,详见 DSV4F 篇 BENCHMARKS)。
- MTP 收益与负载形状相关:短输出/代码补全收益最大;长推理输出收益递减。用你的真实负载测,别只信我们的 +45~50%。

## 何时选这套方案

✅ 只有一台Dell Pro Max with GB10 想要全能主力(文本+视觉+快);✅ agent 后端(诚实拒答特性适合工具链)
❌ 需要 1M 上下文/多机并发 → 本系列 DSV4F 双机篇;❌ 追求极限单发质量 → 上更大参数模型

---
*RyanAI Lab · 数字来自我们的常驻环境实测,更新于 2026-09。欢迎 issue 反馈。*
