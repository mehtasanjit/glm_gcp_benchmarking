# GLM Model Family — Hosting Guide for 2x A100-SXM4-40GB

## Hardware Budget

| Resource | Value |
|----------|-------|
| GPUs | 2x NVIDIA A100-SXM4-40GB |
| Total VRAM | ~79 GB |
| Tensor Parallelism | `--tensor-parallel-size 2` |
| vLLM Version | 0.17.1 |
| CUDA | 11.8 (toolkit) / 12.9 (PyTorch runtime) |

---

## GLM Model Landscape (zai-org, formerly THUDM)

### Official Base Models

| Model | Total Params | Active Params | Architecture | Release |
|-------|-------------|---------------|--------------|---------|
| **GLM-5** | 753B | ~? | MoE (DSA) | Newest |
| **GLM-4.7** | 358B | ~? | MoE | Latest stable |
| **GLM-4.7-Flash** | 31B | ~? | MoE Lite | Smaller, fast |
| **GLM-4.6** | 357B | ~? | MoE | |
| **GLM-4.5** | 358B | 32B | MoE | |
| **GLM-4.5-Air** | 110B | 12B | MoE | Lightweight |
| **GLM-4-32B-0414** | ~32B | 32B | Dense | |
| **GLM-Z1-32B-0414** | 32.5B | 32.5B | Dense | Reasoning |
| **GLM-4-9B-0414** | 9.4B | 9.4B | Dense | |
| **GLM-4.1V-9B-Thinking** | 9B | 9B | Vision | |

> **MoE Note:** For MoE (Mixture of Experts) models, only a subset of parameters ("active params") are used per token, but **all expert weights must reside in GPU memory**. The **total parameter count** determines memory requirements, not just the active count.

### Official FP8 Variants (by zai-org)

| Model | Base | Notes |
|-------|------|-------|
| `zai-org/GLM-5-FP8` | GLM-5 (753B) | Far too large for 79 GB |
| `zai-org/GLM-4.7-FP8` | GLM-4.7 (358B) | ~358 GB FP8 → too large |
| `zai-org/GLM-4.6-FP8` | GLM-4.6 (357B) | ~357 GB FP8 → too large |
| `zai-org/GLM-4.5-Air-FP8` | GLM-4.5-Air (110B) | ~110 GB FP8 → too large |

---

## Memory Estimation Rules of Thumb

| Precision | Bytes/Param | 9B Model | 32B Model | 110B Model | 358B Model |
|-----------|------------|----------|-----------|------------|------------|
| FP16/BF16 | 2 | ~18 GB | ~64 GB | ~220 GB | ~716 GB |
| FP8 | 1 | ~9 GB | ~32 GB | ~110 GB | ~358 GB |
| INT8 | 1 | ~9 GB | ~32 GB | ~110 GB | ~358 GB |
| GPTQ/AWQ 4-bit | 0.5 | ~5 GB | ~16 GB | ~55 GB | ~179 GB |
| GPTQ Int4-Int8 Mix | ~0.75 | ~7 GB | ~24 GB | ~83 GB | ~269 GB |

> These are **weight-only** estimates. Add 10-30% overhead for KV cache, activations, and vLLM internals depending on context length and batch size.

---

## Feasibility Analysis for 79 GB (2x A100-40GB)

### ✅ TIER 1 — Will Comfortably Fit (Recommended)

| Model | Quantization | Est. Weight Size | Fits? | vLLM Support |
|-------|-------------|-----------------|-------|--------------|
| **GLM-4-9B-0414** | FP16 (no quant) | ~18 GB | ✅ Easily | ✅ Native |
| **GLM-4.1V-9B-Thinking** | FP16 (no quant) | ~18 GB | ✅ Easily | ✅ Native |
| **GLM-4-9B-0414** | GPTQ-Int4 | ~5 GB | ✅ Easily | ✅ |
| **GLM-4.7-Flash** | GPTQ-4bit | ~16 GB | ✅ Easily | ✅ |
| **GLM-4-32B-0414** | GPTQ-4bit | ~16 GB | ✅ Yes | ✅ |
| **GLM-Z1-32B-0414** | GPTQ-4bit | ~16 GB | ✅ Yes | ✅ |
| **GLM-4-32B-0414** | FP16 (no quant) | ~64 GB | ✅ Tight fit | ✅ Native |
| **GLM-Z1-32B-0414** | FP16 (no quant) | ~65 GB | ✅ Tight fit | ✅ Native |

### ⚠️ TIER 2 — Borderline / Needs Testing

| Model | Quantization | Est. Weight Size | Fits? | Notes |
|-------|-------------|-----------------|-------|-------|
| **GLM-4.5-Air** | GPTQ-Int4 | ~55 GB | ⚠️ Maybe | 110B total params; tight with KV cache |
| **GLM-4.5-Air** | GPTQ-Int4-Int8 Mix | ~83 GB | ❌ Too large | Exceeds 79 GB |
| **GLM-4.5-Air-FP8** | FP8 | ~110 GB | ❌ Too large | |

### ❌ TIER 3 — Will NOT Fit

| Model | Any Quantization | Reason |
|-------|-----------------|--------|
| **GLM-4.5** (358B) | Even 4-bit → ~179 GB | Way too large |
| **GLM-4.6** (357B) | Even 4-bit → ~179 GB | Way too large |
| **GLM-4.7** (358B) | Even 4-bit → ~179 GB | Way too large |
| **GLM-5** (753B) | Even 4-bit → ~377 GB | Way too large |

---

## Available Quantized Models on HuggingFace

| HuggingFace Repo | Base Model | Quantization | Downloads | vLLM Compat | Fits 79GB? |
|-------------------|-----------|-------------|-----------|-------------|------------|
| `FayeQuant/GLM-4.7-Flash-GPTQ-4bit` | GLM-4.7-Flash (31B) | GPTQ 4-bit | 10,317 | ✅ | ✅ ~16 GB |
| `mratsim/GLM-4-32B-0414.w4a16-gptq` | GLM-4-32B (32B) | GPTQ 4-bit | 42 | ✅ | ✅ ~16 GB |
| `model-scope/glm-4-9b-chat-GPTQ-Int4` | GLM-4-9B | GPTQ Int4 | 55 | ✅ | ✅ ~5 GB |
| `QuantTrio/GLM-4.5-Air-GPTQ-Int4-Int8Mix` | GLM-4.5-Air (110B) | GPTQ Int4/Int8 | 289 | ✅ | ❌ ~83 GB |
| `QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix` | GLM-4.7 (358B) | GPTQ Int4/Int8 | 235 | ✅ | ❌ ~269 GB |
| `QuantTrio/GLM-4.5-GPTQ-Int4-Int8Mix` | GLM-4.5 (358B) | GPTQ Int4/Int8 | 166 | ✅ | ❌ ~269 GB |
| `QuantTrio/GLM-4.6-GPTQ-Int4-Int8Mix` | GLM-4.6 (357B) | GPTQ Int4/Int8 | 17 | ✅ | ❌ ~269 GB |

---

## 🏆 Recommendations (Ranked)

### Rank 1: `GLM-4-32B-0414` — FP16 (No Quantization Needed)
- **Why:** 32B dense model at FP16 ≈ 64 GB weights. Fits in 79 GB with TP=2. No quality loss from quantization.
- **Best for:** General-purpose chat, code, instruction following.
- **Serve command:**
  ```bash
  vllm serve zai-org/GLM-4-32B-0414 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --port 8000
  ```

### Rank 2: `GLM-Z1-32B-0414` — FP16 (No Quantization Needed)
- **Why:** 32.5B dense reasoning model. Similar memory profile to GLM-4-32B. Stronger on reasoning/math tasks.
- **Best for:** Reasoning, chain-of-thought, math, complex analysis.
- **Serve command:**
  ```bash
  vllm serve zai-org/GLM-Z1-32B-0414 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --port 8000
  ```

### Rank 3: `GLM-4.7-Flash` — GPTQ 4-bit Quantized
- **Why:** 31B MoE-lite model quantized to ~16 GB. Most downloaded GLM quantized model. Leaves massive headroom for KV cache / larger batches.
- **Best for:** High-throughput inference, cost-effective serving with good quality.
- **Serve command:**
  ```bash
  vllm serve FayeQuant/GLM-4.7-Flash-GPTQ-4bit \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --quantization gptq \
      --port 8000
  ```

### Rank 4: `GLM-4-32B-0414` — GPTQ 4-bit Quantized
- **Why:** ~16 GB quantized. If you want to maximize KV cache / batch size, quantizing frees up ~48 GB for serving overhead.
- **Trade-off:** Some quality loss vs FP16, but much more room for concurrent requests.
- **Serve command:**
  ```bash
  vllm serve mratsim/GLM-4-32B-0414.w4a16-gptq \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --quantization gptq \
      --port 8000
  ```

### Rank 5: `GLM-4-9B-0414` — FP16 (No Quantization Needed)
- **Why:** Only ~18 GB at FP16. Fastest inference, lowest resource usage. Great for prototyping or latency-sensitive workloads.
- **Serve command:**
  ```bash
  vllm serve zai-org/GLM-4-9B-0414 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --port 8000
  ```

---

## vLLM Serve Notes

- **`--trust-remote-code`**: Required for GLM models (custom architecture code).
- **`--tensor-parallel-size 2`**: Splits model across both A100 GPUs.
- **`--quantization gptq`**: Required when loading GPTQ-quantized checkpoints.
- **Tool calling**: For GLM-4.7+ models, add `--tool-call-parser glm47 --enable-auto-tool-choice`.
- **Reasoning**: For GLM-4.5+ reasoning, add `--reasoning-parser glm45`.
- **MTP (Multi-Token Prediction)**: Some GLM models support speculative decoding via `--speculative-config.method mtp --speculative-config.num_speculative_tokens 1`.

---

## Decision Summary

| Your Goal | Recommended Model | Precision | ~VRAM Used |
|-----------|------------------|-----------|------------|
| Best quality, general purpose | GLM-4-32B-0414 | FP16 | ~64 GB |
| Best reasoning/math | GLM-Z1-32B-0414 | FP16 | ~65 GB |
| High throughput, good quality | GLM-4.7-Flash GPTQ-4bit | GPTQ 4-bit | ~16 GB |
| Max concurrent requests | GLM-4-32B-0414 GPTQ-4bit | GPTQ 4-bit | ~16 GB |
| Fastest / prototyping | GLM-4-9B-0414 | FP16 | ~18 GB |
| Vision + Thinking | GLM-4.1V-9B-Thinking | FP16 | ~18 GB |