# Benchmarking GLM-4.7 on Vertex AI Model-as-a-Service (MaaS)

## 1. Problem Statement

A customer is evaluating **GLM-4.7** (ZhipuAI's 358B parameter Mixture-of-Experts model) hosted on **Google Cloud Vertex AI** for production use. The objective is to establish **factual, reproducible performance baselines** covering:

- **Latency** — end-to-end, TTFT (reasoning and content), and inter-chunk delivery cadence
- **Throughput** — tokens/second under different configurations
- **Accuracy** — multi-hop reasoning quality (EM, F1) on a standard QA benchmark
- **Thinking mode impact** — quantifying the latency/accuracy trade-off of reasoning ON vs OFF
- **Deployment comparison** — MaaS (pay-per-token API) vs self-hosted (dedicated GPU endpoint)

This document covers the **MaaS benchmarking** approach — using Vertex AI's managed GLM-4.7 API endpoint.

---

## 2. Model Under Test

| Property | Value |
|---|---|
| **Model** | `zai-org/glm-4.7-maas` |
| **Architecture** | Mixture of Experts (MoE), 358B total parameters |
| **Quantization** | FP8 (for self-hosted variant: `GLM-4.7-FP8`) |
| **Thinking Mode** | Supports `reasoning_content` tokens before answer |
| **API Endpoint** | `https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global/endpoints/openapi` |
| **Protocol** | OpenAI-compatible chat completions API |

### GLM-4.7 Thinking Mode

GLM-4.7 supports a **thinking/reasoning mode** where the model generates internal reasoning tokens (`reasoning_content`) before producing the final answer (`content`). This is similar to OpenAI's o1/o3 reasoning approach.

- **Thinking ON (default):** Model reasons internally first, then produces the answer. Higher accuracy but higher latency.
- **Thinking OFF (`--no-thinking`):** Model skips reasoning and answers directly. Lower latency but potentially lower accuracy. Disabled via the Vertex AI MaaS parameter `"chat_template_kwargs": {"enable_thinking": false}` (see [Vertex AI Thinking docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/thinking)).

---

## 3. Evaluation Dataset: HotpotQA Distractor

### Why HotpotQA?

[HotpotQA](https://hotpotqa.github.io/) is a standard benchmark for **multi-hop question answering**. It tests a model's ability to:

1. **Reason across multiple documents** — answers require synthesizing information from 2+ paragraphs
2. **Ignore distractors** — the "distractor" setting includes irrelevant context paragraphs alongside relevant ones
3. **Produce concise answers** — ground truth answers are typically short (1-5 words)

### Dataset Details

| Property | Value |
|---|---|
| **Dataset** | `hotpotqa/hotpot_qa` (HuggingFace) |
| **Configuration** | `distractor` |
| **Split** | `validation` |
| **Total Samples** | 7,405 |
| **Question Type** | Multi-hop reasoning |
| **Answer Format** | Short text (1-5 words typically) |

### Prompt Format

Each sample is formatted as:

```
Answer the following question based on the provided context.
Give a short, direct answer.

Context:
**Title 1:** Paragraph text from document 1...

**Title 2:** Paragraph text from document 2...

[... 10 context paragraphs total, including distractors ...]

Question: {question}

Answer:
```

---

## 4. Evaluation Metrics

### 4.1 Quality Metrics

#### Exact Match (EM)

- **Binary** (0 or 1): Does the model's normalized answer exactly equal the ground truth?
- **Normalization:** Both prediction and ground truth are lowercased, articles (a/an/the) removed, punctuation stripped, whitespace collapsed
- **Strict:** No partial credit. "Terry Richardson" vs "Terry Richardson is older" → EM=0

#### F1 Score

- **Token-level overlap** between prediction and ground truth (0.0 to 1.0)
- Treats both as bags of words after normalization
- Computes precision (correct predicted tokens / total predicted) and recall (correct predicted tokens / total ground truth tokens)
- F1 = harmonic mean of precision and recall
- **More forgiving** than EM — partial credit for partial matches
- Example: Ground truth = "Terry Richardson", prediction = "Terry Richardson is older" → F1 ≈ 0.67

### 4.2 Latency Metrics

#### TTFT — Time to First Thinking Token

- **What:** Time from sending the request to receiving the first `reasoning_content` chunk in the stream
- **When it appears:** Only when thinking mode is ON and the model emits reasoning tokens
- **How measured:** In streaming mode, the OpenAI SDK delivers delta chunks. Each chunk's `delta` object may contain a `reasoning_content` field (the model's internal reasoning). We record `time.perf_counter()` at the moment the first non-empty `reasoning_content` is received.
- **Why it matters:** This measures the model's **prompt processing time** (prefill phase) — how quickly the model begins any form of generation after receiving the input

```python
for chunk in stream:
    t_now = time.perf_counter()
    if chunk.choices:
        delta = chunk.choices[0].delta
        # First thinking token
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            if ttft_thinking_s is None:
                ttft_thinking_s = t_now - t_start  # ← TTFT Thinking
```

#### TTFT — Time to First Content Token

- **What:** Time from sending the request to receiving the first `content` chunk in the stream
- **When thinking is ON:** This includes the entire thinking phase (prefill + all reasoning tokens) plus the transition to content generation. This is significantly higher than TTFT-Thinking because the model must finish its internal reasoning first.
- **When thinking is OFF:** This equals the standard TTFT — just the prefill/prompt processing time
- **How measured:** Same streaming approach, but tracking `delta.content` instead of `delta.reasoning_content`

```python
        # First content token
        if delta.content:
            if ttft_s is None:
                ttft_s = t_now - t_start  # ← TTFT Content
```

#### Timeline Visualization (Thinking ON)

```
Request sent
    │
    ├── [prefill phase] ──→ First reasoning_content token (TTFT-Thinking)
    │                            │
    │                            ├── [reasoning tokens stream] ──→ ...
    │                            │
    │                            └── Last reasoning token
    │                                    │
    │                                    └── First content token (TTFT-Content)
    │                                            │
    │                                            ├── [content tokens stream] ──→ ...
    │                                            │
    │                                            └── Last content token
    │
    └── Total Latency (end-to-end)
```

#### Inter-Token Latency (ITL)

- **What:** Time between consecutive content tokens
- **How measured:** Record timestamps of every content chunk, compute differences between consecutive timestamps
- **Reported as:** Average, P50, P90, P99 across all tokens from all samples
- **Why it matters:** Measures the consistency of token generation speed. High P99 ITL indicates occasional stalls.

### 4.3 Throughput Metrics

#### Tokens/Second (Generation Phase)

- **What:** `completion_tokens / generation_time` where `generation_time = total_latency - TTFT`
- **Measures:** Pure decoding speed after the model starts producing content
- **Excludes:** Prefill time and thinking time

#### Tokens/Second (Overall)

- **What:** `completion_tokens / total_latency`
- **Measures:** Effective throughput including all overhead
- **Includes:** Prefill, thinking, and generation

### 4.4 Percentiles Reported

For TTFT and Throughput, we report: **P50, P75, P90, P95, P99, Min, Max**

This provides a complete picture of the latency distribution — P50 shows typical performance while P99 shows worst-case tail latency.

---

## 5. Benchmarking Script

### Script: `benchmark_glm_4_7_maas_hotpot_qa_distractor.py`

#### Key Features

- **Streaming mode** (default): Uses OpenAI SDK streaming for TTFT/ITL measurement
- **Non-streaming mode** (`--no-stream`): Single request/response for simpler throughput measurement
- **Thinking control**: `--thinking` (default ON) / `--no-thinking`
- **Concurrency**: `--num-workers N` for parallel request execution
- **Rate limiting**: `--request-delay` seconds between requests
- **Cost estimation**: Approximate USD cost based on token counts

#### Usage Examples

```bash
# Basic benchmark (50 samples, streaming, thinking ON)
python benchmark_glm_4_7_maas_hotpot_qa_distractor.py --num-samples 50

# Full benchmark with output
python benchmark_glm_4_7_maas_hotpot_qa_distractor.py \
    --num-samples 2048 \
    --max-tokens 2048 \
    --output results_maas_thinking_on.json

# Thinking OFF for latency comparison
python benchmark_glm_4_7_maas_hotpot_qa_distractor.py \
    --num-samples 2048 \
    --max-tokens 2048 \
    --no-thinking \
    --output results_maas_thinking_off.json

# Concurrent execution (4 workers, 0.1s stagger)
python benchmark_glm_4_7_maas_hotpot_qa_distractor.py \
    --num-samples 200 \
    --num-workers 4 \
    --request-delay 0.1

# Non-streaming mode
python benchmark_glm_4_7_maas_hotpot_qa_distractor.py \
    --num-samples 50 \
    --no-stream

# Load from local data file
python benchmark_glm_4_7_maas_hotpot_qa_distractor.py \
    --data-file ./data/hotpotqa/hotpotqa_distractor_validation.jsonl \
    --num-samples 100
```

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--num-samples` | 50 | Number of HotpotQA samples to benchmark |
| `--data-file` | None | Path to local JSONL data file (default: load from HuggingFace) |
| `--output` | None | Output JSON file path for detailed results |
| `--no-stream` | False | Use non-streaming mode (no TTFT/ITL measurement) |
| `--thinking` | ON | Enable thinking/reasoning mode |
| `--no-thinking` | — | Disable thinking/reasoning mode |
| `--max-tokens` | 512 | Maximum output tokens per request |
| `--temperature` | 0.1 | Sampling temperature |
| `--num-workers` | 1 | Number of concurrent workers (1 = sequential) |
| `--request-delay` | 0.0 | Delay in seconds between requests |
| `--project` | default-project-alpha-1 | GCP project ID |

#### Output Format

The script produces:

1. **Real-time per-sample progress** with EM, F1, latency, tok/s, TTFT
2. **Aggregate statistics** printed to console
3. **JSON output** (when `--output` is specified) containing:
   - `config`: Benchmark configuration
   - `aggregate`: All aggregate statistics
   - `per_sample`: Per-sample results with predictions, metrics, and latency data

---

## 6. Architecture: MaaS vs Self-Hosted

| Aspect | MaaS | Self-Hosted (FP8) |
|---|---|---|
| **Endpoint** | Global managed API | Dedicated Vertex AI endpoint |
| **Model** | `zai-org/glm-4.7-maas` | `zai-org/GLM-4.7-FP8` |
| **Hardware** | Google-managed | 8x NVIDIA RTX PRO 6000 or 8x H100 |
| **Pricing** | Pay-per-token | Pay-per-GPU-hour |
| **Scaling** | Auto-scaled | Manual replica configuration |
| **Latency Control** | Limited | Full (vLLM optimizations) |
| **Benchmark Script** | `benchmark_glm_4_7_maas_hotpot_qa_distractor.py` | `benchmark_glm_4_7_fp8_hosted_hotpot_qa.py` |

---

## 7. Results

All runs used `--max-tokens 2048`, streaming mode, and the HotpotQA distractor validation set.

### 7.1 Summary Comparison (All Runs)

> Runs marked with **(exp)** used `--explanation-word-limit 512` for throughput measurement. EM/F1 are near 0% in those runs because the long explanation text is compared against short ground truth answers — accuracy is not meaningful in explanation mode.

| Metric | Think OFF, 1w (exp) | Think OFF, 4w (exp) | Think ON, 1w (exp) | Think ON, 1w | Think ON, 4w |
|---|---|---|---|---|---|
| **Explanation** | 512 words | 512 words | 512 words | None | None |
| **Request Delay** | 2.0s | 2.0s | 2.0s | 0.0s | 0.0s |
| **Successful / Total** | 872 / 2048 | 1595 / 2048 | 294 / 768 | 213 / 768 | **768 / 768** |
| **Success Rate** | 42.6% | 77.9% | 38.3% | 27.7% | **100%** |
| **EM** | 0.00% | 0.00% | 0.00% | **57.28%** | **58.72%** |
| **F1** | 2.38% | 2.41% | 0.13% | **74.36%** | **76.31%** |
| **Latency avg** | 2.12s | 2.11s | 9.80s | 3.60s | 3.30s |
| **Latency P50** | 2.02s | 2.05s | 9.60s | 2.82s | 2.67s |
| **Latency P90** | 2.69s | 2.71s | 11.72s | 7.05s | 4.71s |
| **Latency P99** | 4.41s | 3.40s | 13.32s | 10.60s | 11.32s |
| **TTFT First Visible avg** | 505ms | 521ms | 481ms | 507ms | 478ms |
| **TTFT First Visible P99** | 1165ms | 1016ms | 710ms | 879ms | 900ms |
| **TTFT Reasoning avg** | — | — | 481ms | 507ms | 478ms |
| **TTFT Answer avg** | 505ms | 521ms | 8634ms | 3031ms | 2899ms |
| **TTFT Answer P90** | 563ms | 605ms | 11283ms | 4233ms | 4009ms |
| **Throughput avg** | 194.6 | 195.9 | 223.4 | 228.8 | 230.6 |
| **Throughput P50** | 194.5 | 195.0 | 223.7 | 225.2 | 229.1 |
| **Throughput P90** | 237.9 | 239.5 | 259.3 | 280.0 | 278.4 |
| **ICL avg** | 15.7ms | 15.6ms | 14.6ms | 15.1ms | 15.7ms |
| **ICL P99** | 241.8ms | 237.5ms | 229.1ms | 238.8ms | 234.7ms |
| **Avg completion tokens** | 310 | 307 | 2047 | 682 | 626 |
| **Avg prompt tokens** | 1442 | 1440 | 1446 | 1422 | 1423 |
| **Wall time** | 5963s | 4528s | 4426s | 773s | **633s** |

### 7.2 Thinking OFF — Detailed (Streaming, 1 Worker)

| Metric | avg | P50 | P75 | P90 | P95 | P99 | min | max |
|---|---|---|---|---|---|---|---|---|
| **Latency (s)** | 2.12 | 2.02 | — | 2.69 | — | 4.41 | 1.12 | 11.67 |
| **TTFT Answer (ms)** | 505 | 471 | — | 563 | — | 1165 | — | — |
| **Throughput (tok/s)** | 194.6 | 194.5 | — | 237.9 | — | 263.5 | — | — |
| **ICL (ms)** | 15.7 | 0.5 | — | 18.5 | — | 241.8 | — | — |

- **872 / 2048 successful** (1176 failed due to 401 token expiry — pre-fix)
- Avg completion tokens: 310 (explanation mode)
- No reasoning chunks (thinking OFF confirmed)

### 7.3 Thinking OFF — Detailed (Streaming, 4 Workers)

| Metric | avg | P50 | P90 | P99 |
|---|---|---|---|---|
| **Latency (s)** | 2.11 | 2.05 | 2.71 | 3.40 |
| **TTFT Answer (ms)** | 521 | 499 | 605 | 1016 |
| **Throughput (tok/s)** | 195.9 | 195.0 | 239.5 | 266.8 |
| **ICL (ms)** | 15.6 | 0.5 | 18.6 | 237.5 |

- **1595 / 2048 successful** (453 failed — higher success rate from shorter wall time)
- Concurrency did not improve per-request latency (2.11s ≈ 2.12s)
- Concurrency improved total benchmark time: 75 min vs 99 min

### 7.4 Thinking ON + Explanation — Detailed (Streaming, 1 Worker)

| Metric | avg | P50 | P90 | P99 |
|---|---|---|---|---|
| **Latency (s)** | 9.80 | 9.60 | 11.72 | 13.32 |
| **TTFT Reasoning (ms)** | 481 | 464 | 566 | 710 |
| **TTFT Answer (ms)** | 8634 | 8005 | 11283 | 11686 |
| **Throughput (tok/s)** | 223.4 | 223.7 | 259.3 | 276.5 |
| **ICL (ms)** | 14.6 | 10.2 | 21.6 | 229.1 |

- **294 / 768 successful** (474 failed due to 401 token expiry)
- `--explanation-word-limit 512` — EM/F1 not meaningful
- Avg completion tokens: 2047 (nearly hitting max_tokens=2048 — reasoning consumes most of the budget)
- The ~8s gap between TTFT Reasoning and TTFT Answer = internal reasoning time

### 7.5 Thinking ON, No Explanation — Detailed (Streaming, 1 Worker)

| Metric | avg | P50 | P90 | P99 |
|---|---|---|---|---|
| **Latency (s)** | 3.60 | 2.82 | 7.05 | 10.60 |
| **TTFT Reasoning (ms)** | 507 | 471 | 583 | 879 |
| **TTFT Answer (ms)** | 3031 | 2733 | 4233 | 7576 |
| **Throughput (tok/s)** | 228.8 | 225.2 | 280.0 | 338.2 |
| **ICL (ms)** | 15.1 | 0.6 | 21.0 | 238.8 |

- **213 / 768 successful** (555 failed due to 401 token expiry — no delay, pre-fix)
- **EM: 57.28% | F1: 74.36%** — meaningful accuracy with short answers
- Avg completion tokens: 682 (much less than with explanation — model answers concisely)
- Latency much lower (3.6s vs 9.8s) because less reasoning needed for short answers

### 7.6 Thinking ON, No Explanation — Detailed (Streaming, 4 Workers)

| Metric | avg | P50 | P90 | P99 |
|---|---|---|---|---|
| **Latency (s)** | 3.30 | 2.67 | 4.71 | 11.32 |
| **TTFT Reasoning (ms)** | 478 | 456 | 563 | 900 |
| **TTFT Answer (ms)** | 2899 | 2598 | 4009 | 8658 |
| **Throughput (tok/s)** | 230.6 | 229.1 | 278.4 | 316.4 |
| **ICL (ms)** | 15.7 | 1.0 | 21.7 | 234.7 |

- **768 / 768 successful (100%)** — no failures (0 delay, 4 workers, fast total run)
- **EM: 58.72% | F1: 76.31%** — highest accuracy across all runs
- Avg completion tokens: 626 (lower than 1-worker due to natural variance)
- Per-request latency slightly lower with 4 workers (3.30s vs 3.60s)
- Total wall time: **633s (~10.5 min)** — the fastest run

---

## 8. Key Observations

### Thinking Mode Impact on Latency

- **Thinking ON adds ~7.7s** to total latency (9.80s vs 2.12s) — a **4.6x increase**
- The additional latency is almost entirely from the reasoning phase: TTFT Answer (8.6s) - TTFT Reasoning (0.5s) = **8.1s of internal reasoning**
- **Prefill latency is consistent** at ~480-520ms regardless of thinking mode

### Throughput

- **Generation speed is consistent** across all configurations: 194-223 tok/s
- Thinking ON shows slightly higher throughput (223 tok/s vs 195 tok/s) because the denominator includes the long reasoning stream which has more consistent cadence
- **Throughput is not a bottleneck** — the model generates at a steady ~200 tok/s once it starts outputting

### Accuracy (EM / F1)

- **Without explanation mode:** Thinking ON achieves **EM 57-59%, F1 74-76%** on HotpotQA distractor
- **With explanation mode:** EM/F1 drops to ~0% (expected — long explanation text vs short ground truth)
- Concurrency (4 workers) slightly improved accuracy (58.72% vs 57.28% EM) — likely due to lower tail latency

### Token Budget with Thinking ON

- **With explanation (`--explanation-word-limit 512`):** avg 2047 completion tokens — nearly the entire budget. Reasoning consumes ~1700-1900 tokens, leaving ~100-200 for the answer.
- **Without explanation:** avg 626-682 completion tokens — model reasons more efficiently for short answers
- If `max_tokens` is too low (e.g., 512), all tokens may be consumed by reasoning with no answer produced (`finish_reason=length`)

### Explanation Mode Impact

- **Explanation mode increases latency 3x** for thinking ON (9.80s vs 3.60s) because the model generates more reasoning + explanation tokens
- **Explanation mode decreases latency for thinking OFF** (2.12s) since the model just generates the explanation directly without reasoning overhead
- The primary use of explanation mode is throughput measurement — it forces the model to generate more tokens

### Inter-Chunk Latency

- **ICL is bimodal**: P50 ~0.5ms (fast bursts) but P99 ~230-240ms (occasional stalls)
- Average ICL is ~15ms across all modes
- The bimodal pattern suggests the API delivers tokens in bursts rather than one-at-a-time

### Concurrency (4 Workers)

- **Per-request latency was unchanged** (2.11s vs 2.12s) — the MaaS API handles concurrent requests without degradation
- **Total benchmark time decreased** from 99 min to 75 min (~24% faster)
- **Higher success rate** (78% vs 43%) due to shorter total run time (fewer tokens expired)

### Failure Rate

- All runs had high failure rates (38-78%) due to gcloud access token expiry after ~60 minutes
- This has been fixed with auto-refresh logic (every 45 min) in the latest script version
- Failed samples are excluded from aggregate statistics

---

## 9. Cross-Provider Latency Comparison

### 9.1 Provider Overview

All three providers offer GLM-4.7 as **serverless inference APIs** (no infrastructure to manage):

| | Google Vertex MaaS | Baseten | Cerebras Inference |
|---|---|---|---|
| **Type** | Serverless API | Serverless API | Serverless API |
| **TTFT** | ~500ms | ~500-600ms | ~510ms |
| **Throughput** | ~230 tok/s | ~200-250 tok/s | ~1200 tok/s |
| **Relative Price** | Baseline | ~Similar | ~2.4x more expensive |
| **Hardware** | Google-managed | GPU-based | Wafer Scale Engine (WSE-3) |

Sources: [Artificial Analysis](https://artificialanalysis.ai/models/glm-4-7/providers), [OpenRouter](https://openrouter.ai/z-ai/glm-4.7/performance)

**Key finding:** Google MaaS and Baseten deliver nearly identical TTFT and throughput. Cerebras offers ~6x higher throughput at 2.4x the cost, with the same TTFT.

### 9.2 Latency Speedup Analysis: Google MaaS vs Cerebras

*Assumptions: TTFT = 500ms for both providers. Google MaaS = 230 tok/s, Cerebras = 1200 tok/s. All tokens (thinking + output) counted as completion tokens.*

#### 256 Thinking Tokens (simpler tasks: classification, extraction, summarization)

| Thinking | Output | Total Tokens | Google MaaS | Cerebras | Time Saved | Speedup |
|---|---|---|---|---|---|---|
| 256 | 128 | **384** | 0.5s + 1.7s = **2.2s** | 0.5s + 0.3s = **0.8s** | **1.4s** | 2.8x |
| 256 | 256 | **512** | 0.5s + 2.2s = **2.7s** | 0.5s + 0.4s = **0.9s** | **1.8s** | 3.0x |
| 256 | 512 | **768** | 0.5s + 3.3s = **3.8s** | 0.5s + 0.6s = **1.1s** | **2.7s** | 3.5x |

#### 512 Thinking Tokens (medium tasks: multi-hop QA, code generation)

| Thinking | Output | Total Tokens | Google MaaS | Cerebras | Time Saved | Speedup |
|---|---|---|---|---|---|---|
| 512 | 128 | **640** | 0.5s + 2.8s = **3.3s** | 0.5s + 0.5s = **1.0s** | **2.3s** | 3.3x |
| 512 | 256 | **768** | 0.5s + 3.3s = **3.8s** | 0.5s + 0.6s = **1.1s** | **2.7s** | 3.5x |
| 512 | 512 | **1024** | 0.5s + 4.5s = **5.0s** | 0.5s + 0.9s = **1.4s** | **3.6s** | 3.6x |

### 9.3 Interpretation

- **TTFT is identical** (~500ms) across both providers — prefill latency is not a differentiator
- **The speedup advantage scales with output length**: 1.4s saved at 384 tokens → 3.6s saved at 1024 tokens
- **For low-output tasks** (128-256 total tokens): the difference is 0.3-1.4s — Google MaaS is effectively competitive
- **For high-output tasks** (1024+ total tokens): Cerebras saves 3.6+ seconds per request, which compounds at scale
- **At 10K requests/day** with 1024 total tokens: Cerebras saves ~10 hours of aggregate latency daily — at 2.4x cost

### 9.4 Typical Thinking Token Volume by Task

| Task Type | Typical Thinking Tokens | Typical Output Tokens | Total |
|---|---|---|---|
| Simple factual QA | 50-100 | 10-30 | 60-130 |
| Text classification / sentiment | 30-80 | 5-20 | 35-100 |
| Entity extraction | 50-150 | 20-100 | 70-250 |
| Summarization | 100-200 | 50-200 | 150-400 |
| Multi-hop reasoning (HotpotQA) | 400-600 | 10-50 | 410-650 |
| Code generation | 100-300 | 100-500 | 200-800 |
| Complex math / logic proofs | 1000-2000+ | 100-500 | 1100-2500+ |

For the most common tasks (classification, extraction, simple QA), total output is under 250 tokens — the latency difference between Google MaaS and Cerebras is under 1 second.

---

## 10. External Latency Benchmarks

For detailed, real-time cross-provider comparisons:

- **[OpenRouter — GLM-4.7 Performance](https://openrouter.ai/z-ai/glm-4.7/performance)** — TTFT, output speed (tok/s), and total latency across 8+ providers
- **[Artificial Analysis — GLM-4.7 Providers](https://artificialanalysis.ai/models/glm-4-7/providers)** — Independent TTFT and output speed benchmarks with median/P25/P75 distributions

### GLM-4.7 Model Specifications

| Property | Value |
|---|---|
| **Context Length** | 200,000 tokens |
| **Architecture** | MoE, 358B total parameters |
| **Reasoning** | Enabled by default (`reasoning_content` field) |
| **Max Completion Tokens** | 128,000 (Vertex AI MaaS) |
| **Providers** | 8+ via OpenRouter (DeepInfra, Nebius, Together, Parasail, SiliconFlow, AtlasCloud, Venice, Z.AI, Google Vertex) |
| **Quantization** | FP8 (most providers), FP4 (DeepInfra, Venice) |
| **Rate Limit (Vertex)** | 250 RPM |

---

## 11. References

- [Vertex AI MaaS — Thinking for Open Models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/thinking) — Official docs on enabling/disabling reasoning
- [Artificial Analysis — GLM-4.7](https://artificialanalysis.ai/models/glm-4-7/providers) — Independent provider benchmarks
- [OpenRouter — GLM-4.7](https://openrouter.ai/z-ai/glm-4.7/performance) — Cross-provider performance data
- [HotpotQA](https://hotpotqa.github.io/) — Dataset documentation and leaderboard
