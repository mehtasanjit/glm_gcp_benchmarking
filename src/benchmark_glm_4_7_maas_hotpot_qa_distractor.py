#!/usr/bin/env python3
"""
benchmark_glm_4_7_maas_hotpot_qa_distractor.py — Benchmark GLM-4.7 MaaS on HotpotQA.

Runs HotpotQA distractor validation set through GLM-4.7 MaaS endpoint on Vertex AI.
Measures: TTFT, total latency, tokens/sec, ITL, accuracy (EM, F1), cost estimation.

Usage:
    python benchmark_glm_4_7_maas_hotpot_qa_distractor.py --num-samples 50
    python benchmark_glm_4_7_maas_hotpot_qa_distractor.py --num-samples 100 --output results.json
    python benchmark_glm_4_7_maas_hotpot_qa_distractor.py --data-file ./data/hotpotqa/hotpotqa_distractor_validation.jsonl
"""

import argparse
import json
import os
import re
import string
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from openai import OpenAI


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "zai-org/glm-4.7-maas"
PROJECT_ID = ""
BASE_URL = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi"
DATASET_ID = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DEFAULT_SPLIT = "validation"
MAX_TOKENS = 512
TEMPERATURE = 0.1


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_access_token() -> str:
    r = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Failed to get access token: {r.stderr}")
    return r.stdout.strip()


# ---------------------------------------------------------------------------
# HotpotQA Data Loading
# ---------------------------------------------------------------------------

def load_hotpotqa_from_file(path: str, num_samples: int) -> list[dict]:
    """Load from local JSONL file."""
    samples = []
    with open(path) as f:
        for line in f:
            if len(samples) >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def load_hotpotqa_from_hf(num_samples: int) -> list[dict]:
    """Load from HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DEFAULT_SPLIT)
    return [ds[i] for i in range(min(num_samples, len(ds)))]


def format_hotpotqa_prompt(sample: dict, explanation_word_limit: int = 0) -> str:
    """Format a HotpotQA sample into a prompt with context paragraphs.
    
    Args:
        sample: HotpotQA sample dict.
        explanation_word_limit: If > 0, ask the model for a detailed explanation
            of how it arrived at the answer, up to this many words. This generates
            more output tokens, useful for measuring throughput when thinking is OFF.
    """
    context_parts = []
    for title, sentences in zip(sample["context"]["title"], sample["context"]["sentences"]):
        text = " ".join(sentences)
        context_parts.append(f"**{title}:** {text}")

    context_text = "\n\n".join(context_parts)
    question = sample["question"]

    if explanation_word_limit > 0:
        return (
            f"Answer the following question based on the provided context. "
            f"First give the short, direct answer, then provide a detailed explanation "
            f"of how you arrived at the answer in approximately {explanation_word_limit} words.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        return (
            f"Answer the following question based on the provided context. "
            f"Give a short, direct answer.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )


# ---------------------------------------------------------------------------
# Evaluation Metrics (EM, F1)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Inference with Metrics
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    sample_id: str = ""
    question: str = ""
    ground_truth: str = ""
    prediction: str = ""
    reasoning_content: str = ""
    # Latency
    total_latency_s: float = 0.0
    time_to_response_s: float | None = None          # non-streaming only: total round-trip
    ttft_s: float | None = None                      # streaming: first answer/content chunk
    ttft_ms: float | None = None
    ttft_thinking_s: float | None = None             # streaming: first reasoning chunk
    ttft_thinking_ms: float | None = None
    ttft_first_visible_s: float | None = None        # min(thinking, answer) — first emission
    ttft_first_visible_ms: float | None = None
    generation_time_s: float = 0.0
    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Throughput
    tokens_per_sec: float = 0.0
    overall_tokens_per_sec: float = 0.0
    ms_per_token: float = 0.0
    # ITL
    itl_times_ms: list = field(default_factory=list)
    avg_itl_ms: float | None = None
    p50_itl_ms: float | None = None
    p90_itl_ms: float | None = None
    p99_itl_ms: float | None = None
    # Quality
    em: float = 0.0
    f1: float = 0.0
    # Meta
    streamed: bool = True
    error: str | None = None


def run_streaming_inference(
    client: OpenAI, prompt: str, sample: dict, thinking: bool = True
) -> InferenceResult:
    """Run streaming inference and collect all metrics.

    Tracks reasoning and answer chunks separately. Token counts come from
    the API usage block when available; chunk counts are NOT used as a
    proxy for token counts (chunks ≠ tokens in streaming).
    """
    result = InferenceResult(
        sample_id=sample.get("id", ""),
        question=sample.get("question", ""),
        ground_truth=sample.get("answer", ""),
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions concisely based on provided context."},
        {"role": "user", "content": prompt},
    ]

    t_start = time.perf_counter()
    reasoning_chunks: list[str] = []
    reasoning_times: list[float] = []
    answer_chunks: list[str] = []
    answer_times: list[float] = []

    try:
        create_kwargs = dict(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        )
        if not thinking:
            create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        stream = client.chat.completions.create(**create_kwargs)

        for chunk in stream:
            t_now = time.perf_counter()

            # Capture usage from the final stream chunk (OpenAI SDK convention)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                result.prompt_tokens = chunk.usage.prompt_tokens or 0
                result.completion_tokens = chunk.usage.completion_tokens or 0
                result.total_tokens = chunk.usage.total_tokens or 0

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Capture reasoning chunks
            reasoning_part = getattr(delta, "reasoning_content", None)
            if reasoning_part:
                reasoning_chunks.append(reasoning_part)
                reasoning_times.append(t_now)
                if result.ttft_thinking_s is None:
                    result.ttft_thinking_s = t_now - t_start
                    result.ttft_thinking_ms = result.ttft_thinking_s * 1000

            # Capture answer/content chunks
            if delta.content:
                answer_chunks.append(delta.content)
                answer_times.append(t_now)
                if result.ttft_s is None:
                    result.ttft_s = t_now - t_start
                    result.ttft_ms = result.ttft_s * 1000

        t_end = time.perf_counter()
        result.total_latency_s = t_end - t_start
        result.prediction = "".join(answer_chunks)
        result.reasoning_content = "".join(reasoning_chunks)

        # First visible emission (whichever came first: reasoning or answer)
        candidates = [v for v in [result.ttft_thinking_s, result.ttft_s] if v is not None]
        if candidates:
            result.ttft_first_visible_s = min(candidates)
            result.ttft_first_visible_ms = result.ttft_first_visible_s * 1000

        # Generation time — from first emission (reasoning or content) to end
        if result.ttft_first_visible_s is not None:
            result.generation_time_s = result.total_latency_s - result.ttft_first_visible_s

        # Throughput — use real token counts from the API usage block.
        if result.completion_tokens > 0:
            if result.generation_time_s > 0:
                result.tokens_per_sec = result.completion_tokens / result.generation_time_s
            if result.total_latency_s > 0:
                result.overall_tokens_per_sec = result.completion_tokens / result.total_latency_s
                result.ms_per_token = (result.total_latency_s / result.completion_tokens) * 1000

        # Inter-chunk latency — use answer chunks if available, else reasoning chunks
        timing_source = answer_times if len(answer_times) > 1 else reasoning_times
        itls = [timing_source[i] - timing_source[i - 1] for i in range(1, len(timing_source))]
        result.itl_times_ms = [t * 1000 for t in itls]
        if itls:
            sorted_itls = sorted(itls)
            result.avg_itl_ms = sum(itls) / len(itls) * 1000
            result.p50_itl_ms = sorted_itls[len(sorted_itls) // 2] * 1000
            result.p90_itl_ms = sorted_itls[int(len(sorted_itls) * 0.9)] * 1000
            result.p99_itl_ms = sorted_itls[int(len(sorted_itls) * 0.99)] * 1000

        # Quality
        result.em = exact_match(result.prediction, result.ground_truth)
        result.f1 = f1_score(result.prediction, result.ground_truth)

    except Exception as e:
        result.error = str(e)

    return result


def run_non_streaming_inference(
    client: OpenAI, prompt: str, sample: dict, thinking: bool = True
) -> InferenceResult:
    """Run non-streaming inference."""
    result = InferenceResult(
        sample_id=sample.get("id", ""),
        question=sample.get("question", ""),
        ground_truth=sample.get("answer", ""),
        streamed=False,
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions concisely based on provided context."},
        {"role": "user", "content": prompt},
    ]

    t_start = time.perf_counter()
    try:
        create_kwargs = dict(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=False,
        )
        if not thinking:
            create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        response = client.chat.completions.create(**create_kwargs)
        t_end = time.perf_counter()

        result.total_latency_s = t_end - t_start
        # Non-streaming: we only know total round-trip, not true TTFT
        result.time_to_response_s = result.total_latency_s
        result.generation_time_s = result.total_latency_s

        choice = response.choices[0]
        result.prediction = choice.message.content or ""
        if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content:
            result.reasoning_content = choice.message.reasoning_content

        usage = response.usage
        result.prompt_tokens = usage.prompt_tokens
        result.completion_tokens = usage.completion_tokens
        result.total_tokens = usage.total_tokens

        if result.total_latency_s > 0 and result.completion_tokens > 0:
            result.tokens_per_sec = result.completion_tokens / result.total_latency_s
            result.overall_tokens_per_sec = result.tokens_per_sec
            result.ms_per_token = (result.total_latency_s / result.completion_tokens) * 1000

        result.em = exact_match(result.prediction, result.ground_truth)
        result.f1 = f1_score(result.prediction, result.ground_truth)

    except Exception as e:
        result.error = str(e)

    return result


# ---------------------------------------------------------------------------
# Aggregate Stats
# ---------------------------------------------------------------------------

def compute_aggregate_stats(results: list[InferenceResult]) -> dict:
    """Compute aggregate statistics across all results."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    if not successful:
        return {"error": "No successful results"}

    latencies = [r.total_latency_s for r in successful]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    ttfts_thinking = [r.ttft_thinking_ms for r in successful if r.ttft_thinking_ms is not None]
    tps = [r.tokens_per_sec for r in successful if r.tokens_per_sec > 0]
    ems = [r.em for r in successful]
    f1s = [r.f1 for r in successful]
    prompt_toks = [r.prompt_tokens for r in successful if r.prompt_tokens > 0]
    comp_toks = [r.completion_tokens for r in successful if r.completion_tokens > 0]

    def percentile(data, p):
        if not data:
            return None
        s = sorted(data)
        idx = round((len(s) - 1) * p)
        return s[min(idx, len(s) - 1)]

    # All ITLs flattened
    all_itls = []
    for r in successful:
        all_itls.extend(r.itl_times_ms)

    stats = {
        "total_samples": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "failed_ids": [r.sample_id for r in failed],

        # Quality
        "exact_match": sum(ems) / len(ems) if ems else 0,
        "f1_score": sum(f1s) / len(f1s) if f1s else 0,

        # Latency
        "latency_avg_s": sum(latencies) / len(latencies),
        "latency_p50_s": percentile(latencies, 0.5),
        "latency_p90_s": percentile(latencies, 0.9),
        "latency_p99_s": percentile(latencies, 0.99),
        "latency_min_s": min(latencies),
        "latency_max_s": max(latencies),

        # TTFT (first thinking token)
        "ttft_thinking_avg_ms": sum(ttfts_thinking) / len(ttfts_thinking) if ttfts_thinking else None,
        "ttft_thinking_p50_ms": percentile(ttfts_thinking, 0.5),
        "ttft_thinking_p75_ms": percentile(ttfts_thinking, 0.75),
        "ttft_thinking_p90_ms": percentile(ttfts_thinking, 0.9),
        "ttft_thinking_p95_ms": percentile(ttfts_thinking, 0.95),
        "ttft_thinking_p99_ms": percentile(ttfts_thinking, 0.99),
        "ttft_thinking_min_ms": min(ttfts_thinking) if ttfts_thinking else None,
        "ttft_thinking_max_ms": max(ttfts_thinking) if ttfts_thinking else None,

        # TTFT (first content token)
        "ttft_avg_ms": sum(ttfts) / len(ttfts) if ttfts else None,
        "ttft_p50_ms": percentile(ttfts, 0.5),
        "ttft_p75_ms": percentile(ttfts, 0.75),
        "ttft_p90_ms": percentile(ttfts, 0.9),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "ttft_min_ms": min(ttfts) if ttfts else None,
        "ttft_max_ms": max(ttfts) if ttfts else None,

        # Throughput
        "tokens_per_sec_avg": sum(tps) / len(tps) if tps else 0,
        "tokens_per_sec_p50": percentile(tps, 0.5),
        "tokens_per_sec_p75": percentile(tps, 0.75),
        "tokens_per_sec_p90": percentile(tps, 0.9),
        "tokens_per_sec_p95": percentile(tps, 0.95),
        "tokens_per_sec_p99": percentile(tps, 0.99),

        # ITL (global)
        "itl_avg_ms": sum(all_itls) / len(all_itls) if all_itls else None,
        "itl_p50_ms": percentile(all_itls, 0.5),
        "itl_p90_ms": percentile(all_itls, 0.9),
        "itl_p99_ms": percentile(all_itls, 0.99),

        # Token counts
        "avg_prompt_tokens": sum(prompt_toks) / len(prompt_toks) if prompt_toks else 0,
        "avg_completion_tokens": sum(comp_toks) / len(comp_toks) if comp_toks else 0,
        "total_prompt_tokens": sum(prompt_toks),
        "total_completion_tokens": sum(comp_toks),
        "total_tokens": sum(prompt_toks) + sum(comp_toks),

        # Cost estimate (GLM-4.7 MaaS pricing — approximate)
        "est_input_cost_usd": sum(prompt_toks) / 1_000_000 * 0.10,
        "est_output_cost_usd": sum(comp_toks) / 1_000_000 * 0.30,

        # First visible emission (min of thinking TTFT and content TTFT)
        "ttft_first_visible_avg_ms": None,

        # Timing
        "total_wall_time_s": sum(latencies),
        "benchmark_wall_time_s": None,  # set externally for concurrent runs
    }
    stats["est_total_cost_usd"] = stats["est_input_cost_usd"] + stats["est_output_cost_usd"]

    return stats


# ---------------------------------------------------------------------------
# CLI & Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark GLM-4.7 MaaS on HotpotQA distractor.",
    )
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to benchmark (default: 50)")
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to local JSONL data file (default: load from HF)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--no-stream", action="store_true",
                        help="Use non-streaming mode")
    parser.add_argument("--thinking", action="store_true", default=True, dest="thinking",
                        help="Enable thinking/reasoning mode (default: ON)")
    parser.add_argument("--no-thinking", action="store_false", dest="thinking",
                        help="Disable thinking/reasoning mode")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max output tokens (default: {MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Temperature (default: {TEMPERATURE})")
    parser.add_argument("--project", type=str, default=PROJECT_ID,
                        help=f"GCP project ID (default: {PROJECT_ID})")
    parser.add_argument("--request-delay", type=float, default=0.0,
                        help="Delay in seconds between requests (default: 0.0)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of concurrent workers (default: 1 = sequential)")
    parser.add_argument("--explanation-word-limit", type=int, default=0,
                        help="Ask model for detailed explanation up to N words (default: 0 = short answer only). "
                             "Useful for generating more output tokens to measure throughput.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    global MAX_TOKENS, TEMPERATURE, PROJECT_ID, BASE_URL, MODEL
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    PROJECT_ID = args.project
    BASE_URL = f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/global/endpoints/openapi"

    print()
    print("=" * 70)
    print("🏆 GLM-4.7 MaaS — HotpotQA Distractor Benchmark")
    print("=" * 70)
    print(f"  Model:       {MODEL}")
    print(f"  Endpoint:    {BASE_URL}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  Max Tokens:  {MAX_TOKENS}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Streaming:   {'No' if args.no_stream else 'Yes'}")
    print(f"  Thinking:    {'ON' if args.thinking else 'OFF'}")
    print(f"  Workers:     {args.num_workers}")
    print(f"  Req Delay:   {args.request_delay}s")
    if args.explanation_word_limit > 0:
        print(f"  Explanation: ~{args.explanation_word_limit} words")
    print("=" * 70)
    print()

    # Auth — token refreshes automatically every 45 minutes
    print("🔑 Getting access token...")
    token = get_access_token()
    token_refresh_time = time.time()
    TOKEN_REFRESH_INTERVAL = 45 * 60  # refresh every 45 min (expires at 60)
    print("  ✅ Token obtained (auto-refreshes every 45 min)")
    print()

    # Client
    client = OpenAI(
        base_url=BASE_URL,
        api_key=token,
    )

    # Load data
    print("📦 Loading HotpotQA data...")
    if args.data_file:
        samples = load_hotpotqa_from_file(args.data_file, args.num_samples)
        print(f"  Loaded {len(samples)} samples from {args.data_file}")
    else:
        samples = load_hotpotqa_from_hf(args.num_samples)
        print(f"  Loaded {len(samples)} samples from HuggingFace")
    print()

    # Run benchmark
    results: list[InferenceResult] = []
    num_workers = args.num_workers
    request_delay = args.request_delay
    use_stream = not args.no_stream
    thinking = args.thinking

    print(f"🚀 Running benchmark ({len(samples)} samples, {num_workers} worker(s))...")
    print()

    def _run_single(idx_sample):
        idx, sample = idx_sample
        prompt = format_hotpotqa_prompt(sample, explanation_word_limit=explanation_word_limit)
        if use_stream:
            return idx, run_streaming_inference(client, prompt, sample, thinking=thinking)
        else:
            return idx, run_non_streaming_inference(client, prompt, sample, thinking=thinking)

    def _print_progress(i, result, total):
        status = "✅" if result.error is None else "❌"
        em_icon = "🎯" if result.em == 1.0 else "  "
        latency_str = f"{result.total_latency_s:.2f}s"
        tps_str = f"{result.tokens_per_sec:.0f} tok/s" if result.tokens_per_sec > 0 else "N/A"
        ttft_think_str = f"T-Think={result.ttft_thinking_ms:.0f}ms" if result.ttft_thinking_ms else ""
        ttft_str = f"TTFT={result.ttft_ms:.0f}ms" if result.ttft_ms else ""
        print(
            f"  [{i+1:3d}/{total}] {status} {em_icon} "
            f"EM={result.em:.0f} F1={result.f1:.2f} | "
            f"{latency_str} {tps_str} {ttft_think_str} {ttft_str} | "
            f"Q: {result.question[:60]}..."
        )
        if result.error:
            print(f"           ❌ Error: {result.error[:80]}")

    benchmark_start = time.perf_counter()

    explanation_word_limit = args.explanation_word_limit

    if num_workers <= 1:
        # Sequential execution
        for i, sample in enumerate(samples):
            # Auto-refresh token before expiry
            if time.time() - token_refresh_time > TOKEN_REFRESH_INTERVAL:
                print("  🔄 Refreshing access token...")
                token = get_access_token()
                token_refresh_time = time.time()
                client = OpenAI(base_url=BASE_URL, api_key=token)
                print("  ✅ Token refreshed")

            prompt = format_hotpotqa_prompt(sample, explanation_word_limit=explanation_word_limit)
            if use_stream:
                result = run_streaming_inference(client, prompt, sample, thinking=thinking)
            else:
                result = run_non_streaming_inference(client, prompt, sample, thinking=thinking)
            results.append(result)
            _print_progress(i, result, len(samples))
            if request_delay > 0 and i < len(samples) - 1:
                time.sleep(request_delay)
    else:
        # Concurrent execution — submit in batches to allow progress printing
        results_by_idx = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            submitted = 0

            for idx, sample in enumerate(samples):
                # Auto-refresh token before expiry
                if time.time() - token_refresh_time > TOKEN_REFRESH_INTERVAL:
                    print("  🔄 Refreshing access token...")
                    token = get_access_token()
                    token_refresh_time = time.time()
                    client = OpenAI(base_url=BASE_URL, api_key=token)
                    print("  ✅ Token refreshed")

                future = executor.submit(_run_single, (idx, sample))
                futures[future] = idx
                submitted += 1

                # Collect completed results so far (non-blocking)
                done_futures = [f for f in futures if f.done()]
                for f in done_futures:
                    if f not in results_by_idx.values():
                        fidx, result = f.result()
                        if fidx not in results_by_idx:
                            results_by_idx[fidx] = result
                            completed += 1
                            _print_progress(fidx, result, len(samples))

                if request_delay > 0 and idx < len(samples) - 1:
                    time.sleep(request_delay)

            # Collect remaining futures
            for future in as_completed(futures):
                idx_f = futures[future]
                if idx_f not in results_by_idx:
                    fidx, result = future.result()
                    results_by_idx[fidx] = result
                    completed += 1
                    _print_progress(fidx, result, len(samples))

        # Preserve original order
        results = [results_by_idx[i] for i in range(len(samples))]

    benchmark_wall_time_s = time.perf_counter() - benchmark_start

    # Aggregate
    print()
    print("=" * 70)
    print("📊 AGGREGATE RESULTS")
    print("=" * 70)

    stats = compute_aggregate_stats(results)
    stats["benchmark_wall_time_s"] = benchmark_wall_time_s

    # Compute first_visible aggregate
    first_visibles = [r.ttft_first_visible_ms for r in results if r.error is None and r.ttft_first_visible_ms is not None]
    if first_visibles:
        def _pct(data, p):
            s = sorted(data)
            return s[min(round((len(s) - 1) * p), len(s) - 1)]
        stats["ttft_first_visible_avg_ms"] = sum(first_visibles) / len(first_visibles)
        stats["ttft_first_visible_p50_ms"] = _pct(first_visibles, 0.5)
        stats["ttft_first_visible_p90_ms"] = _pct(first_visibles, 0.9)
        stats["ttft_first_visible_p99_ms"] = _pct(first_visibles, 0.99)

    print(f"\n  Quality:")
    print(f"    Exact Match:       {stats['exact_match']:.2%}")
    print(f"    F1 Score:          {stats['f1_score']:.2%}")

    print(f"\n  Latency:")
    print(f"    Average:           {stats['latency_avg_s']:.3f} s")
    print(f"    P50:               {stats['latency_p50_s']:.3f} s")
    print(f"    P90:               {stats['latency_p90_s']:.3f} s")
    print(f"    P99:               {stats['latency_p99_s']:.3f} s")
    print(f"    Min:               {stats['latency_min_s']:.3f} s")
    print(f"    Max:               {stats['latency_max_s']:.3f} s")

    if stats.get("ttft_first_visible_avg_ms"):
        print(f"\n  TTFT — First Visible Emission (min of reasoning/answer):")
        print(f"    Average:           {stats['ttft_first_visible_avg_ms']:.1f} ms")
        print(f"    P50:               {stats['ttft_first_visible_p50_ms']:.1f} ms")
        print(f"    P90:               {stats['ttft_first_visible_p90_ms']:.1f} ms")
        print(f"    P99:               {stats['ttft_first_visible_p99_ms']:.1f} ms")

    if stats.get("ttft_thinking_avg_ms"):
        print(f"\n  TTFT — First Reasoning Chunk:")
        print(f"    Average:           {stats['ttft_thinking_avg_ms']:.1f} ms")
        print(f"    P50:               {stats['ttft_thinking_p50_ms']:.1f} ms")
        print(f"    P75:               {stats['ttft_thinking_p75_ms']:.1f} ms")
        print(f"    P90:               {stats['ttft_thinking_p90_ms']:.1f} ms")
        print(f"    P95:               {stats['ttft_thinking_p95_ms']:.1f} ms")
        print(f"    P99:               {stats['ttft_thinking_p99_ms']:.1f} ms")
        print(f"    Min:               {stats['ttft_thinking_min_ms']:.1f} ms")
        print(f"    Max:               {stats['ttft_thinking_max_ms']:.1f} ms")

    if stats.get("ttft_avg_ms"):
        print(f"\n  TTFT — First Answer Chunk:")
        print(f"    Average:           {stats['ttft_avg_ms']:.1f} ms")
        print(f"    P50:               {stats['ttft_p50_ms']:.1f} ms")
        print(f"    P75:               {stats['ttft_p75_ms']:.1f} ms")
        print(f"    P90:               {stats['ttft_p90_ms']:.1f} ms")
        print(f"    P95:               {stats['ttft_p95_ms']:.1f} ms")
        print(f"    P99:               {stats['ttft_p99_ms']:.1f} ms")
        print(f"    Min:               {stats['ttft_min_ms']:.1f} ms")
        print(f"    Max:               {stats['ttft_max_ms']:.1f} ms")

    print(f"\n  Throughput (from first emission to end):")
    print(f"    Average tok/s:     {stats['tokens_per_sec_avg']:.1f}")
    print(f"    P50 tok/s:         {stats['tokens_per_sec_p50']:.1f}")
    print(f"    P75 tok/s:         {stats['tokens_per_sec_p75']:.1f}")
    print(f"    P90 tok/s:         {stats['tokens_per_sec_p90']:.1f}")
    print(f"    P95 tok/s:         {stats['tokens_per_sec_p95']:.1f}")
    print(f"    P99 tok/s:         {stats['tokens_per_sec_p99']:.1f}")

    if stats.get("itl_avg_ms"):
        print(f"\n  Inter-Chunk Latency (Content stream cadence):")
        print(f"    Average:           {stats['itl_avg_ms']:.1f} ms")
        print(f"    P50:               {stats['itl_p50_ms']:.1f} ms")
        print(f"    P90:               {stats['itl_p90_ms']:.1f} ms")
        print(f"    P99:               {stats['itl_p99_ms']:.1f} ms")

    print(f"\n  Token Counts:")
    print(f"    Avg prompt:        {stats['avg_prompt_tokens']:.0f}")
    print(f"    Avg completion:    {stats['avg_completion_tokens']:.0f}")
    print(f"    Total tokens:      {stats['total_tokens']:,}")

    print(f"\n  Cost Estimate:")
    print(f"    Input:             ${stats['est_input_cost_usd']:.4f}")
    print(f"    Output:            ${stats['est_output_cost_usd']:.4f}")
    print(f"    Total:             ${stats['est_total_cost_usd']:.4f}")

    print(f"\n  Summary:")
    print(f"    Samples:           {stats['successful']}/{stats['total_samples']} successful")
    wall_label = stats.get("benchmark_wall_time_s") or stats["total_wall_time_s"]
    print(f"    Wall time:         {wall_label:.1f} s")

    print()
    print("=" * 70)

    # Save results
    if args.output:
        # Build contextual notes based on config
        notes = []
        if not args.no_stream and args.thinking:
            notes.append("Thinking ON: completion_tokens from API includes both reasoning and content tokens. "
                         "tokens_per_sec = completion_tokens / (total_latency - ttft_first_visible). "
                         "This is combined reasoning+content throughput, not content-only throughput.")
        if not (not args.no_stream):
            notes.append("Non-streaming mode: TTFT metrics (ttft_ms, ttft_thinking_ms) are not available. "
                         "tokens_per_sec = completion_tokens / total_latency (includes full round-trip).")
        if args.explanation_word_limit > 0:
            notes.append(f"Explanation mode ({args.explanation_word_limit} words): EM and F1 scores will be "
                         "significantly lower because the model's full explanation text is compared against "
                         "the short ground truth answer. These accuracy metrics are not meaningful in this mode — "
                         "use this mode for throughput measurement only.")
        if not args.thinking:
            notes.append("Thinking OFF: reasoning was disabled via chat_template_kwargs.enable_thinking=false. "
                         "No reasoning_content chunks should appear. TTFT measures true prefill latency.")

        output_data = {
            "config": {
                "model": MODEL,
                "num_samples": len(samples),
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "streaming": not args.no_stream,
                "thinking": args.thinking,
                "num_workers": args.num_workers,
                "request_delay": args.request_delay,
                "explanation_word_limit": args.explanation_word_limit,
                "project": PROJECT_ID,
            },
            "notes": notes,
            "aggregate": stats,
            "per_sample": [
                {
                    "id": r.sample_id,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "prediction": r.prediction,
                    "reasoning_content": r.reasoning_content,
                    "em": r.em,
                    "f1": r.f1,
                    "total_latency_s": r.total_latency_s,
                    "ttft_ms": r.ttft_ms,
                    "ttft_thinking_ms": r.ttft_thinking_ms,
                    "ttft_first_visible_ms": r.ttft_first_visible_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                    "overall_tokens_per_sec": r.overall_tokens_per_sec,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
