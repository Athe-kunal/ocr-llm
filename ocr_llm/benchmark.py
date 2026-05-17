"""OCR pipeline benchmark harness.

Measures four metric categories across implementations:
  1. PDF ingestion throughput (render time / PDF-ready time)
  2. Token throughput (output tokens/s, total tokens/s)
  3. Request latency (TTFT, ITL, TPOT, E2EL) via SSE streaming
  4. Normalised page completion time (E2EL ms / output tokens)

Leverages vllm.benchmarks internals for all latency/throughput math.

Usage:
    python -m ocr_llm.benchmark --pdf_dir=<dir> --label=<name> --notes=<text>
    make benchmark LABEL="my-run" NOTES="8xA100, batch=16"
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import aiohttp
import fire
import numpy as np
from pypdf import PdfReader

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt

# vLLM benchmark internals — provide all latency / throughput aggregation
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_chat_completions,
)
from vllm.benchmarks.serve import BenchmarkMetrics, calculate_metrics
from vllm.tokenizers import TokenizerLike

DEFAULT_SERVER = "http://127.0.0.1:8000"
DEFAULT_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
DEFAULT_METRICS_DIR = "metrics"
DEFAULT_TARGET_IMAGE_DIM = 1288
DEFAULT_MAX_TOKENS = 8000
DEFAULT_CONCURRENCY = 4        # number of PDFs processed concurrently
DEFAULT_PAGES_PER_PDF = 16     # max page requests in-flight per PDF
DEFAULT_PERCENTILES = [50.0, 90.0, 95.0, 99.0]

# Prometheus metric names we care about on the vLLM server
_PROM_METRICS = [
    "vllm:gpu_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:inter_token_latency_seconds_sum",
    "vllm:inter_token_latency_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    pdf_dir: str
    server: str = DEFAULT_SERVER
    model: str = DEFAULT_MODEL
    label: str = "default"
    notes: str = ""
    concurrency: int = DEFAULT_CONCURRENCY
    pages_per_pdf: int = DEFAULT_PAGES_PER_PDF
    target_image_dim: int = DEFAULT_TARGET_IMAGE_DIM
    max_tokens: int = DEFAULT_MAX_TOKENS
    percentiles: list[float] = field(default_factory=lambda: list(DEFAULT_PERCENTILES))
    metrics_dir: str = DEFAULT_METRICS_DIR
    api_key: str | None = None


# ---------------------------------------------------------------------------
# Per-page benchmark result
# ---------------------------------------------------------------------------

@dataclass
class PageBenchmarkData:
    """Wraps a RequestFuncOutput with OCR-specific context fields."""

    pdf_path: str
    page_num: int
    render_time_s: float          # time to render PDF page → base64 PNG
    request_output: RequestFuncOutput

    @property
    def normalized_page_time_ms(self) -> float:
        """E2EL ms / output_tokens — client-side ms per generated token."""
        tokens = self.request_output.output_tokens or 1
        return (self.request_output.latency * 1000) / tokens


@dataclass
class PDFBenchmarkResult:
    pdf_path: str
    num_pages: int
    render_time_s: float    # wall time from open → all pages rendered
    total_time_s: float     # wall time from open → all responses received
    page_data: list[PageBenchmarkData]


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------

async def _render_page_to_base64(
    local_pdf_path: str,
    page_num: int,
    target_image_dim: int,
) -> str:
    """Render one PDF page to base64 PNG in a thread pool."""
    return await asyncio.to_thread(
        render_pdf_to_base64png,
        local_pdf_path,
        page_num,
        target_longest_image_dim=target_image_dim,
    )


def _build_request_func_input(
    config: BenchmarkConfig,
    image_base64: str,
    prompt_len_hint: int = 0,
) -> RequestFuncInput:
    """Build a RequestFuncInput for one OCR page request.

    Uses the same prompt text as the production pipeline
    (build_no_anchoring_v4_yaml_prompt) and passes the rendered page image
    as multi_modal_content so that async_request_openai_chat_completions can
    attach it correctly to the streaming chat request.
    """
    prompt_text = build_no_anchoring_v4_yaml_prompt()
    image_content = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
    }
    chat_url = f"{config.server.rstrip('/')}/v1/chat/completions"

    headers: dict | None = None
    if config.api_key:
        headers = {"Authorization": f"Bearer {config.api_key}"}

    return RequestFuncInput(
        prompt=prompt_text,
        api_url=chat_url,
        prompt_len=prompt_len_hint,
        output_len=config.max_tokens,
        model=config.model,
        model_name=config.model,
        multi_modal_content=image_content,
        extra_headers=headers,
    )


# ---------------------------------------------------------------------------
# PDF-level benchmark
# ---------------------------------------------------------------------------

async def benchmark_pdf(
    config: BenchmarkConfig,
    pdf_path: str,
    session: aiohttp.ClientSession,
) -> PDFBenchmarkResult:
    """Process one PDF: render pages, stream through vLLM, collect metrics."""
    t0 = time.perf_counter()

    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf", delete=False) as tf:
        tmp_path = tf.name
        data = await asyncio.to_thread(lambda: open(pdf_path, "rb").read())
        tf.write(data)
        tf.flush()

    try:
        reader = PdfReader(tmp_path)
        num_pages = reader.get_num_pages()

        # --- Render phase: time until all pages are ready for vLLM ---
        pages_semaphore = asyncio.Semaphore(config.pages_per_pdf)

        render_times: list[float] = []
        images: list[str] = []

        async def _render_one(page_num: int) -> tuple[int, str, float]:
            async with pages_semaphore:
                r0 = time.perf_counter()
                img = await _render_page_to_base64(
                    tmp_path, page_num, config.target_image_dim
                )
                return page_num, img, time.perf_counter() - r0

        render_tasks = [
            asyncio.create_task(_render_one(p)) for p in range(1, num_pages + 1)
        ]
        render_results = await asyncio.gather(*render_tasks)

        t_after_render = time.perf_counter()
        render_wall_time = t_after_render - t0

        # Sort by page number so page_data list is ordered
        render_results = sorted(render_results, key=lambda x: x[0])
        for _, img, rt in render_results:
            images.append(img)
            render_times.append(rt)

        # --- Inference phase: SSE streaming requests ---
        async def _request_one(
            page_num: int, image_base64: str, render_time: float
        ) -> PageBenchmarkData:
            async with pages_semaphore:
                req_input = _build_request_func_input(config, image_base64)
                output = await async_request_openai_chat_completions(
                    req_input, session
                )
            return PageBenchmarkData(
                pdf_path=pdf_path,
                page_num=page_num,
                render_time_s=render_time,
                request_output=output,
            )

        infer_tasks = [
            asyncio.create_task(_request_one(pn, img, rt))
            for (pn, img, rt) in render_results
        ]
        page_data = await asyncio.gather(*infer_tasks)

        total_time = time.perf_counter() - t0

        return PDFBenchmarkResult(
            pdf_path=pdf_path,
            num_pages=num_pages,
            render_time_s=render_wall_time,
            total_time_s=total_time,
            page_data=list(page_data),
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Top-level benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(config: BenchmarkConfig) -> dict:
    """Run benchmark on all PDFs in pdf_dir and return a results dict."""
    pdf_files = sorted(Path(config.pdf_dir).rglob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found recursively in {config.pdf_dir!r}")

    print(f"\n[benchmark] label={config.label!r}  pdfs={len(pdf_files)}  server={config.server}")

    pdf_semaphore = asyncio.Semaphore(config.concurrency)
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)

    t_start = time.perf_counter()
    pdf_results: list[PDFBenchmarkResult] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _run_one(pdf_path: Path) -> PDFBenchmarkResult:
            async with pdf_semaphore:
                print(f"  → {pdf_path.name}")
                return await benchmark_pdf(config, str(pdf_path), session)

        tasks = [asyncio.create_task(_run_one(p)) for p in pdf_files]
        pdf_results = await asyncio.gather(*tasks)

    dur_s = time.perf_counter() - t_start

    # Flatten all page RequestFuncOutputs
    all_outputs: list[RequestFuncOutput] = [
        pd.request_output
        for pr in pdf_results
        for pd in pr.page_data
    ]

    # calculate_metrics() does all the stats math: mean/median/std/percentiles
    # for TTFT, ITL, TPOT, E2EL, plus throughput figures.
    # input_requests arg is only in the signature but never read — pass empty.
    # vLLM's calculate_metrics body explicitly handles tokenizer=None, but the
    # type signature doesn't admit None. cast() satisfies the type checker.
    bench_metrics, _ = calculate_metrics(
        input_requests=[],
        outputs=all_outputs,
        dur_s=dur_s,
        tokenizer=cast(TokenizerLike, None),
        selected_percentiles=config.percentiles,
        goodput_config_dict={},
    )

    vllm_metrics = await scrape_vllm_metrics(config.server)

    return {
        "config": config,
        "pdf_results": pdf_results,
        "bench_metrics": bench_metrics,
        "vllm_metrics": vllm_metrics,
        "dur_s": dur_s,
        "all_outputs": all_outputs,
    }


# ---------------------------------------------------------------------------
# vLLM server-side Prometheus metrics
# ---------------------------------------------------------------------------

async def scrape_vllm_metrics(server: str) -> dict[str, float]:
    """Scrape /metrics from the vLLM server and return key gauge/counter values."""
    url = f"{server.rstrip('/')}/metrics"
    result: dict[str, float] = {}
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return result
                text = await resp.text()

        # Parse Prometheus text exposition format.
        # Lines like:  metric_name{labels} value timestamp
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for name in _PROM_METRICS:
                # Match exact metric name at start (ignore label variants)
                if re.match(rf"^{re.escape(name)}(\{{[^}}]*\}})?\s", line):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            result[name] = float(parts[1])
                        except ValueError:
                            pass
                    break
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def _pct_str(percentiles: list[tuple[float, float]]) -> str:
    return "  ".join(f"p{int(p)}={v:.1f}" for p, v in percentiles)


def print_report(
    config: BenchmarkConfig,
    bench_metrics: BenchmarkMetrics,
    pdf_results: list[PDFBenchmarkResult],
    vllm_metrics: dict[str, float],
    all_outputs: list[RequestFuncOutput],
) -> None:
    m = bench_metrics
    sep = "─" * 64

    print(f"\n{'═' * 64}")
    print(f"  OCR Benchmark  │  {config.label}")
    if config.notes:
        print(f"  Notes: {config.notes}")
    print(f"{'═' * 64}")

    num_pages = sum(pr.num_pages for pr in pdf_results)
    num_pdfs = len(pdf_results)

    print(f"\n  PDFs processed : {num_pdfs}")
    print(f"  Pages processed: {num_pages}")
    print(f"  Completed reqs : {m.completed}  Failed: {m.failed}")

    # --- Throughput ---
    print(f"\n{sep}")
    print("  THROUGHPUT")
    print(f"    output tokens/s     : {m.output_throughput:.1f}")
    print(f"    total tokens/s      : {m.total_token_throughput:.1f}")
    print(f"    requests/s          : {m.request_throughput:.3f}")
    print(f"    peak out tokens/s   : {m.max_output_tokens_per_s:.1f}  (max concurrent: {m.max_concurrent_requests})")

    # --- Latency ---
    print(f"\n{sep}")
    print("  LATENCY (ms)")
    print(f"    TTFT  mean={m.mean_ttft_ms:.1f}  median={m.median_ttft_ms:.1f}  std={m.std_ttft_ms:.1f}")
    print(f"          {_pct_str(m.percentiles_ttft_ms)}")
    print(f"    ITL   mean={m.mean_itl_ms:.1f}  median={m.median_itl_ms:.1f}  std={m.std_itl_ms:.1f}")
    print(f"          {_pct_str(m.percentiles_itl_ms)}")
    print(f"    TPOT  mean={m.mean_tpot_ms:.1f}  median={m.median_tpot_ms:.1f}  std={m.std_tpot_ms:.1f}")
    print(f"          {_pct_str(m.percentiles_tpot_ms)}")
    print(f"    E2EL  mean={m.mean_e2el_ms:.1f}  median={m.median_e2el_ms:.1f}  std={m.std_e2el_ms:.1f}")
    print(f"          {_pct_str(m.percentiles_e2el_ms)}")

    # --- PDF ready times ---
    render_times = [pr.render_time_s for pr in pdf_results]
    total_times = [pr.total_time_s for pr in pdf_results]
    print(f"\n{sep}")
    print("  PDF PROCESSING")
    print(f"    mean render time (PDF→vLLM ready) : {np.mean(render_times):.2f}s")
    print(f"    mean total time per PDF           : {np.mean(total_times):.2f}s")

    # --- Normalised page time ---
    norm_times = [
        pd.normalized_page_time_ms
        for pr in pdf_results
        for pd in pr.page_data
        if pd.request_output.success and pd.request_output.output_tokens
    ]
    if norm_times:
        print(f"\n{sep}")
        print("  NORMALISED PAGE TIME (E2EL ms / output token)")
        print(f"    mean={np.mean(norm_times):.2f}  median={np.median(norm_times):.2f}  "
              f"p90={np.percentile(norm_times, 90):.2f}  p99={np.percentile(norm_times, 99):.2f}")

    # --- vLLM server-side ---
    if vllm_metrics:
        print(f"\n{sep}")
        print("  vLLM SERVER METRICS (from /metrics)")
        for k, v in vllm_metrics.items():
            short = k.replace("vllm:", "")
            print(f"    {short:<45}: {v:.4g}")

    print(f"\n{'═' * 64}\n")


# ---------------------------------------------------------------------------
# Save results to disk
# ---------------------------------------------------------------------------

def _percentile_dict(
    percentiles: list[tuple[float, float]],
) -> dict[str, float]:
    return {f"p{int(p)}": round(v, 3) for p, v in percentiles}


def save_results(
    config: BenchmarkConfig,
    bench_metrics: BenchmarkMetrics,
    pdf_results: list[PDFBenchmarkResult],
    vllm_metrics: dict[str, float],
    all_outputs: list[RequestFuncOutput],
) -> Path:
    """Serialise benchmark results to metrics/{label}_{timestamp}.json."""
    metrics_dir = Path(config.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_label = re.sub(r"[^\w\-]", "_", config.label)
    out_path = metrics_dir / f"{safe_label}_{timestamp}.json"

    m = bench_metrics
    num_pages = sum(pr.num_pages for pr in pdf_results)
    render_times = [pr.render_time_s for pr in pdf_results]
    total_times = [pr.total_time_s for pr in pdf_results]

    norm_times = [
        pd.normalized_page_time_ms
        for pr in pdf_results
        for pd in pr.page_data
        if pd.request_output.success and pd.request_output.output_tokens
    ]

    payload = {
        "metadata": {
            "label": config.label,
            "notes": config.notes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "server": config.server,
            "model": config.model,
            "pdf_dir": config.pdf_dir,
            "num_pdfs": len(pdf_results),
            "num_pages": num_pages,
        },
        "throughput": {
            "output_tokens_per_s": round(m.output_throughput, 3),
            "total_tokens_per_s": round(m.total_token_throughput, 3),
            "requests_per_s": round(m.request_throughput, 4),
            "request_goodput_per_s": round(m.request_goodput, 4),
            "peak_output_tokens_per_s": round(m.max_output_tokens_per_s, 3),
            "max_concurrent_requests": m.max_concurrent_requests,
            "completed": m.completed,
            "failed": m.failed,
            "total_input_tokens": m.total_input,
            "total_output_tokens": m.total_output,
        },
        "latency_ms": {
            "ttft": {
                "mean": round(m.mean_ttft_ms, 3),
                "median": round(m.median_ttft_ms, 3),
                "std": round(m.std_ttft_ms, 3),
                **_percentile_dict(m.percentiles_ttft_ms),
            },
            "itl": {
                "mean": round(m.mean_itl_ms, 3),
                "median": round(m.median_itl_ms, 3),
                "std": round(m.std_itl_ms, 3),
                **_percentile_dict(m.percentiles_itl_ms),
            },
            "tpot": {
                "mean": round(m.mean_tpot_ms, 3),
                "median": round(m.median_tpot_ms, 3),
                "std": round(m.std_tpot_ms, 3),
                **_percentile_dict(m.percentiles_tpot_ms),
            },
            "e2el": {
                "mean": round(m.mean_e2el_ms, 3),
                "median": round(m.median_e2el_ms, 3),
                "std": round(m.std_e2el_ms, 3),
                **_percentile_dict(m.percentiles_e2el_ms),
            },
        },
        "pdf_metrics": {
            "mean_render_time_s": round(float(np.mean(render_times)), 4) if render_times else 0.0,
            "mean_total_time_s": round(float(np.mean(total_times)), 4) if total_times else 0.0,
            "per_pdf": [
                {
                    "path": pr.pdf_path,
                    "pages": pr.num_pages,
                    "render_time_s": round(pr.render_time_s, 4),
                    "total_time_s": round(pr.total_time_s, 4),
                }
                for pr in pdf_results
            ],
        },
        "normalized_page_time_ms_per_token": {
            "mean": round(float(np.mean(norm_times)), 3) if norm_times else 0.0,
            "median": round(float(np.median(norm_times)), 3) if norm_times else 0.0,
            "p90": round(float(np.percentile(norm_times, 90)), 3) if norm_times else 0.0,
            "p99": round(float(np.percentile(norm_times, 99)), 3) if norm_times else 0.0,
        },
        "vllm_server": {
            k: round(v, 6) for k, v in vllm_metrics.items()
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[benchmark] Results saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    pdf_dir: str,
    label: str,
    notes: str = "",
    server: str = DEFAULT_SERVER,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    pages_per_pdf: int = DEFAULT_PAGES_PER_PDF,
    target_image_dim: int = DEFAULT_TARGET_IMAGE_DIM,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    metrics_dir: str = DEFAULT_METRICS_DIR,
    api_key: str | None = None,
) -> None:
    """Run the OCR benchmark harness.

    Args:
        pdf_dir:         Directory containing *.pdf files to process.
        label:           Short name for this implementation/run (used in filenames).
        notes:           Free-text description of this run (hardware, config, etc.).
        server:          vLLM server base URL.
        model:           Model name as served by vLLM.
        concurrency:     Number of PDFs processed concurrently.
        pages_per_pdf:   Max in-flight page requests per PDF.
        target_image_dim: Longest image dimension for PDF rendering.
        max_tokens:      max_completion_tokens sent to the model.
        metrics_dir:     Directory to write result JSON files.
        api_key:         Bearer token for authenticated vLLM servers.
    """
    config = BenchmarkConfig(
        pdf_dir=pdf_dir,
        server=server,
        model=model,
        label=label,
        notes=notes,
        concurrency=concurrency,
        pages_per_pdf=pages_per_pdf,
        target_image_dim=target_image_dim,
        max_tokens=max_tokens,
        metrics_dir=metrics_dir,
        api_key=api_key,
    )

    results = asyncio.run(run_benchmark(config))

    print_report(
        config=config,
        bench_metrics=results["bench_metrics"],
        pdf_results=results["pdf_results"],
        vllm_metrics=results["vllm_metrics"],
        all_outputs=results["all_outputs"],
    )

    save_results(
        config=config,
        bench_metrics=results["bench_metrics"],
        pdf_results=results["pdf_results"],
        vllm_metrics=results["vllm_metrics"],
        all_outputs=results["all_outputs"],
    )


if __name__ == "__main__":
    fire.Fire(main)
