# ocr-llm

PDF → vision-OCR tooling around **olmocr** / **vLLM**, Rust-accelerated rendering (`ocr_render_rs`), and benchmark helpers.

**Setup:** [uv](https://github.com/astral-sh/uv) + `make`. From the repo root, install deps with `uv sync` (editable Rust crate via `tool.uv.sources`).

## Makefile targets

| Target | What it runs |
|--------|----------------|
| `build-rust` | `maturin develop --release` for `crates_ocr_render/` (Rust extension). |
| `vllm-olmocr-serve` | Starts `vllm serve` with `MODEL`; host/port are taken from `SERVER` so it matches benchmark clients. Tune GPU/length parallelism with `GPU_*`, `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`, `DATA_PARALLEL_SIZE`. |
| `benchmark` | `python -m ocr_llm.benchmark` over `PDF_DIR` against `SERVER`; writes one JSON per run under `METRICS_DIR` (label from `LABEL`, free text in `NOTES`). Concurrency: `MAX_CONCURRENT_REQUESTS`. |
| `plot` | `python -m ocr_llm.plot` — reads all `*.json` in `METRICS_DIR`, saves comparison figure to `PLOT_OUT` (default `comparison.png`). |
| `ocr-save-md` | Async OCR pipeline (`run_olmo_ocr`) with Markdown output; uses `PDF_DIR`, `SERVER`, `MODEL`, worker/concurrency knobs, plus `WORKSPACE`, `WORKERS`, `APPLY_FILTER`, `GUIDED_DECODING`, optional `API_KEY`, `DISK_LOGGING`. |
| `guidellm-benchmark` | [guidellm](https://github.com/vllm-project/guidellm) throughput-style run against localhost on the parsed vLLM port; writes `benchmark.yaml`. |

Overrides are ordinary Make variables, e.g. `make benchmark PDF_DIR=my_pdfs LABEL=run-a NOTES="single A100"` — see comments at the top of `Makefile` for defaults.

## Docker

GPU image (`Dockerfile`) that installs deps with **uv**, builds `ocr_render_rs`, and runs either the full OCR pipeline or a standalone vLLM server. Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`--gpus all`).

**Build:**

```bash
docker build -t ocr-llm .
```

**Run OCR pipeline** (default `pipeline` mode — downloads the model, starts internal vLLM, runs `run_olmo_ocr`):

```bash
docker run --gpus all --rm \
  -v "$PWD/pdfs:/data/pdfs:ro" \
  -v "$PWD/workspace:/data/workspace" \
  -v ocr-llm-hf-cache:/cache/huggingface \
  ocr-llm
```

**Run vLLM server only** (OpenAI-compatible API on port 8000):

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v ocr-llm-hf-cache:/cache/huggingface \
  ocr-llm serve
```

| Mode | Command | What it does |
|------|---------|--------------|
| `pipeline` | `ocr-llm` (default) | Sets `OLMOCR_LAUNCH_VLLM_FROM_SCRIPT=true`, downloads `MODEL`, starts vLLM on `PORT` (default 30024), OCRs PDFs from `PDF_DIR` into `WORKSPACE`. |
| `serve` | `ocr-llm serve` | Runs `vllm serve` only; listens on `PORT` (default 8000). |

Mount points: `/data/pdfs` (input PDFs), `/data/workspace` (OCR output), `/cache/huggingface` (model cache — use a named volume to avoid re-downloading).

Environment variables mirror the Makefile knobs: `PDF_DIR`, `WORKSPACE`, `MODEL`, `WORKERS`, `MAX_CONCURRENT_REQUESTS`, `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `TENSOR_PARALLEL_SIZE`, `DATA_PARALLEL_SIZE`, `APPLY_FILTER`, `GUIDED_DECODING`, `MARKDOWN`, `API_KEY`, `DISK_LOGGING`, `HF_TOKEN`. Set `SERVER` to point at an external vLLM instance instead of launching one in-container.
