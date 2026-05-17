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
