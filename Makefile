# OCR-LLM — development tasks
#
# Benchmark usage:
#   make benchmark LABEL="olmocr-fp8-baseline" NOTES="8xA100, batch=16, guided_decoding=off"
#
# Optional overrides:
#   PDF_DIR  — path to folder of PDFs (default: pdf_dir)
#   SERVER   — vLLM server URL        (default: http://127.0.0.1:8000)
#   MODEL    — served model name      (default: allenai/olmOCR-2-7B-1025-FP8)
#   MAX_CONCURRENT_REQUESTS — total vLLM requests in-flight across all PDFs (default: 64)
#   METRICS_DIR — output directory for JSON results (default: metrics)
#   PLOT_OUT    — benchmark comparison PNG path (default: comparison.png)

LABEL       ?= rust
NOTES       ?=
PDF_DIR     ?= pdfs
SERVER      ?= http://127.0.0.1:8000
MODEL       ?= allenai/olmOCR-2-7B-1025-FP8
MAX_CONCURRENT_REQUESTS ?= 128
METRICS_DIR             ?= metrics
PLOT_OUT                ?= comparison.png

# vLLM serve settings
GPU_MEMORY_UTILIZATION ?= 0.90
MAX_MODEL_LEN          ?= 16384
TENSOR_PARALLEL_SIZE   ?= 1
DATA_PARALLEL_SIZE     ?= 1

# ocr-save-md settings
WORKSPACE        ?= pdf_dir
WORKERS          ?= 20
APPLY_FILTER     ?= False
GUIDED_DECODING  ?= False
API_KEY          ?=
DISK_LOGGING     ?=

# Parse host and port out of SERVER so both benchmark and vllm-olmocr-serve
# share a single source of truth.
_SERVER_STRIPPED = $(shell echo "$(SERVER)" | sed 's|https\?://||' | sed 's|/$$||')
_VLLM_HOST       = $(shell echo "$(_SERVER_STRIPPED)" | cut -d: -f1)
_VLLM_PORT       = $(shell echo "$(_SERVER_STRIPPED)" | grep -o ':[0-9]*$$' | tr -d ':')

.PHONY: ocr-save-md
ocr-save-md:
	uv run python -c "\
import asyncio; \
from ocr_llm.olmo_ocr_pipeline import run_olmo_ocr; \
asyncio.run(run_olmo_ocr( \
    pdf_dir='$(PDF_DIR)', \
    workspace='$(WORKSPACE)', \
    server='$(SERVER)', \
    model='$(MODEL)', \
    workers=$(WORKERS), \
    max_concurrent_requests=$(MAX_CONCURRENT_REQUESTS), \
    tensor_parallel_size=$(TENSOR_PARALLEL_SIZE), \
    data_parallel_size=$(DATA_PARALLEL_SIZE), \
    gpu_memory_utilization=$(GPU_MEMORY_UTILIZATION), \
    max_model_len=$(MAX_MODEL_LEN), \
    apply_filter=$(APPLY_FILTER), \
    guided_decoding=$(GUIDED_DECODING), \
    markdown=True, \
    api_key='$(API_KEY)' or None, \
    disk_logging='$(DISK_LOGGING)' or None, \
))"

.PHONY: build-rust
build-rust:
	uv run maturin develop --release --manifest-path crates_ocr_render/Cargo.toml

.PHONY: benchmark

benchmark:
	uv run python -m ocr_llm.benchmark \
		--pdf_dir="$(PDF_DIR)" \
		--server="$(SERVER)" \
		--model="$(MODEL)" \
		--label="$(LABEL)" \
		--notes="$(NOTES)" \
		--max_concurrent_requests=$(MAX_CONCURRENT_REQUESTS) \
		--metrics_dir="$(METRICS_DIR)"

.PHONY: plot

plot:
	uv run python -m ocr_llm.plot \
		--metrics_dir="$(METRICS_DIR)" \
		--out="$(PLOT_OUT)"

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched-tokens 65536 \
		--max-num-seqs 256 \
		--limit-mm-per-prompt '{"video": 0}' \
		--host $(_VLLM_HOST) \
		--port $(_VLLM_PORT)

.PHONY: guidellm-benchmark
guidellm-benchmark:
	uv run guidellm benchmark \
		--target "http://localhost:$(_VLLM_PORT)" \
		--profile throughput \
		--max-seconds 300 \
		--rate 20 \
		--data "prompt_tokens=1024,output_tokens=4096" \
		--output-path benchmark.yaml