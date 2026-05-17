# OCR-LLM — development tasks
#
# Benchmark usage:
#   make benchmark LABEL="olmocr-fp8-baseline" NOTES="8xA100, batch=16, guided_decoding=off"
#
# Optional overrides:
#   PDF_DIR  — path to folder of PDFs (default: pdf_dir)
#   SERVER   — vLLM server URL        (default: http://127.0.0.1:8000)
#   MODEL    — served model name      (default: allenai/olmOCR-2-7B-1025-FP8)
#   CONCURRENCY — PDFs processed in parallel (default: 4)
#   METRICS_DIR — output directory for JSON results (default: metrics)

LABEL       ?= default
NOTES       ?=
PDF_DIR     ?= pdfs
SERVER      ?= http://127.0.0.1:8000
MODEL       ?= allenai/olmOCR-2-7B-1025-FP8
CONCURRENCY ?= 4
METRICS_DIR ?= metrics

# vLLM serve settings
GPU_MEMORY_UTILIZATION ?= 0.90
MAX_MODEL_LEN          ?= 16384
TENSOR_PARALLEL_SIZE   ?= 1
DATA_PARALLEL_SIZE     ?= 1

# Parse host and port out of SERVER so both benchmark and vllm-olmocr-serve
# share a single source of truth.
_SERVER_STRIPPED = $(shell echo "$(SERVER)" | sed 's|https\?://||' | sed 's|/$$||')
_VLLM_HOST       = $(shell echo "$(_SERVER_STRIPPED)" | cut -d: -f1)
_VLLM_PORT       = $(shell echo "$(_SERVER_STRIPPED)" | grep -o ':[0-9]*$$' | tr -d ':')

.PHONY: benchmark

benchmark:
	uv run python -m ocr_llm.benchmark \
		--pdf_dir="$(PDF_DIR)" \
		--server="$(SERVER)" \
		--model="$(MODEL)" \
		--label="$(LABEL)" \
		--notes="$(NOTES)" \
		--concurrency=$(CONCURRENCY) \
		--metrics_dir="$(METRICS_DIR)"

.PHONY: vllm-olmocr-serve
vllm-olmocr-serve:
	uv run vllm serve $(MODEL) \
		--gpu-memory-utilization $(GPU_MEMORY_UTILIZATION) \
		--max-model-len $(MAX_MODEL_LEN) \
		--tensor-parallel-size $(TENSOR_PARALLEL_SIZE) \
		--data-parallel-size $(DATA_PARALLEL_SIZE) \
		--max-num-batched-tokens 65536 \
		--max-num-seqs 128 \
		--limit-mm-per-prompt '{"video": 0}' \
		--host $(_VLLM_HOST) \
		--port $(_VLLM_PORT)