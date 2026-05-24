#!/usr/bin/env bash
set -euo pipefail

cd /app

MODE="${1:-pipeline}"

if [[ "${MODE}" == "serve" ]]; then
  exec uv run vllm serve "${MODEL:-allenai/olmOCR-2-7B-1025-FP8}" \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --data-parallel-size "${DATA_PARALLEL_SIZE:-1}" \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 256 \
    --limit-mm-per-prompt '{"video": 0}'
fi

if [[ "${MODE}" == "pipeline" ]]; then
  export OLMOCR_LAUNCH_VLLM_FROM_SCRIPT=true
  exec uv run python - <<'PY'
import asyncio
import os

from ocr_llm.olmo_ocr_pipeline import run_olmo_ocr


def env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


async def main() -> None:
    server = os.getenv("SERVER")
    if server is None or server.strip() == "" or server.strip().lower() == "none":
        server = None

    await run_olmo_ocr(
        pdf_dir=os.getenv("PDF_DIR", "/data/pdfs"),
        workspace=os.getenv("WORKSPACE", "/data/workspace"),
        model=os.getenv("MODEL", "allenai/olmOCR-2-7B-1025-FP8"),
        workers=env_int("WORKERS", 20),
        max_concurrent_requests=env_int("MAX_CONCURRENT_REQUESTS", 128),
        apply_filter=env_bool("APPLY_FILTER", False),
        guided_decoding=env_bool("GUIDED_DECODING", False),
        markdown=env_bool("MARKDOWN", True),
        server=server,
        api_key=os.getenv("API_KEY") or None,
        disk_logging=os.getenv("DISK_LOGGING") or None,
        gpu_memory_utilization=env_float("GPU_MEMORY_UTILIZATION", 0.90),
        max_model_len=env_int("MAX_MODEL_LEN", 16384),
        tensor_parallel_size=env_int("TENSOR_PARALLEL_SIZE", 1),
        data_parallel_size=env_int("DATA_PARALLEL_SIZE", 1),
        port=env_int("PORT", 30024),
    )


asyncio.run(main())
PY
fi

echo "Unknown mode: ${MODE}. Use 'pipeline' (default) or 'serve'." >&2
exit 1
