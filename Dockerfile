# GPU image for olmOCR: downloads the model, starts an internal vLLM server,
# and runs the async OCR pipeline (run_olmo_ocr).
#
# Build:
#   docker build -t ocr-llm .
#
# Run OCR pipeline (mount PDFs + output workspace + HF cache):
#   docker run --gpus all --rm \
#     -v "$PWD/pdfs:/data/pdfs:ro" \
#     -v "$PWD/workspace:/data/workspace" \
#     -v ocr-llm-hf-cache:/cache/huggingface \
#     ocr-llm
#
# Run vLLM server only (OpenAI-compatible API on port 8000):
#   docker run --gpus all --rm -p 8000:8000 \
#     -v ocr-llm-hf-cache:/cache/huggingface \
#     ocr-llm serve

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON=3.12 \
    HF_HOME=/cache/huggingface \
    PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY crates_ocr_render/ crates_ocr_render/
COPY ocr_llm/ ocr_llm/

RUN uv sync --frozen --no-dev

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/data/pdfs", "/data/workspace", "/cache/huggingface"]

EXPOSE 30024 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["pipeline"]
