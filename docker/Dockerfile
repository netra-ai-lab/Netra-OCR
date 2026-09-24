# syntax=docker/dockerfile:1.7
#
# Netra OCR — web app + REST API with all model weights baked in (runs offline).
#
#   docker build -f docker/Dockerfile -t netra-ocr .
#   docker run -p 8000:8000 -v netra-data:/data netra-ocr
#   then open http://localhost:8000
#
# CPU only: the 17M-parameter recognizer and the YOLO detectors are small
# enough that a page takes a few seconds on an ordinary CPU, and leaving out
# CUDA keeps the image several gigabytes smaller.

ARG PYTHON_VERSION=3.12

# ── base: system libraries shared by the build and runtime stages ────────────
# tesseract-ocr(-khm): the optional Tesseract line detector works out of the box.
# No libgl1: only the headless OpenCV build is installed (see the prune step),
# which saves the ~200 MB Mesa/LLVM stack libgl1 drags in.
FROM python:${PYTHON_VERSION}-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 tesseract-ocr tesseract-ocr-khm && \
    rm -rf /var/lib/apt/lists/*

# ── build: virtualenv with CPU torch + netra-ocr[server], pruned ─────────────
FROM base AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU torch first, so nothing later pulls the CUDA build from PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.12.*" "torchvision==0.27.*"

WORKDIR /src
COPY pyproject.toml readme.md LICENSE MANIFEST.in requirements.txt ./
COPY netra_ocr ./netra_ocr
RUN pip install ".[server]"

# Prune what inference never imports (checked by docker/smoke_test.py below):
# - opencv-python (GUI build, needs libgl1) -> the same version, headless;
# - training/analysis-only deps of ultralytics / doclayout-yolo / netra-ocr:
#   polars, seaborn, albumentations, transformers (+ their own deps);
# - torch's C++ headers, test libraries and protoc; bundled test suites; pip.
# matplotlib and pandas stay: doclayout-yolo imports them at startup.
RUN SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    OCV=$(pip show opencv-python | awk '/^Version/{print $2}') && \
    pip uninstall -y opencv-python opencv-python-headless \
        polars polars-runtime-32 seaborn albumentations albucore \
        transformers tokenizers safetensors && \
    pip install --no-deps "opencv-python-headless==${OCV}" && \
    rm -rf "$SP"/torch/include "$SP"/torch/bin/protoc* "$SP"/torch/bin/test_* \
           "$SP"/torch/lib/libjitbackend_test.so "$SP"/torch/lib/libtorchbind_test.so \
           "$SP"/torch/lib/libbackend_with_compiler.so "$SP"/torch/lib/libaoti_custom_ops.so && \
    find "$SP" -depth -type d \( -name tests -o -name testing_data \) -path "*/scipy/*" -exec rm -rf {} + && \
    find "$SP" -depth -type d -name tests -path "*/pandas/*" -exec rm -rf {} + && \
    find /opt/venv -depth -name __pycache__ -type d -exec rm -rf {} + && \
    pip uninstall -y pip

# ── weights: everything the app loads, fetched once at build time ────────────
# Then run the whole pipeline once against the pruned environment: a prune
# that removed something needed fails here, not on the first upload.
FROM build AS weights
ENV HF_HOME=/models YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib
COPY docker/fetch_weights.py docker/smoke_test.py /tmp/
RUN python /tmp/fetch_weights.py && rm -rf /models/xet /models/hub/.locks && \
    cd /tmp && HF_HUB_OFFLINE=1 python /tmp/smoke_test.py

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
LABEL org.opencontainers.image.title="Netra OCR" \
      org.opencontainers.image.description="Khmer + English document OCR: layout analysis, recognition, editable structured export" \
      org.opencontainers.image.source="https://github.com/netra-ai-lab/Netra-OCR" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0-only"

RUN useradd --create-home --uid 10001 netra \
 && mkdir -p /data /home/netra/.config/ultralytics \
 && chown -R netra /data /home/netra/.config
COPY --from=build /opt/venv /opt/venv
COPY --from=weights /models /models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    NETRA_DATA_DIR=/data \
    NETRA_HOST=0.0.0.0 \
    NETRA_PORT=8000 \
    NETRA_DEVICE=cpu \
    YOLO_CONFIG_DIR=/home/netra/.config/ultralytics \
    YOLO_OFFLINE=1 \
    MPLCONFIGDIR=/home/netra/.config/matplotlib

USER netra
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('NETRA_PORT','8000'), timeout=4)"
CMD ["netra_ocr", "serve"]
