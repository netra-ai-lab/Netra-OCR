# syntax=docker/dockerfile:1.7
#
# Netra OCR — web app + REST API with all model weights baked in (runs offline).
#
#   docker build -f docker/Dockerfile -t netra-ocr .
#   docker run -p 8000:8000 -v netra-data:/data netra-ocr
#   then open http://localhost:8000
#
# CPU only, and no PyTorch: every model runs on ONNX Runtime (netra_ocr/inference).
# A page takes about a second on an ordinary CPU.

ARG PYTHON_VERSION=3.12

# ── base: system libraries shared by the build and runtime stages ────────────
# tesseract-ocr(-khm): the optional Tesseract line detector works out of the box.
# No libgl1: only the headless OpenCV build is installed (see the prune step),
# which saves the ~200 MB Mesa/LLVM stack libgl1 drags in.
FROM python:${PYTHON_VERSION}-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 tesseract-ocr tesseract-ocr-khm && \
    rm -rf /var/lib/apt/lists/*

# ── build: virtualenv with netra-ocr[server] on ONNX Runtime (CPU), pruned ──
FROM base AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml readme.md LICENSE MANIFEST.in requirements.txt ./
COPY netra_ocr ./netra_ocr
RUN pip install ".[server,cpu]"

# Prune what inference never needs (checked by docker/smoke_test.py below):
# opencv-python (GUI build, needs libgl1) -> the same version, headless;
# bundled test suites; pip.
RUN SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    OCV=$(pip show opencv-python | awk '/^Version/{print $2}') && \
    pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-deps "opencv-python-headless==${OCV}" && \
    find "$SP" -depth -type d \( -name tests -o -name testing_data \) -path "*/scipy/*" -exec rm -rf {} + && \
    find "$SP" -depth -type d -name tests -path "*/skimage/*" -exec rm -rf {} + && \
    find /opt/venv -depth -name __pycache__ -type d -exec rm -rf {} + && \
    pip uninstall -y pip

# ── weights: the model bundles, fetched once at build time ──────────────────
# Then run the whole pipeline once against the pruned environment, offline:
# a prune that removed something needed fails here, not on the first upload.
FROM build AS weights
COPY docker/fetch_weights.py docker/smoke_test.py /tmp/
RUN HF_HOME=/tmp/hf python /tmp/fetch_weights.py /models/netra-ocr && rm -rf /tmp/hf && \
    cd /tmp && NETRA_BUNDLES=/models/netra-ocr HF_HUB_OFFLINE=1 python /tmp/smoke_test.py

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
LABEL org.opencontainers.image.title="Netra OCR" \
      org.opencontainers.image.description="Khmer + English document OCR: layout analysis, recognition, editable structured export" \
      org.opencontainers.image.source="https://github.com/netra-ai-lab/Netra-OCR" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0-only"

RUN useradd --create-home --uid 10001 netra \
 && mkdir -p /data \
 && chown -R netra /data
COPY --from=build /opt/venv /opt/venv
COPY --from=weights /models /models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NETRA_BUNDLES=/models/netra-ocr \
    HF_HUB_OFFLINE=1 \
    NETRA_DATA_DIR=/data \
    NETRA_HOST=0.0.0.0 \
    NETRA_PORT=8000 \
    NETRA_DEVICE=cpu

USER netra
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('NETRA_PORT','8000'), timeout=4)"
CMD ["netra_ocr", "serve"]
