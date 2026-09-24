"""Netra OCR web app and REST API.

Run with ``netra_ocr serve`` (or ``uvicorn netra_ocr.server.app:app``), then
open http://localhost:8000 for the UI or http://localhost:8000/docs for the
interactive API reference.
"""

import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Literal, Optional

from fastapi import Body, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..document import DocumentValidationError
from ..exporters import FORMATS, export
from ..io import count_pages, sniff_kind
from .jobs import JobManager, JobOptions, RevisionConflict
from .settings import Settings

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DETECTORS = ("yolo", "legacy", "tesseract")
DECODERS = ("ar", "blockwise")


def _package_version() -> str:
    try:
        return version("netra-ocr")
    except PackageNotFoundError:
        return "dev"


def _safe_basename(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename or "document"))[0]
    stem = re.sub(r"[^\w\-. ]+", "_", stem, flags=re.UNICODE).strip(" ._") or "document"
    return stem[:80]


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = JobManager(settings)
        if settings.warmup:
            app.state.jobs.warmup()
        else:
            app.state.jobs.models_ready.set()
        yield

    app = FastAPI(
        title="Netra OCR",
        version=_package_version(),
        description="Khmer + English document OCR: layout analysis, text recognition, "
                    "and structured export (DOCX, HTML, Markdown, JSON, TXT).",
        lifespan=lifespan,
    )

    def jobs() -> JobManager:
        return app.state.jobs

    # ── auth (only when NETRA_API_KEY is set) ──────────────────────────
    @app.middleware("http")
    async def require_key(request: Request, call_next):
        if settings.api_key and request.url.path.startswith("/v1/"):
            supplied = request.headers.get("x-api-key") or ""
            auth = request.headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                supplied = auth[7:]
            if not supplied and request.method == "GET":
                # <img src> and download links can't send headers.
                supplied = request.query_params.get("key", "")
            if not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
                return JSONResponse({"detail": "Missing or invalid API key."}, status_code=401)
        return await call_next(request)

    def job_or_404(job_id: str):
        try:
            return jobs().get(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found.")

    def document_or_error(job_id: str, original: bool = False):
        job_or_404(job_id)
        try:
            return jobs().document(job_id, original=original)
        except LookupError as e:
            raise HTTPException(409, f"Document not ready (job is {e}).")

    # ── meta ───────────────────────────────────────────────────────────
    @app.get("/health", include_in_schema=False)
    def health():
        return {"ok": True, "models_ready": jobs().models_ready.is_set()}

    @app.get("/v1/info", summary="Server, model and limit information")
    def info():
        import torch
        from ..recognition.config import _default_device, describe_device
        return {
            "version": _package_version(),
            "device": _default_device(),
            "device_name": describe_device(),
            "cuda": torch.cuda.is_available(),
            "models_ready": jobs().models_ready.is_set(),
            "detectors": list(DETECTORS),
            "decoders": list(DECODERS),
            "defaults": {"detector": settings.default_detector, "decoder": settings.default_decoder,
                         "layout": True},
            "formats": list(FORMATS),
            "limits": {"max_upload_mb": settings.max_upload_mb, "max_pdf_pages": settings.max_pdf_pages,
                       "job_ttl_hours": settings.job_ttl_hours},
            "auth": bool(settings.api_key),
        }

    # ── OCR ────────────────────────────────────────────────────────────
    @app.post("/v1/ocr", summary="Upload an image or PDF for OCR",
              responses={200: {"description": "Finished (single image, within the sync timeout)"},
                         202: {"description": "Accepted; poll GET /v1/jobs/{id}"}})
    async def ocr(
        file: UploadFile = File(..., description="PDF, JPG, PNG, TIFF, BMP or WebP"),
        layout: bool = Form(True, description="Structure the page (headings, paragraphs, tables, figures)"),
        detector: str = Form(None, description="Text-line detector: yolo | legacy | tesseract"),
        decoder: str = Form(None, description="Recognition decoder: ar (default) | blockwise"),
        include_headers_footers: bool = Form(True),
        beam_width: int = Form(1, ge=1, le=5, description="Beam width for the ar decoder (1 = greedy)"),
        conf: Optional[float] = Form(None, ge=0.01, le=0.99, description="YOLO line-detector confidence"),
        wait: bool = Form(True, description="For single images, wait for the result instead of returning 202"),
    ):
        detector = detector or settings.default_detector
        decoder = decoder or settings.default_decoder
        if detector not in DETECTORS:
            raise HTTPException(422, f"Unknown detector '{detector}'. Use one of {list(DETECTORS)}.")
        if decoder not in DECODERS:
            raise HTTPException(422, f"Unknown decoder '{decoder}'. Use one of {list(DECODERS)}.")

        data = await file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"File too large. The limit is {settings.max_upload_mb} MB.")
        kind = sniff_kind(data[:16])
        if kind is None:
            raise HTTPException(415, "Unsupported file. Upload a PDF, JPG, PNG, TIFF, BMP or WebP.")
        try:
            pages = count_pages(data)
        except Exception:
            raise HTTPException(422, "The file could not be read. It may be damaged or password-protected.")
        if pages < 1:
            raise HTTPException(422, "The document has no pages.")
        if pages > settings.max_pdf_pages:
            raise HTTPException(413, f"The document has {pages} pages; the limit is {settings.max_pdf_pages}.")

        options = JobOptions(layout=layout, detector=detector, decoder=decoder,
                             include_headers_footers=include_headers_footers,
                             beam_width=beam_width, conf=conf if detector == "yolo" else None)
        job = jobs().create(data, _safe_basename(file.filename), kind, pages, options)

        if wait and pages == 1:
            from starlette.concurrency import run_in_threadpool
            job = await run_in_threadpool(jobs().wait, job.id, settings.sync_timeout)
            if job.status == "done":
                return {"job": job.public(), "document": jobs().document(job.id).to_dict()}
            if job.status == "error":
                return JSONResponse({"job": job.public()}, status_code=500)
        return JSONResponse({"job": job.public()}, status_code=202)

    @app.get("/v1/jobs", summary="Recent jobs, newest first")
    def list_jobs(limit: int = Query(50, ge=1, le=500)):
        return {"jobs": [j.public() for j in jobs().list(limit)]}

    @app.get("/v1/jobs/{job_id}", summary="Job status and progress")
    def get_job(job_id: str):
        return {"job": job_or_404(job_id).public()}

    @app.delete("/v1/jobs/{job_id}", status_code=204, summary="Delete a job and all its files")
    def delete_job(job_id: str):
        job_or_404(job_id)
        jobs().delete(job_id)
        return Response(status_code=204)

    # ── document editing ───────────────────────────────────────────────
    @app.get("/v1/jobs/{job_id}/document", summary="The structured document (with any saved edits)")
    def get_document(job_id: str, original: bool = Query(False, description="Return the unedited OCR result")):
        doc = document_or_error(job_id, original)
        return JSONResponse(doc.to_dict(), headers={"ETag": f'"{doc.revision}"'})

    @app.put("/v1/jobs/{job_id}/document", summary="Save an edited document",
             description="Send the whole document. Pass the revision you edited in `If-Match` "
                         "(or the body's `revision`) to get 409 instead of overwriting a newer save.")
    def put_document(job_id: str, body: dict = Body(...), if_match: Optional[str] = Header(None)):
        job_or_404(job_id)
        expected = None
        raw = (if_match or "").strip().strip('"') or body.get("revision")
        if raw is not None and raw != "":
            try:
                expected = int(raw)
            except (TypeError, ValueError):
                raise HTTPException(400, "If-Match must be a document revision number.")
        try:
            doc = jobs().save_document(job_id, body, expected)
        except LookupError as e:
            raise HTTPException(409, f"Document not ready (job is {e}).")
        except RevisionConflict as c:
            return JSONResponse({"detail": "The document was changed elsewhere.",
                                 "current": c.current.to_dict()}, status_code=409)
        except DocumentValidationError as e:
            raise HTTPException(422, str(e))
        return JSONResponse({"revision": doc.revision}, headers={"ETag": f'"{doc.revision}"'})

    @app.post("/v1/jobs/{job_id}/document/reset", summary="Discard edits and restore the OCR result")
    def reset_document(job_id: str):
        document_or_error(job_id)
        return jobs().reset_document(job_id).to_dict()

    @app.get("/v1/jobs/{job_id}/export", summary="Download the document",
             description="Exports the edited document. `md` becomes a .zip when the document has figures.")
    def export_document(job_id: str, format: Literal["docx", "html", "md", "json", "txt"] = "docx"):
        doc = document_or_error(job_id)
        job = jobs().get(job_id)
        data, filename, mime = export(doc, format, basename=job.filename)
        from urllib.parse import quote
        return Response(data, media_type=mime, headers={
            "Content-Disposition": f"attachment; filename=\"document{os.path.splitext(filename)[1]}\"; "
                                   f"filename*=UTF-8''{quote(filename)}"})

    @app.get("/v1/jobs/{job_id}/assets/{asset_id}", summary="A page render or figure/table crop")
    def get_asset(job_id: str, asset_id: str):
        document_or_error(job_id)
        path = jobs().asset_path(job_id, asset_id)
        if path is None:
            raise HTTPException(404, "Asset not found.")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=86400, immutable"})

    # ── web UI ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
