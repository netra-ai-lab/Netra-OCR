"""OCR jobs: an on-disk store plus a single inference worker thread.

One worker owns every model, so inference is serialized: a single GPU (or
the CPU) is used by one job at a time and torch models are never shared
across threads. Jobs are plain directories, so they survive restarts::

    <data_dir>/<job_id>/
        job.json             status, progress, options, original filename
        upload.bin           the uploaded file (deleted once processed)
        document.ocr.json    the OCR result as produced
        document.json        the current (possibly edited) document
        assets/              page renders + figure/table crops
"""

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..document import Document
from .settings import Settings

logger = logging.getLogger(__name__)

_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass
class JobOptions:
    layout: bool = True
    detector: str = "yolo"
    decoder: str = "ar"
    include_headers_footers: bool = True
    beam_width: int = 1
    conf: Optional[float] = None


@dataclass
class Job:
    id: str
    filename: str
    kind: str                       # "pdf" or an image kind
    options: JobOptions
    status: str = "queued"          # queued | running | done | error
    pages_done: int = 0
    pages_total: int = 0
    error: Optional[str] = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    edited: bool = False

    def public(self) -> dict:
        d = asdict(self)
        d["progress"] = {"done": self.pages_done, "total": self.pages_total}
        del d["pages_done"], d["pages_total"]
        return d


class JobManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir
        os.makedirs(self.root, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._doc_locks: dict[str, threading.Lock] = {}
        self._queue: "queue.Queue[str | None]" = queue.Queue()
        self._pipelines: dict = {}
        self._done = threading.Condition(self._lock)
        self.models_ready = threading.Event()
        self._load_existing()
        self._worker = threading.Thread(target=self._run, name="netra-ocr-worker", daemon=True)
        self._worker.start()
        self._janitor = threading.Thread(target=self._cleanup_loop, name="netra-ocr-janitor", daemon=True)
        self._janitor.start()

    # ── persistence ────────────────────────────────────────────────────
    def _dir(self, job_id: str) -> str:
        if not _JOB_ID.match(job_id):
            raise KeyError(job_id)
        return os.path.join(self.root, job_id)

    def _write_meta(self, job: Job) -> None:
        job.updated = time.time()
        path = os.path.join(self._dir(job.id), "job.json")
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(asdict(job), f, ensure_ascii=False)
        os.replace(path + ".tmp", path)

    def _load_existing(self) -> None:
        for name in os.listdir(self.root):
            if not _JOB_ID.match(name):
                continue
            try:
                with open(os.path.join(self.root, name, "job.json"), encoding="utf-8") as f:
                    raw = json.load(f)
                raw["options"] = JobOptions(**raw.get("options", {}))
                job = Job(**raw)
            except Exception:
                logger.warning("Skipping unreadable job directory %s", name)
                continue
            if job.status in ("queued", "running"):
                job.status, job.error = "error", "Interrupted by a server restart. Please upload again."
                self._write_meta(job)
            self._jobs[job.id] = job

    # ── public API ─────────────────────────────────────────────────────
    def create(self, data: bytes, filename: str, kind: str, pages: int, options: JobOptions) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename, kind=kind, options=options, pages_total=pages)
        d = self._dir(job.id)
        os.makedirs(d)
        with open(os.path.join(d, "upload.bin"), "wb") as f:
            f.write(data)
        with self._lock:
            self._jobs[job.id] = job
            self._write_meta(job)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)
        return jobs[:limit]

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            raise KeyError(job_id)
        shutil.rmtree(self._dir(job_id), ignore_errors=True)

    def wait(self, job_id: str, timeout: float) -> Job:
        deadline = time.time() + timeout
        with self._done:
            while True:
                job = self._jobs.get(job_id)
                if job is None or job.status in ("done", "error"):
                    return job
                remaining = deadline - time.time()
                if remaining <= 0:
                    return job
                self._done.wait(remaining)

    def _doc_lock(self, job_id: str) -> threading.Lock:
        with self._lock:
            return self._doc_locks.setdefault(job_id, threading.Lock())

    def document(self, job_id: str, original: bool = False) -> Document:
        job = self.get(job_id)
        if job.status != "done":
            raise LookupError(job.status)
        return Document.load(self._dir(job_id), "document.ocr.json" if original else "document.json")

    def save_document(self, job_id: str, data: dict, expected_revision: Optional[int]) -> Document:
        """Validate and store an edited document. Raises RevisionConflict on a stale save."""
        with self._doc_lock(job_id):
            current = self.document(job_id)
            if expected_revision is not None and expected_revision != current.revision:
                raise RevisionConflict(current)
            doc = Document.from_dict(data, current.assets)
            doc.revision = current.revision + 1
            doc.save(self._dir(job_id))
            job = self.get(job_id)
            with self._lock:
                job.edited = True
                self._write_meta(job)
            return doc

    def reset_document(self, job_id: str) -> Document:
        with self._doc_lock(job_id):
            current = self.document(job_id)
            doc = self.document(job_id, original=True)
            doc.revision = current.revision + 1
            doc.save(self._dir(job_id))
            job = self.get(job_id)
            with self._lock:
                job.edited = False
                self._write_meta(job)
            return doc

    def asset_path(self, job_id: str, asset_id: str) -> Optional[str]:
        self.get(job_id)
        return self.document(job_id).assets.path(asset_id)

    # ── worker ─────────────────────────────────────────────────────────
    def _pipeline(self, options: JobOptions):
        from ..ocr_engine import KhmerOCRPipeline
        key = (options.detector, options.layout, options.decoder, options.conf)
        if key not in self._pipelines:
            self._pipelines[key] = KhmerOCRPipeline(detector=options.detector, conf=options.conf,
                                                    decoder=options.decoder, layout=options.layout)
        return self._pipelines[key]

    def warmup(self) -> None:
        """Load the default models in the worker thread (runs before any job)."""
        self._queue.put(None)

    def _run(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                self._do_warmup()
                continue
            try:
                self._process(job_id)
            except Exception:
                logger.exception("Job %s crashed", job_id)

    def _do_warmup(self) -> None:
        try:
            from ..recognition.config import describe_device
            from ..recognition.recognize_text import _get_predictor
            s = self.settings
            self._pipeline(JobOptions(layout=True, detector=s.default_detector, decoder=s.default_decoder))
            _get_predictor(decoder=s.default_decoder)
            logger.info("Models loaded on %s", describe_device())
        except Exception:
            logger.exception("Model warmup failed; models will load on the first job")
        finally:
            self.models_ready.set()

    def _process(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            self._write_meta(job)
        d = self._dir(job_id)
        upload = os.path.join(d, "upload.bin")

        def progress(done: int, total: int) -> None:
            with self._lock:
                job.pages_done, job.pages_total = done, total
                self._write_meta(job)

        try:
            o = job.options
            pipeline = self._pipeline(o)
            doc = pipeline.process_document(upload, beam_width=o.beam_width,
                                            include_headers_footers=o.include_headers_footers,
                                            max_pages=self.settings.max_pdf_pages, progress=progress)
            doc.save(d, "document.ocr.json")
            # The editable copy shares the same assets directory.
            shutil.copyfile(os.path.join(d, "document.ocr.json"), os.path.join(d, "document.json"))
            status, error = "done", None
        except Exception as e:
            logger.exception("OCR failed for job %s", job_id)
            status, error = "error", f"{type(e).__name__}: {e}"
        finally:
            try:
                os.remove(upload)
            except OSError:
                pass
        with self._done:
            job.status, job.error = status, error
            self._write_meta(job)
            self._done.notify_all()

    # ── retention ──────────────────────────────────────────────────────
    def _cleanup_loop(self) -> None:
        while True:
            time.sleep(600)
            self.cleanup()

    def cleanup(self) -> None:
        cutoff = time.time() - self.settings.job_ttl_hours * 3600
        for job in self.list(limit=10_000):
            if job.status in ("done", "error") and job.updated < cutoff:
                try:
                    self.delete(job.id)
                except KeyError:
                    pass


class RevisionConflict(Exception):
    def __init__(self, current: Document):
        super().__init__("document was changed elsewhere")
        self.current = current
