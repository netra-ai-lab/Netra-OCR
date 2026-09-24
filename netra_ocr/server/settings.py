"""Server configuration, read from NETRA_* environment variables."""

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Where jobs (uploads, page renders, documents) live.
    data_dir: str = field(default_factory=lambda: os.environ.get(
        "NETRA_DATA_DIR", os.path.join(os.path.expanduser("~"), ".cache", "netra-ocr", "jobs")))
    max_upload_mb: int = field(default_factory=lambda: _env_int("NETRA_MAX_UPLOAD_MB", 50))
    max_pdf_pages: int = field(default_factory=lambda: _env_int("NETRA_MAX_PDF_PAGES", 100))
    # Jobs (and their edits) are deleted this long after their last change.
    job_ttl_hours: int = field(default_factory=lambda: _env_int("NETRA_JOB_TTL_HOURS", 24))
    # Optional. Unset (the default) means no key: anyone who can reach the
    # port can use it. Set it before exposing an instance beyond your machine.
    api_key: str | None = field(default_factory=lambda: os.environ.get("NETRA_API_KEY") or None)
    default_detector: str = field(default_factory=lambda: os.environ.get("NETRA_DETECTOR", "yolo"))
    default_decoder: str = field(default_factory=lambda: os.environ.get("NETRA_DECODER", "ar"))
    # Load models at startup so the first request isn't slow.
    warmup: bool = field(default_factory=lambda: _env_bool("NETRA_WARMUP", True))
    # Seconds POST /v1/ocr waits for a single image before answering 202.
    sync_timeout: int = field(default_factory=lambda: _env_int("NETRA_SYNC_TIMEOUT", 120))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
