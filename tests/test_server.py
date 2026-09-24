"""API tests with a fake OCR pipeline (no models are loaded)."""

import io
import time

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from netra_ocr.document import Document  # noqa: E402
from netra_ocr.server import jobs as jobs_mod  # noqa: E402
from netra_ocr.server.app import create_app  # noqa: E402
from netra_ocr.server.settings import Settings  # noqa: E402


class FakePipeline:
    def process_document(self, source, progress=None, **_):
        doc = Document()
        doc.assets.put("page-0", Image.new("RGB", (100, 140), "white"))
        doc.pages.append({"index": 0, "width": 100, "height": 140, "image": "page-0"})
        doc.blocks = [
            {"id": "a", "type": "heading", "level": 1, "text": "ចំណងជើង", "page": 0, "bbox": [10, 10, 90, 30]},
            {"id": "b", "type": "paragraph", "text": "Body", "page": 0, "bbox": [10, 40, 90, 80]},
        ]
        if progress:
            progress(1, 1)
        return doc


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (60, 40), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod.JobManager, "_pipeline", lambda self, options: FakePipeline())
    settings = Settings(data_dir=str(tmp_path), warmup=False, api_key=None)
    with TestClient(create_app(settings)) as c:
        yield c


def test_image_roundtrip_edit_and_export(client):
    r = client.post("/v1/ocr", files={"file": ("scan.png", png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    job_id = r.json()["job"]["id"]
    assert [b["type"] for b in r.json()["document"]["blocks"]] == ["heading", "paragraph"]

    doc = client.get(f"/v1/jobs/{job_id}/document")
    assert doc.headers["etag"] == '"0"'
    body = doc.json()
    body["blocks"][1]["text"] = "Edited body"
    assert client.put(f"/v1/jobs/{job_id}/document", json=body, headers={"If-Match": '"0"'}).json() == {"revision": 1}
    # A second save based on the old revision is a conflict, not an overwrite.
    stale = client.put(f"/v1/jobs/{job_id}/document", json=body, headers={"If-Match": '"0"'})
    assert stale.status_code == 409 and stale.json()["current"]["revision"] == 1

    txt = client.get(f"/v1/jobs/{job_id}/export", params={"format": "txt"})
    assert "Edited body" in txt.text
    assert "filename*=UTF-8''scan.txt" in txt.headers["content-disposition"]
    assert client.get(f"/v1/jobs/{job_id}/export", params={"format": "docx"}).content[:2] == b"PK"
    assert client.get(f"/v1/jobs/{job_id}/assets/page-0").headers["content-type"] == "image/jpeg"

    reset = client.post(f"/v1/jobs/{job_id}/document/reset").json()
    assert reset["blocks"][1]["text"] == "Body" and reset["revision"] == 2
    assert client.get("/v1/jobs").json()["jobs"][0]["id"] == job_id


def test_async_pdf_style_flow(client):
    r = client.post("/v1/ocr", files={"file": ("scan.png", png_bytes(), "image/png")}, data={"wait": "false"})
    assert r.status_code == 202
    job_id = r.json()["job"]["id"]
    for _ in range(50):
        job = client.get(f"/v1/jobs/{job_id}").json()["job"]
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["progress"] == {"done": 1, "total": 1}
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_rejects_bad_input(client):
    assert client.post("/v1/ocr", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/v1/ocr", files={"file": ("a.png", png_bytes(), "image/png")},
                       data={"detector": "nope"}).status_code == 422
    assert client.get("/v1/jobs/not-a-job").status_code == 404
    r = client.post("/v1/ocr", files={"file": ("scan.png", png_bytes(), "image/png")})
    job_id = r.json()["job"]["id"]
    assert client.get(f"/v1/jobs/{job_id}/assets/..%2Fjob").status_code == 404
    bad = {"pages": [], "blocks": [{"id": "x", "type": "script", "text": "", "page": 0}]}
    assert client.put(f"/v1/jobs/{job_id}/document", json=bad).status_code == 422


def test_api_key(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod.JobManager, "_pipeline", lambda self, options: FakePipeline())
    with TestClient(create_app(Settings(data_dir=str(tmp_path), warmup=False, api_key="s3cret"))) as c:
        assert c.get("/v1/jobs").status_code == 401
        assert c.get("/v1/jobs", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/v1/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/v1/jobs?key=s3cret").status_code == 200
        assert c.get("/health").status_code == 200          # health and UI stay open
        assert c.get("/").status_code == 200
