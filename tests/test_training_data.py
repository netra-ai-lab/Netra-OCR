"""The training data specs: every local layout reads the same lines, bad input
gets a clear error. Needs the training package (pip install -e training)."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
data = pytest.importorskip("netra_ocr_train.data")

EXAMPLE = Path(__file__).resolve().parent.parent / "training" / "examples"


def _lines(rows):
    return [(Path(p).name, label) for p, label in zip(rows.paths, rows.labels)]


@pytest.fixture
def expected():
    return _lines(data.load_source(str(EXAMPLE / "lines")))


def test_example_folder(expected):
    rows = data.load_source(str(EXAMPLE / "lines"))
    assert len(rows) == 24 and rows.kind == "folder: images/ + labels/" and not rows.problems
    assert rows[0]["image"].mode == "RGB"


def test_manifests_read_the_same_lines(tmp_path, expected):
    shutil.copytree(EXAMPLE / "lines", tmp_path / "lines")
    tsv = (EXAMPLE / "lines.tsv").read_text(encoding="utf-8").splitlines()
    pairs = [line.split("\t", 1) for line in tsv]
    (tmp_path / "a.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("file_name,transcription\n" + "".join(f'{p},"{t}"\n' for p, t in pairs),
                                    encoding="utf-8")
    (tmp_path / "a.jsonl").write_text("".join(json.dumps({"image": p, "text": t}, ensure_ascii=False) + "\n"
                                              for p, t in pairs), encoding="utf-8")
    for name in ("a.tsv", "a.csv", "a.jsonl"):
        assert _lines(data.load_source(str(tmp_path / name))) == expected, name


def test_flat_folder_with_gt_txt(tmp_path, expected):
    for image in sorted((EXAMPLE / "lines" / "images").iterdir()):
        shutil.copy(image, tmp_path / image.name)
        shutil.copy(EXAMPLE / "lines" / "labels" / f"{image.stem}.txt", tmp_path / f"{image.stem}.gt.txt")
    rows = data.load_source(str(tmp_path))
    assert rows.kind == "folder: image + .txt beside it"
    assert _lines(rows) == expected


def test_problems_are_reported_not_fatal(tmp_path):
    shutil.copytree(EXAMPLE / "lines", tmp_path / "lines")
    (tmp_path / "lines" / "labels" / "0001.txt").unlink()
    (tmp_path / "lines" / "labels" / "0002.txt").write_text("  \n", encoding="utf-8")
    rows = data.load_source(str(tmp_path / "lines"))
    assert len(rows) == 22
    assert rows.problems == ["no label for 0001.jpg", "empty label 0002.txt"]


def test_bad_input_explains_the_formats(tmp_path):
    (tmp_path / "bad.tsv").write_text("images/0001.png the text without a tab\n", encoding="utf-8")
    with pytest.raises(data.DataError, match="no tab"):
        data.load_source(str(tmp_path / "bad.tsv"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(data.DataError, match="No labelled lines(.|\n)*a manifest"):
        data.load_source(str(tmp_path / "empty"))
    with pytest.raises(data.DataError, match="isn't a folder"):
        data.load_source("not a dataset")


def test_hugging_face_specs_and_presets():
    m = data._HF_SPEC.match("Netra-Lab/khmer-document-synthetic-low-res:train[:5000]:label")
    assert (m["repo"], m["split"], m["column"]) == ("Netra-Lab/khmer-document-synthetic-low-res", "train[:5000]", "label")
    assert data._HF_SPEC.match("user/name")["split"] is None
    assert data.resolve("khob") == ["Netra-Lab/khob-eval"]
    assert data.resolve("synthetic") == data.DEFAULT_TRAIN_DATA


def test_small_sets_still_get_a_validation_split():
    train, val = data.split_off(list(range(24)))
    assert (len(train), len(val)) == (20, 4)
    train, val = data.split_off(list(range(100_000)))
    assert len(val) == 500
