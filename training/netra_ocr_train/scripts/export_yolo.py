"""Export an Ultralytics-style YOLO model as a model folder the app runs.

Works with any model object that has Ultralytics' ``.export(format="onnx", ...)``
and ``.names`` -- ``ultralytics.YOLO`` and forks such as ``doclayout_yolo.YOLOv10``.

Usage:
    python -m netra_ocr_train.scripts.export_yolo \
        training/weights/yolo26s_best/best.pt bundles/lines --name netra-ocr-lines --license AGPL-3.0
    python -m netra_ocr_train.scripts.export_yolo doclayout_docstructbench.pt bundles/layout \
        --name netra-ocr-layout --imgsz 1024 --conf 0.2 --loader doclayout_yolo:YOLOv10
"""

import argparse
import importlib
import shutil
import tempfile
from pathlib import Path

from ..model_folder import write_config

OPSET = 18


def _output_shape(onnx_path: Path) -> list:
    import onnx
    output = onnx.load(str(onnx_path), load_external_data=False).graph.output[0]
    return [d.dim_param or d.dim_value for d in output.type.tensor_type.shape.dim]


def export_yolo(model, out_dir: str | Path, name: str, version: str, imgsz: int = 640,
                conf: float = 0.25, iou: float = 0.7, stride: int = 32, license: str | None = None,
                metadata: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [model.names[i] for i in sorted(model.names)]

    with tempfile.TemporaryDirectory() as tmp:
        # Ultralytics writes the .onnx next to the weights file, so export from a copy.
        weights = Path(tmp) / "model.pt"
        shutil.copyfile(model.ckpt_path, weights)
        exported = type(model)(str(weights)).export(
            format="onnx", dynamic=True, simplify=False, opset=OPSET, imgsz=imgsz)
        shutil.copyfile(exported, out_dir / "model.onnx")

    shape = _output_shape(out_dir / "model.onnx")
    # (batch, max_det, 6) = NMS-free end-to-end head; (batch, 4 + classes, anchors) = raw.
    if shape[-1] == 6 and isinstance(shape[1], int):
        head = {"type": "yolo-end2end", "labels": labels, "conf": conf}
    else:
        head = {"type": "yolo-nms", "labels": labels, "conf": conf, "iou": iou}

    write_config(out_dir, {
        "name": name,
        "version": version,
        "task": "object-detection",
        "graphs": {"model": "model.onnx"},
        "components": {
            # Dynamic input shapes: pad only to a multiple of the stride, like
            # Ultralytics' own rectangular predict-time letterbox.
            "preprocess": {"type": "letterbox", "imgsz": imgsz, "stride": stride, "auto": True,
                           "pad_value": 114},
            "head": head,
        },
        "metadata": {"license": license, "source_checkpoint": Path(model.ckpt_path).name,
                     **(metadata or {})},
    })
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights")
    parser.add_argument("out")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--license", default=None)
    parser.add_argument("--loader", default="ultralytics:YOLO",
                        help="module:class that loads the weights (e.g. doclayout_yolo:YOLOv10)")
    args = parser.parse_args(argv)

    module, _, cls = args.loader.partition(":")
    model = getattr(importlib.import_module(module), cls)(args.weights)
    out = export_yolo(model, args.out, args.name, args.version, imgsz=args.imgsz,
                      conf=args.conf, iou=args.iou, license=args.license)
    print(f"Model folder written to {out}")


if __name__ == "__main__":
    main()
