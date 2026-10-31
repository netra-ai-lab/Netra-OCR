"""Add reduced-precision copies of a model folder's graphs, or shrink its files.

Usage:
    python -m netra_ocr_train.scripts.make_variant bundles/lines fp16 --prefer-on cuda tensorrt
    python -m netra_ocr_train.scripts.make_variant bundles/recognition int8
    python -m netra_ocr_train.scripts.make_variant bundles/recognition int8 --remove
    python -m netra_ocr_train.scripts.make_variant bundles/recognition store-fp16

fp16   all float weights/activations in half precision, graph inputs and
       outputs kept float32 so the app's code doesn't change. For GPUs.
int8   dynamic quantization: weights stored as int8, activations quantized on
       the fly. Only MatMul/Gemm by default -- Transformer layers gain the most,
       while ConvInteger is usually *slower* than float Conv on CPU.

store-fp16  not a variant: rewrites the folder's default graphs in place so
       their weights are *stored* as fp16 (half the file size) and cast back
       to float32 when the model loads. ONNX Runtime folds those casts, so the
       maths and speed stay float32; only the weights' rounding changes.

A new variant is only used once ``--prefer-on`` names devices for it -- do that
after checking accuracy on a real test set, since precision changes outputs.
The same goes for store-fp16: compare the text before uploading.
"""

import argparse
from pathlib import Path

from ..model_folder import read_config, write_config

DEFAULT_INT8_OPS = ("MatMul", "Gemm")


def _convert_fp16(src: Path, dst: Path):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    # When two nodes share a float32 constant (e.g. two Resize ops with the
    # same scales), the converter inserts one identical Cast per consumer, all
    # writing the same tensor -- an invalid graph. Keep the first of each.
    produced, kept = set(), []
    for node in model.graph.node:
        if node.output and all(out in produced for out in node.output):
            continue
        produced.update(node.output)
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    onnx.save(model, str(dst))


def _convert_int8(src: Path, dst: Path, op_types=DEFAULT_INT8_OPS):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=list(op_types))


CONVERTERS = {"fp16": _convert_fp16, "int8": _convert_int8}
MIN_STORED_FP16 = 256   # smaller weights (biases, norms) stay float32: no size gain


def _store_fp16(path: Path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    model = onnx.load(str(path))
    graph = model.graph
    initializers, casts = [], []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= MIN_STORED_FP16:
            half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + "__fp16")
            initializers.append(half)
            casts.append(helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT))
        else:
            initializers.append(init)
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    for node in reversed(casts):
        graph.node.insert(0, node)
    onnx.save(model, str(path))


def store_fp16(folder: str | Path, roles: list[str] | None = None) -> dict:
    """Rewrite the folder's default graphs (all roles, or only ``roles``) with fp16-stored weights."""
    folder = Path(folder)
    config = read_config(folder)
    sizes = {}
    for role, file in config["graphs"].items():
        if roles and role not in roles:
            continue
        path = folder / file
        before = path.stat().st_size
        _store_fp16(path)
        sizes[file] = f"{before / 1e6:.1f} -> {path.stat().st_size / 1e6:.1f} MB"
    write_config(folder, config)
    return sizes


def make_variant(folder: str | Path, variant: str, prefer_on: list[str] | None = None,
                 roles: list[str] | None = None) -> dict:
    """Convert the folder's default graphs (all roles, or only ``roles``) and register the variant."""
    if variant not in CONVERTERS:
        raise ValueError(f"Unknown variant '{variant}'. Available: {sorted(CONVERTERS)}")
    folder = Path(folder)
    config = read_config(folder)
    graphs = {}
    for role, file in config["graphs"].items():
        if roles and role not in roles:
            continue
        path = folder / file
        out = path.with_name(f"{path.stem}.{variant}{path.suffix}")
        CONVERTERS[variant](path, out)
        graphs[role] = out.name
    config.setdefault("variants", {})[variant] = {"graphs": graphs, "devices": list(prefer_on or [])}
    write_config(folder, config)
    return config["variants"][variant]


def remove_variant(folder: str | Path, variant: str):
    """Delete a variant's graph files and its config entry."""
    folder = Path(folder)
    config = read_config(folder)
    entry = config.get("variants", {}).pop(variant, None)
    if entry is None:
        raise ValueError(f"Model folder has no variant '{variant}'")
    default_files = set(config["graphs"].values())
    for file in entry.get("graphs", {}).values():
        if file not in default_files:
            (folder / file).unlink(missing_ok=True)
    write_config(folder, config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("variant", choices=sorted(CONVERTERS) + ["store-fp16"])
    parser.add_argument("--prefer-on", nargs="*", default=[],
                        help="devices that should load this variant by default (e.g. cuda tensorrt)")
    parser.add_argument("--roles", nargs="*", help="only convert these graph roles")
    parser.add_argument("--remove", action="store_true", help="delete this variant instead")
    args = parser.parse_args(argv)
    if args.variant == "store-fp16":
        print(f"store-fp16: {store_fp16(args.folder, args.roles)}")
        return
    if args.remove:
        remove_variant(args.folder, args.variant)
        print(f"{args.variant}: removed")
        return
    print(f"{args.variant}: {make_variant(args.folder, args.variant, args.prefer_on, args.roles)}")


if __name__ == "__main__":
    main()
