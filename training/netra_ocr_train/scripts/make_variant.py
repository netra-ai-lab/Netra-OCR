"""Add reduced-precision copies of a model folder's graphs.

Usage:
    python -m netra_ocr_train.scripts.make_variant bundles/lines fp16 --prefer-on cuda tensorrt
    python -m netra_ocr_train.scripts.make_variant bundles/recognition int8
    python -m netra_ocr_train.scripts.make_variant bundles/recognition int8 --remove

fp16   all float weights/activations in half precision, graph inputs and
       outputs kept float32 so the app's code doesn't change. For GPUs.
int8   dynamic quantization: weights stored as int8, activations quantized on
       the fly. Only MatMul/Gemm by default -- Transformer layers gain the most,
       while ConvInteger is usually *slower* than float Conv on CPU.

A new variant is only used once ``--prefer-on`` names devices for it -- do that
after checking accuracy on a real test set, since precision changes outputs.
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
    parser.add_argument("variant", choices=sorted(CONVERTERS))
    parser.add_argument("--prefer-on", nargs="*", default=[],
                        help="devices that should load this variant by default (e.g. cuda tensorrt)")
    parser.add_argument("--roles", nargs="*", help="only convert these graph roles")
    parser.add_argument("--remove", action="store_true", help="delete this variant instead")
    args = parser.parse_args(argv)
    if args.remove:
        remove_variant(args.folder, args.variant)
        print(f"{args.variant}: removed")
        return
    print(f"{args.variant}: {make_variant(args.folder, args.variant, args.prefer_on, args.roles)}")


if __name__ == "__main__":
    main()
