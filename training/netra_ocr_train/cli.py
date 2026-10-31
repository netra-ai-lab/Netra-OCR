"""``netra-train``: one command for the recognizer's train / evaluate / export.

    netra-train train --init published --data my_lines/
    netra-train evaluate published training/runs/<run>/best.pth
    netra-train export --model training/runs/<run>/best.pth --out bundles/recognition

The bash scripts in training/ wrap these with sensible defaults.
"""

import sys

COMMANDS = {
    "train": "netra_ocr_train.scripts.train",
    "evaluate": "netra_ocr_train.scripts.evaluate",
    "export": "netra_ocr_train.scripts.export_recognizer",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__.strip() + f"\n\ncommands: {', '.join(COMMANDS)} (add --help to any)")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    from importlib import import_module
    sys.argv = [f"netra-train {argv[0]}"] + argv[1:]
    import_module(COMMANDS[argv[0]]).main(argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
