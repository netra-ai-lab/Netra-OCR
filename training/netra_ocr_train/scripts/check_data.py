"""Check a dataset before training on it: format, line count, unreadable
images, empty labels, image sizes and characters the model can't learn.

    bash training/check_data.sh my_lines/
    bash training/check_data.sh my_lines/labels.tsv user/dataset:train[:1000] khob

Every local image is opened (a progress bar follows); a Hugging Face dataset
is checked on a sample of --sample lines. Exits with 1 when a dataset can't
be used as it is, and says why.
"""

import argparse
import statistics
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..utils import HF_CACHE_DIR, use_hf_cache

DEFAULT_VOCAB = Path(__file__).resolve().parents[1] / "char2idx_cluster.json"


def _range(values: list[float], fmt: str = "{:.0f}") -> str:
    if not values:
        return "-"
    return f"{fmt.format(min(values))} – {fmt.format(max(values))} (median {fmt.format(statistics.median(values))})"


def check(spec: str, tokenizer, sample: int) -> bool:
    """Print a report for one source; False when it has a blocking problem."""
    from ..data import FORMATS, LocalLines, clean_label, load_source

    print(f"\n== {spec}")
    try:
        rows = load_source(spec)
    except Exception as e:   # DataError, or a Hub error for a bad id
        message = str(e).replace(FORMATS, "").strip()
        print(f"   ✗ {message}\n\n{FORMATS}")
        return False
    local = isinstance(rows, LocalLines)
    print(f"   format    {getattr(rows, 'kind', 'dataset')}")
    print(f"   lines     {len(rows):,}")
    ok = True

    problems = rows.problems if local else []
    if problems:
        print(f"   ⚠ skipped {len(problems)}: " + "; ".join(problems[:5]) + (" ..." if len(problems) > 5 else ""))

    indices = range(len(rows)) if local else range(min(sample, len(rows)))
    heights, widths, ratios, lengths, unreadable = [], [], [], [], []
    for i in tqdm(indices, desc="   reading", unit="line", leave=False, dynamic_ncols=True, disable=None):
        try:
            image = Image.open(rows.paths[i]) if local else rows[i]["image"]
            image.load()
        except Exception as e:
            unreadable.append(f"{rows.paths[i].name if local else f'#{i}'} ({type(e).__name__})")
            continue
        heights.append(image.height)
        widths.append(image.width)
        ratios.append(image.width / max(1, image.height))
    labels = [clean_label(t) for t in (rows.labels if local else rows.select(indices)["label"])]
    lengths = [len(t) for t in labels]

    note = "" if local else f" (first {len(indices):,})"
    print(f"   height    {_range(heights)} px{note}")
    print(f"   width     {_range(widths)} px")
    print(f"   aspect    {_range(ratios, '{:.1f}')} (width / height)")
    print(f"   text      {_range(lengths)} characters")
    if unreadable:
        ok = False
        print(f"   ✗ {len(unreadable)} image{'s' if len(unreadable) > 1 else ''} can't be read: " + "; ".join(unreadable[:5]))
    if heights and statistics.median(ratios) < 1.5:
        print("   ⚠ most images are about as tall as they are wide: the model reads one text line per image,"
              " cropped closely")
    empty = sum(1 for t in labels if not t)
    if empty:
        print(f"   ⚠ {empty} labels are empty after cleaning")

    unknown = {}
    for text in labels:
        for ch in text:
            if ch not in tokenizer.char2idx and not ch.isspace():
                unknown[ch] = unknown.get(ch, 0) + 1
    if unknown:
        top = ", ".join(f"{ch!r} U+{ord(ch):04X} ×{n}" for ch, n in sorted(unknown.items(), key=lambda kv: -kv[1])[:10])
        print(f"   ⚠ characters the model can't learn (read back as a space): {top}")
    else:
        print("   ✓ every character is in the vocabulary")

    print("   examples")
    for i in list(indices)[:5]:
        name = rows.paths[i].name if local else f"#{i}"
        print(f"     {name[:28]:<28} {labels[i][:70]}")
    print("   ✓ ready" if ok else "   ✗ fix the errors above first")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="+", help="Folders, manifests, Hugging Face ids or presets")
    parser.add_argument("--sample", type=int, default=1000, help="Lines checked from a Hugging Face dataset")
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB))
    parser.add_argument("--data-dir", default=str(HF_CACHE_DIR), help="Hugging Face cache")
    args = parser.parse_args(argv)

    use_hf_cache(args.data_dir)
    from ..cluster_tokenizer import ClusterTokenizer
    from ..data import resolve
    tokenizer = ClusterTokenizer(args.vocab)

    results = [check(source, tokenizer, args.sample) for spec in args.data for source in resolve(spec)]
    if not all(results):
        sys.exit(1)
    print("\nAll good.")


if __name__ == "__main__":
    main()
