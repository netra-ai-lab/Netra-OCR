"""Validate the cluster tokenizer against the training corpus.

Checks, over every corpus line:
    1. Round-trip correctness: decode(encode(line)) == clean_text(line).
    2. Zero <unk> occurrences (the cluster tokenizer should never need it for
       valid Khmer/Latin/digit/punctuation text -- every codepoint in the base
       176-token vocab is kept as a fallback atom).
    3. Average/median sequence-length reduction vs. the plain codepoint
       tokenizer -- the actual "is this worth it" payoff metric.

Also spot-checks a handful of hand-picked mixed Khmer+English+digit strings
(passport/ID-style lines) to confirm script-dispatch boundaries are correct
at script-transition points.

Usage:
    python -m netra_ocr_train.scripts.validate_tokenizer \\
        --corpus training/texts/khmer_corpus.txt \\
        --base-vocab training/netra_ocr_train/char2idx_new.json \\
        --cluster-vocab training/netra_ocr_train/char2idx_cluster.json
"""

import argparse
import statistics
from pathlib import Path

from netra_ocr_train.cluster_tokenizer import ClusterTokenizer
from netra_ocr_train.postprocess import clean_text

SPOT_CHECKS = [
    "លេខ Passport123 P<KHM",
    "ថ្ងៃទី ២៩ មករា ២០២៦",
    "P<KHMSOK<<SREY<<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
    "Hello World 123 សួស្តី",
    "ស្ត្រីម្នាក់មានឈ្មោះថា Sophea",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("training/texts/khmer_corpus.txt"))
    parser.add_argument("--base-vocab", type=Path,
                         default=Path("training/netra_ocr_train/char2idx_new.json"))
    parser.add_argument("--cluster-vocab", type=Path,
                         default=Path("training/netra_ocr_train/char2idx_cluster.json"))
    parser.add_argument("--limit", type=int, default=None,
                         help="Only check the first N lines (default: all).")
    args = parser.parse_args()

    # ClusterTokenizer pointed at the base 176-vocab degrades to the exact
    # per-codepoint behavior of the old plain Tokenizer, so it's a drop-in
    # reference (avoids importing the archived, relative-import-broken Tokenizer).
    base_tok = ClusterTokenizer(args.base_vocab)
    cluster_tok = ClusterTokenizer(args.cluster_vocab)

    print("=== Spot checks (mixed script) ===")
    all_spot_ok = True
    for text in SPOT_CHECKS:
        ids = cluster_tok.encode(text)
        decoded = cluster_tok.decode(ids)
        expected = clean_text(text)
        ok = decoded == expected
        all_spot_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] {text!r}")
        if not ok:
            print(f"         decoded:  {decoded!r}")
            print(f"         expected: {expected!r}")
    print()

    mismatches = 0
    unk_lines = 0
    total_lines = 0
    base_lengths = []
    cluster_lengths = []

    with open(args.corpus, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            total_lines += 1

            cluster_ids = cluster_tok.encode(line)
            decoded = cluster_tok.decode(cluster_ids)
            expected = clean_text(line)
            if decoded != expected:
                mismatches += 1
                if mismatches <= 5:
                    print(f"[MISMATCH line {i}] {line!r}")
                    print(f"    decoded:  {decoded!r}")
                    print(f"    expected: {expected!r}")

            if cluster_tok.unk_idx in cluster_ids:
                unk_lines += 1
                if unk_lines <= 5:
                    print(f"[<unk> line {i}] {line!r}")

            base_ids = base_tok.encode(line)
            base_lengths.append(len(base_ids))
            cluster_lengths.append(len(cluster_ids))

    print()
    print("=== Corpus-wide validation ===")
    print(f"Lines checked: {total_lines}")
    print(f"Round-trip mismatches: {mismatches} ({mismatches / total_lines:.4%})")
    print(f"Lines containing <unk>: {unk_lines} ({unk_lines / total_lines:.4%})")
    print()

    mean_base = statistics.mean(base_lengths)
    mean_cluster = statistics.mean(cluster_lengths)
    median_base = statistics.median(base_lengths)
    median_cluster = statistics.median(cluster_lengths)
    print(f"Mean sequence length   -- codepoint: {mean_base:.2f}  cluster: {mean_cluster:.2f}"
          f"  reduction: {(1 - mean_cluster / mean_base):.2%}")
    print(f"Median sequence length -- codepoint: {median_base:.2f}  cluster: {median_cluster:.2f}"
          f"  reduction: {(1 - median_cluster / median_base):.2%}")

    if mismatches == 0 and unk_lines == 0 and all_spot_ok:
        print()
        print("All checks passed.")


if __name__ == "__main__":
    main()
