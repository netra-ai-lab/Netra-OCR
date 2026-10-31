"""Build a Khmer-Character-Cluster-aware vocabulary from a training corpus.

Streams a corpus, segments every line with ``MultiScriptSegmenter``, counts
the frequency of every Khmer-script cluster produced, and keeps clusters
until a cumulative-coverage threshold is crossed. The base 176-token
codepoint vocab (``char2idx_new.json``) is kept unchanged at ids 0-175 (so an
existing checkpoint's embedding/output-projection rows for those ids stay
meaningfully aligned for warm-start migration); new multi-codepoint clusters
are appended starting at id 176, followed by one new ``<mask>`` token used by
the semi-autoregressive decoder.

Usage:
    python -m netra_ocr_train.scripts.build_cluster_vocab \\
        --corpus training/texts/khmer_corpus.txt --coverage 0.995 \\
        --out training/netra_ocr_train/char2idx_cluster.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from netra_ocr_train.tokenization.dispatch import MultiScriptSegmenter


def scan_corpus(corpus_path: Path, segmenter: MultiScriptSegmenter,
                 base_vocab: dict) -> tuple[Counter, Counter]:
    """Count Khmer-cluster frequencies (single-codepoint pieces are dropped --
    those are already covered by the base vocab), plus any single codepoint
    that appears in the corpus but isn't in the base vocab at all (a
    corpus/vocab consistency check -- e.g. this catches punctuation like
    curly quotes that the original 176-token vocab never included)."""
    counts: Counter = Counter()
    missing_singletons: Counter = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            for piece in segmenter.segment(line):
                if len(piece) > 1:
                    counts[piece] += 1
                elif piece not in base_vocab:
                    missing_singletons[piece] += 1
    return counts, missing_singletons


def build_vocab(cluster_counts: Counter, base_vocab: dict, coverage: float,
                 missing_singletons: Counter | None = None) -> tuple[dict, int]:
    total = sum(cluster_counts.values())
    ranked = cluster_counts.most_common()

    keep: list[str] = []
    cumulative = 0
    for cluster, count in ranked:
        if total > 0 and cumulative / total >= coverage:
            break
        keep.append(cluster)
        cumulative += count

    vocab = dict(base_vocab)
    next_id = max(base_vocab.values()) + 1

    # Close any corpus/vocab gap first (e.g. curly quotes missing from the
    # original 176-token vocab) so both old and new tokenizers benefit and
    # these atoms are available as fallback targets for cluster decomposition.
    if missing_singletons:
        for ch, _ in missing_singletons.most_common():
            vocab[ch] = next_id
            next_id += 1

    for cluster in keep:
        vocab[cluster] = next_id
        next_id += 1
    vocab["<mask>"] = next_id

    return vocab, len(keep)


def report(cluster_counts: Counter, base_vocab: dict, coverages: list[float],
           missing_singletons: Counter | None = None) -> None:
    total = sum(cluster_counts.values())
    unique = len(cluster_counts)
    print(f"Unique multi-codepoint Khmer clusters seen: {unique}")
    print(f"Total cluster occurrences: {total}")
    if missing_singletons:
        print(f"Singleton codepoints missing from base vocab: {dict(missing_singletons)}")
    print()
    for cov in coverages:
        _, kept = build_vocab(cluster_counts, base_vocab, cov, missing_singletons)
        n_missing = len(missing_singletons) if missing_singletons else 0
        print(f"coverage={cov:>6.3%}  clusters kept={kept:5d}  "
              f"final vocab size={len(base_vocab) + n_missing + kept + 1}")
    print()
    print("Top 20 most frequent clusters:")
    for cluster, count in cluster_counts.most_common(20):
        print(f"  {cluster!r:12s} count={count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("training/texts/khmer_corpus.txt"))
    parser.add_argument("--base-vocab", type=Path,
                         default=Path("training/netra_ocr_train/char2idx_new.json"))
    parser.add_argument("--coverage", type=float, default=0.995)
    parser.add_argument("--out", type=Path,
                         default=Path("training/netra_ocr_train/char2idx_cluster.json"))
    parser.add_argument("--report-only", action="store_true",
                         help="Only print the coverage/vocab-size report, don't write files.")
    args = parser.parse_args()

    with open(args.base_vocab, "r", encoding="utf-8") as f:
        base_vocab = json.load(f)

    segmenter = MultiScriptSegmenter()
    cluster_counts, missing_singletons = scan_corpus(args.corpus, segmenter, base_vocab)

    report(cluster_counts, base_vocab, [0.99, 0.995, 0.999], missing_singletons)

    if args.report_only:
        return

    vocab, kept = build_vocab(cluster_counts, base_vocab, args.coverage, missing_singletons)
    idx2char = {v: k for k, v in vocab.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    idx2char_path = args.out.parent / args.out.name.replace("char2idx", "idx2char")
    with open(idx2char_path, "w", encoding="utf-8") as f:
        json.dump(idx2char, f, ensure_ascii=False, indent=2)

    print()
    print(f"Wrote {len(vocab)}-token vocab ({kept} new clusters + <mask>) to {args.out}")
    print(f"Wrote inverse vocab to {idx2char_path}")


if __name__ == "__main__":
    main()
