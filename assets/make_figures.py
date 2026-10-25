"""README figures in the Netra style (from the Netra README Kit's templates/figures/make_figures.py).

    python assets/make_figures.py          # writes results.png and blockwise_decoding.png next to this file

  results.png             CER of each OCR engine on the three test sets (OCR_COMPARISON_RESULTS.md)
  blockwise_decoding.png  one step of blockwise parallel decoding, as implemented in recognition/predictor.py

The numbers are copied from OCR_COMPARISON_RESULTS.md (git-ignored); docs/RESULTS.md publishes them. Update both together.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
# <repo>/assets/brand when copied into a repo; the kit's assets/brand when run from the kit
for brand in (HERE / "brand", HERE.parents[1] / "assets" / "brand"):
    if (brand / "brand_colors.py").exists():
        sys.path.insert(0, str(brand))
        break
from brand_colors import BASELINES, CHART, COLOR, NEUTRALS, OURS, apply, legend_dot, ours_bar  # noqa: E402

INK = COLOR["foreground"]


def results_bars(panels, labels, out, ylabel="Error rate (%), lower is better", ylim=None, legend_top=False):
    """Grouped bars, one panel per model family.

    panels: [(title, [(series_label, [value per label], color or None for ours), ...]), ...]
    labels: x tick labels, one per value (e.g. "DDD\\n(Khmer, CER)"). NaN prints "n/a".
    legend_top: one legend row above the plot instead of one per panel (use it with many series or tall bars).
    """
    n_series = max(len(series) for _, series in panels)
    bar_w, pitch = 0.3, 0.42
    group = max(1.6, n_series * pitch + 0.5)  # room for every bar plus a gap between groups
    width = max(6.5, 0.75 * n_series * len(labels) + 1.5)
    fig, axes = plt.subplots(1, len(panels), figsize=(width * len(panels), 4.4), sharey=True, squeeze=False)
    x = np.arange(len(labels)) * group
    for ax, (title, series) in zip(axes[0], panels):
        mid = (len(series) - 1) / 2
        for j, (_, vals, color) in enumerate(series):
            pos = x + (j - mid) * pitch
            ax.bar(pos, vals, bar_w, **(ours_bar() if color is None else {"color": color, "zorder": 3}))
            for p, v in zip(pos, vals):
                ax.text(p, (0 if np.isnan(v) else v) + 0.5, "n/a" if np.isnan(v) else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(x, labels, fontsize=10)
        if title:
            ax.set_title(title, loc="left", fontsize=12)
        handles = [legend_dot(OURS if c is None else c, s) for s, _, c in series]
        if legend_top:
            fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10, bbox_to_anchor=(0.5, 0.9))
        else:
            ax.legend(handles=handles, loc="upper right", fontsize=9)
    axes[0][0].set_ylabel(ylabel)
    if ylim:
        axes[0][0].set_ylim(*ylim)  # leave headroom for the legend
    fig.savefig(out)
    plt.close(fig)


def before_after_bars(groups, out, ylabel, before_label, after_label, fmt="{:.0f}", ratio="{:.1f}× faster"):
    """Two bars per group: the reference (muted) and ours (hatched orange), ratio printed in bold on ours.

    groups: [(tick label, before value, after value), ...]; the tick label can carry a second line
    such as "325M → 73M". ``ratio`` is formatted with before / after.
    """
    fig, ax = plt.subplots(figsize=(max(5.5, 3.5 * len(groups)), 4.8))
    w, top = 0.32, max(max(b, a) for _, b, a in groups)
    for g, (_, b, a) in enumerate(groups):
        ax.bar(g - w / 2 - 0.02, b, w, color=BASELINES[0], zorder=3)
        ax.bar(g + w / 2 + 0.02, a, w, **ours_bar())
        ax.text(g - w / 2 - 0.02, b + 0.015 * top, fmt.format(b), ha="center", va="bottom", fontsize=11, color=INK)
        ax.text(g + w / 2 + 0.02, a + 0.015 * top, f"{fmt.format(a)}\n{ratio.format(b / a)}", ha="center",
                va="bottom", fontsize=11, color=INK, weight="bold", linespacing=1.2)
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=11)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, top * 1.2)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[legend_dot(BASELINES[0], before_label), legend_dot(OURS, after_label)], loc="lower center",
              bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def block_diagram(kinds, before, after, out, before_title, after_title, arrow_label, arrow_note="",
                  group_labels=(), cells=4):
    """Before -> after schematic of a model as columns of blocks (paper-figure style, e.g. DiariZen Fig. 1).

    kinds:  {kind: (solid colour token, tint colour token, legend label)}, legend in this order
    before: [(kind, kept), ...] or "..." for a gap; ``kept`` of ``cells`` blocks are solid, the rest dashed
    after:  the same columns for the smaller model; only the kept blocks are drawn, packed together
    group_labels: text under each group that follows a "..." in ``before`` (e.g. "layer 12")
    """
    cw, ch, gap = 3.0, 2.6, 0.75
    col_h = cells * ch + (cells - 1) * gap
    y0 = 10.5
    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.set_aspect("equal")
    ax.axis("off")

    def cell(x, y, kind, on):
        solid, tint, _ = kinds[kind]
        if on:
            ax.add_patch(Rectangle((x, y), cw, ch, fc=COLOR[solid], ec=INK, lw=0.6))
        else:
            ax.add_patch(Rectangle((x, y), cw, ch, fc=COLOR[tint], ec=COLOR[solid], lw=1.0, ls=(0, (2.2, 1.6))))

    def dots(x):
        for dy in (-1.2, 0, 1.2):
            ax.plot(x, y0 + col_h / 2 + dy, "o", ms=2.2, color=INK)

    def model(x0, columns, packed, title):
        x, starts, rng = x0 + 2.2, [], np.random.default_rng(3)
        for col in columns:
            if col == "...":
                dots(x + 0.6)
                x += 2.4
                starts.append(x)
                continue
            kind, n = col
            if packed and n == 0:
                continue
            if packed:  # survivors only, centred
                top = y0 + col_h / 2 + (n * ch + (n - 1) * gap) / 2
                for j in range(n):
                    cell(x, top - (j + 1) * ch - j * gap, kind, True)
            else:
                on = set(rng.choice(cells, size=n, replace=False).tolist())
                for j in range(cells):
                    cell(x, y0 + j * (ch + gap), kind, j in on)
            x += cw + 1.0
        x += 1.6
        ax.add_patch(FancyBboxPatch((x0, y0 - 4.0), x - x0, col_h + 6.2, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc="none", ec=INK, lw=1.3, ls=(0, (5, 3)) if packed else "-"))
        ax.text((x0 + x) / 2, y0 - 5.4, title, ha="center", va="top", fontsize=12, color=INK)
        if not packed:
            for sx, label in zip(starts, group_labels):
                ax.text(sx + cw + 0.5, y0 - 0.7, label, ha="center", va="top", fontsize=8.5, color=COLOR["muted"])
        return x

    left = model(1, before, False, before_title)
    mid = y0 + col_h / 2
    ax.add_patch(FancyArrowPatch((left + 1.5, mid), (left + 21.5, mid), arrowstyle="-|>", mutation_scale=16,
                                 color=INK, lw=1.4))
    ax.text(left + 11.5, mid + 1.2, arrow_label, ha="center", va="bottom", fontsize=12, color=INK)
    ax.text(left + 11.5, mid - 1.2, arrow_note, ha="center", va="top", fontsize=9.5, color=COLOR["muted"])
    right = model(left + 23, after, True, after_title)

    lx = right + 4.5
    for k, kind in enumerate(kinds):
        y = y0 + col_h - 1.2 - k * 4.4
        cell(lx, y, kind, True)
        ax.text(lx + cw + 1.2, y + ch / 2, kinds[kind][2], va="center", fontsize=11, color=INK)
    y = y0 + col_h - 1.2 - len(kinds) * 4.4
    ax.add_patch(Rectangle((lx, y), cw, ch, fc=COLOR["card"], ec=COLOR["muted-2"], lw=1.0, ls=(0, (2.2, 1.6))))
    ax.text(lx + cw + 1.2, y + ch / 2, "removed", va="center", fontsize=11, color=INK)
    ax.set_xlim(0, lx + 22)
    ax.set_ylim(min(y, y0 - 7.5) - 0.5, y0 + col_h + 2.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def blockwise_steps(prefix, proposals, greedy, out, block_size=4):
    """One step of blockwise parallel decoding (Stern et al. 2018), as three rows: predict, verify, accept.

    prefix:    tokens decoded so far
    proposals: the block_size tokens proposed in one pass (p1 is the base decoder itself, so it always matches)
    greedy:    the base decoder's own greedy choice at each proposed position, from one verification pass
    """
    n_ok = next((i for i, (p, g) in enumerate(zip(proposals, greedy)) if p != g), block_size)
    w, h, gx = 1.0, 1.0, 0.18
    rows = {"predict": 6.2, "verify": 3.55, "accept": 0.9}
    x0 = 5.4
    xs = [x0 + i * (w + gx) for i in range(len(prefix) + block_size)]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")

    def box(i, y, tok, fc, ec, ls="-", tc=INK, strike=False, weight="normal"):
        ax.add_patch(FancyBboxPatch((xs[i], y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.1, ls=ls))
        ax.text(xs[i] + w / 2, y + h / 2, "␣" if tok == " " else tok, ha="center", va="center", fontsize=13,
                color=tc, weight=weight)
        if strike:
            ax.plot([xs[i] + 0.15, xs[i] + w - 0.15], [y + 0.15, y + h - 0.15], color=COLOR["muted-2"], lw=1.2)

    def prefix_row(y):
        for i, tok in enumerate(prefix):
            box(i, y, tok, COLOR["card"], COLOR["border"])

    def label(y, num, title, note):
        ax.text(0.2, y + h / 2 + 0.22, f"{num}  {title}", ha="left", va="center", fontsize=12.5, color=INK,
                weight="bold")
        ax.text(0.2, y + h / 2 - 0.42, note, ha="left", va="center", fontsize=9, color=COLOR["muted"])

    k = len(prefix)
    right = xs[-1] + w + 0.6
    # 1. predict: the proposal head guesses block_size tokens at once
    y = rows["predict"]
    label(y, 1, "Predict", "one decoder pass")
    prefix_row(y)
    for j, tok in enumerate(proposals):
        box(k + j, y, tok, COLOR["tint-orange"], OURS, ls=(0, (3, 2)))
        ax.text(xs[k + j] + w / 2, y + h + 0.12, f"p{j + 1}", ha="center", va="bottom", fontsize=9,
                color=COLOR["muted"])
    ax.text(right, y + h / 2, f"{block_size} tokens proposed", va="center", fontsize=10, color=INK)
    # 2. verify: the base decoder reads prefix + proposals once and gives its greedy token at every position
    y = rows["verify"]
    label(y, 2, "Verify", "one decoder pass, in parallel")
    prefix_row(y)
    for j, tok in enumerate(greedy):
        ok = j < n_ok
        checked = j <= n_ok
        box(k + j, y, tok, "white" if checked else COLOR["background"], INK if checked else COLOR["border"],
            tc=INK if checked else COLOR["muted-2"])
        if checked:
            ax.text(xs[k + j] + w / 2, y + h + 0.12, "✓" if ok else "✗", ha="center", va="bottom", fontsize=12,
                    color=OURS if ok else COLOR["accent-maroon"], weight="bold")
    ax.text(right, y + h / 2, f"greedy choice matches p1–p{n_ok}", va="center", fontsize=10, color=INK)
    # 3. accept: the matching prefix, plus the base decoder's own token where they first disagree
    y = rows["accept"]
    label(y, 3, "Accept", "longest matching prefix")
    prefix_row(y)
    for j in range(block_size):
        if j < n_ok:
            box(k + j, y, proposals[j], OURS, OURS, tc="white", weight="bold")
        elif j == n_ok:
            box(k + j, y, greedy[j], COLOR["accent-maroon"], COLOR["accent-maroon"], tc="white", weight="bold")
        else:
            box(k + j, y, proposals[j], COLOR["background"], COLOR["muted-2"], ls=(0, (3, 2)),
                tc=COLOR["muted-2"], strike=True)
    ax.text(right, y + h / 2, f"+{n_ok + 1} tokens from 2 passes\n(greedy needs {n_ok + 1} passes)",
            va="center", fontsize=10, color=INK, linespacing=1.3)
    # arrows between rows, from the proposals down
    mid = (xs[k] + xs[k + block_size - 1] + w) / 2
    for a, b in (("predict", "verify"), ("verify", "accept")):
        ax.add_patch(FancyArrowPatch((mid, rows[a] - 0.08), (mid, rows[b] + h + 0.62), arrowstyle="-|>",
                                     mutation_scale=12, color=COLOR["muted-2"], lw=1.0))
    # legend
    ly, lx = -0.75, x0
    for fc, ec, ls, text in ((COLOR["card"], COLOR["border"], "-", "decoded so far"),
                             (OURS, OURS, "-", "accepted proposal"),
                             (COLOR["accent-maroon"], COLOR["accent-maroon"], "-", "base decoder's own token"),
                             (COLOR["background"], COLOR["muted-2"], (0, (3, 2)), "discarded")):
        ax.add_patch(FancyBboxPatch((lx, ly), 0.42, 0.42, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=fc, ec=ec, lw=1.0, ls=ls))
        ax.text(lx + 0.6, ly + 0.21, text, va="center", fontsize=9.5, color=INK)
        lx += 0.6 + 0.16 * len(text) + 0.8
    ax.set_xlim(0, max(right + 4.2, lx))
    ax.set_ylim(ly - 0.3, rows["predict"] + h + 0.7)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main(out_dir: Path) -> None:
    apply()
    plt.rcParams["hatch.color"] = CHART["figureBackground"]
    # OCR_COMPARISON_RESULTS.md, CER x 100. Ours: AR beam=3. The two VLMs are our LoRA fine-tunes (Qwen-ours-FT,
    # DeepSeek-ours-FT). Printed Word is the first 500 of 5,000 lines for every engine.
    labels = ["KHOB", "Legal Documents", "Printed Words"]
    results_bars([("", [("Tesseract 5", [5.12, 41.78, 20.14], NEUTRALS[0]),
                        ("Surya OCR", [14.52, 51.16, 28.45], NEUTRALS[1]),
                        ("Qwen2.5-VL-3B (LoRA)", [89.21, 84.63, 98.62], NEUTRALS[2]),
                        ("DeepSeek-OCR (LoRA)", [33.23, 44.22, 69.79], NEUTRALS[3]),
                        ("Netra-OCR", [0.96, 5.31, 2.95], None)])],
                 labels, out_dir / "results.png", ylabel="Character error rate (%), lower is better",
                 ylim=(0, 105), legend_top=True)
    # an English line, so the tokens are single characters: "Phn" decoded, the head proposes "omen",
    # the base decoder's greedy choice is "om P" -> accept "om" plus its own " "
    blockwise_steps(list("Phn"), list("omen"), list("om P"), out_dir / "blockwise_decoding.png")
    print("wrote results.png, blockwise_decoding.png in", out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
