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


def blockwise_steps(prefix, proposals, greedy, out, token_font="Noto Sans Khmer"):
    """One step of blockwise parallel decoding (Stern et al. 2018): propose, verify, accept.

    prefix:    tokens decoded so far
    proposals: the tokens the proposal head guesses in one pass (p1 is the base decoder itself, so it always matches)
    greedy:    the base decoder's own greedy choice at each proposed position, from one verification pass
    token_font: font for the tokens; Khmer clusters need one with Khmer glyphs
    """
    block = len(proposals)
    n_ok = next((i for i, (p, g) in enumerate(zip(proposals, greedy)) if p != g), block)
    w, h, gx = 1.1, 1.1, 0.2
    rows = {"propose": 6.0, "verify": 3.7, "accept": 1.4}
    x0 = 5.2
    xs = [x0 + i * (w + gx) for i in range(len(prefix) + block)]
    k = len(prefix)
    right = xs[-1] + w + 0.7
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_aspect("equal")
    ax.axis("off")

    def box(i, y, tok, fc, ec, ls="-", tc=INK, strike=False):
        ax.add_patch(FancyBboxPatch((xs[i], y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=1.2, ls=ls))
        ax.text(xs[i] + w / 2, y + h / 2, tok, ha="center", va="center", fontsize=16, color=tc, family=token_font)
        if strike:
            ax.plot([xs[i] + 0.15, xs[i] + w - 0.15], [y + 0.15, y + h - 0.15], color=COLOR["muted-2"], lw=1.3)

    def prefix_row(y):
        for i, tok in enumerate(prefix):
            box(i, y, tok, COLOR["card"], COLOR["border"])

    def label(y, num, title, note):
        ax.text(0.2, y + h / 2 + 0.24, f"{num}  {title}", ha="left", va="center", fontsize=13, color=INK,
                weight="bold")
        ax.text(0.2, y + h / 2 - 0.42, note, ha="left", va="center", fontsize=9.5, color=COLOR["muted"])

    def note(y, text):
        ax.text(right, y + h / 2, text, va="center", fontsize=10.5, color=INK, linespacing=1.35)

    # 1. propose: the head guesses the next tokens all at once
    y = rows["propose"]
    label(y, 1, "Propose", "proposal head, one pass")
    prefix_row(y)
    for j, tok in enumerate(proposals):
        box(k + j, y, tok, COLOR["tint-orange"], OURS, ls=(0, (3, 2)))
        ax.text(xs[k + j] + w / 2, y + h + 0.14, f"p{j + 1}", ha="center", va="bottom", fontsize=9.5,
                color=COLOR["muted"])
    note(y, f"Guess the next {block} tokens\nat once")
    # 2. verify: the base decoder reads prefix + guesses once; compare its choice under each guess
    y = rows["verify"]
    label(y, 2, "Verify", "base decoder, one pass")
    for j in range(n_ok + 1 if n_ok < block else block):
        box(k + j, y, greedy[j], CHART["figureBackground"], INK)
        same = j < n_ok
        ax.text(xs[k + j] + w / 2, (rows["propose"] + y + h) / 2, "=" if same else "≠", ha="center", va="center",
                fontsize=17, color=OURS if same else COLOR["accent-maroon"], weight="bold")
    ax.text(xs[k] - gx - 0.15, y + h / 2, "decoder's\nown choice", ha="right", va="center", fontsize=9.5,
            color=COLOR["muted"], linespacing=1.2)
    note(y, f"Same as p1–p{n_ok}, differs at p{n_ok + 1}:\nstop checking there" if n_ok < block
         else f"Same as p1–p{block}")
    # 3. accept: the matching guesses, plus the decoder's own token where they first differ
    y = rows["accept"]
    label(y, 3, "Accept", "keep matches + the fix")
    prefix_row(y)
    for j in range(block):
        if j < n_ok:
            box(k + j, y, proposals[j], OURS, OURS, tc="white")
        elif j == n_ok:
            box(k + j, y, greedy[j], COLOR["accent-maroon"], COLOR["accent-maroon"], tc="white")
        else:
            box(k + j, y, proposals[j], COLOR["background"], COLOR["muted-2"], ls=(0, (3, 2)),
                tc=COLOR["muted-2"], strike=True)
    added = min(n_ok + 1, block)
    note(y, f"+{added} tokens from 2 passes\n(greedy decoding: {added} passes)")
    # legend, laid out from measured text widths
    ly, lx = 0.0, x0
    xmax = right + 5.2
    ax.set_xlim(0, xmax)  # fix the scale before measuring text
    ax.set_ylim(ly - 0.3, rows["propose"] + h + 0.7)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for fc, ec, ls, text in ((COLOR["card"], COLOR["border"], "-", "decoded so far"),
                             (COLOR["tint-orange"], OURS, (0, (3, 2)), "guess"),
                             (OURS, OURS, "-", "accepted guess"),
                             (COLOR["accent-maroon"], COLOR["accent-maroon"], "-", "decoder's own token"),
                             (COLOR["background"], COLOR["muted-2"], (0, (3, 2)), "discarded")):
        ax.add_patch(FancyBboxPatch((lx, ly), 0.45, 0.45, boxstyle="round,pad=0,rounding_size=0.07",
                                    fc=fc, ec=ec, lw=1.0, ls=ls))
        t = ax.text(lx + 0.65, ly + 0.22, text, va="center", fontsize=10, color=INK)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        lx = inv.transform((bb.x1, bb.y1))[0] + 0.8
    ax.set_xlim(0, max(xmax, lx))
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
    # ព្រះរាជាណាចក្រកម្ពុជា as the cluster tokenizer splits it. The head drops the subscript of ក្រ at p3, so the
    # base decoder keeps ណា ច and adds its own ក្រ
    blockwise_steps(["ព្រះ", "រា", "ជា"], ["ណា", "ច", "ក", "ម្ពុ"], ["ណា", "ច", "ក្រ", "ក"],
                    out_dir / "blockwise_decoding.png")
    print("wrote results.png, blockwise_decoding.png in", out_dir)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
