"""Netra Lab brand colours for charts and figures.

Reads tokens.json next to this file, so the hex values live in one place.

    from brand_colors import COLOR, apply, ours_bar, legend_dot
    apply()                                    # netra.mplstyle + Khmer font if present
    ax.bar(x, ours, **ours_bar())              # orange with white //// hatch
    ax.bar(x, base, color=BASELINES[0])
"""
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TOKENS = json.loads((_HERE / "tokens.json").read_text(encoding="utf-8"))

COLOR = {name: spec["hex"] for name, spec in _TOKENS["color"].items() if "hex" in spec}
CHART = _TOKENS["chart"]
OURS = CHART["ours"]
BASELINES = CHART["baselines"]


def apply(font_dir=None):
    """Use the Netra chart style; register Kantumruy Pro / Noto Sans Khmer from font_dir."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    if font_dir:
        for path in Path(font_dir).glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(path))
    plt.style.use(str(_HERE / "netra.mplstyle"))


def ours_bar():
    """Bar kwargs for the Netra series: exact brand orange, white hatch, no outline."""
    return {"color": OURS, "hatch": "////", "edgecolor": CHART["figureBackground"], "linewidth": 0, "zorder": 3}


def legend_dot(color, label):
    """Circular legend handle, per the chart guide (no rectangle patches)."""
    from matplotlib.lines import Line2D

    return Line2D([], [], marker="o", linestyle="None", markersize=9,
                  markerfacecolor=color, markeredgewidth=0, label=label)
