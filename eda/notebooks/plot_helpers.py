"""Shared plotting helpers for the dead-ball EDA notebooks (pandas + matplotlib).

Notebooks only; the scripts stay stdlib + numpy. One look across notebooks:

    TYPE_COLOURS / type_colour   one fixed colour per set-piece Type, everywhere
    REAL, NULL_FILL, NULL_LINE   observed values vs the circular-shift null
    null_band                    the null drawn the same way everywhere (p5-p95 band, dashed mean)
    fmt_p                        p-values, "p <= 0.001" at the resolution floor
    bars_with_n                  bars annotated with n; n < MIN_N greyed and labelled
    load_summary / get / pick    read the long-format summary CSVs

Possession notebook (eda_possession), added without changing anything above:

    TEAM_COLOURS / team_colour   one fixed colour per team: H (Barcelona on the match) blue,
                                 A (Athletic) orange, the validated first two TYPE slots
    DATASETS / dataset_style     match, Sample Game 3 clean and degraded: neutral inks + markers
    threshold                    a verdict threshold, always dashed and labelled
    log_seconds                  log axis for durations, said in the axis label
    annotated_heatmap            heatmap with the number written in every cell
    poss_folder / load_poss      the poss_* summary CSVs of one dataset
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_N = 5

# Categorical slots in fixed order (validated: CVD and normal-vision separation pass; contrast
# below 3:1 for aqua, yellow and pink, so every chart also labels directly or has a table).
TYPE_ORDER = ("Corner Kick", "Free Kick", "Goal Kick", "Throw-in", "Kick Off", "Penalty")
TYPE_COLOURS = {"Corner Kick": "#2a78d6", "Free Kick": "#eb6834", "Goal Kick": "#1baf7a",
                "Throw-in": "#eda100", "Kick Off": "#e87ba4", "Penalty": "#6250d6"}
OTHER_TYPE = "#9a9a94"       # a Type outside TYPE_ORDER
NO_TYPE = "#d4d3cd"          # "none" / "(no Type)"
REAL = "#2b2b29"             # observed values, and SET PIECES overall
NULL_FILL = "#e3e2dc"        # null p5-p95 band
NULL_LINE = "#8c8b85"        # null mean, dashed
SMALL_N = "#cfcec8"          # bars and points with n < MIN_N
INK_2 = "#6b6a64"            # secondary neutral (second series that is not a Type)
INK_3 = "#b5b4ad"            # tertiary neutral
GREYS = ("#2b2b29", "#6b6a64", "#a3a29b", "#cfcec8", "#e9e8e2")   # ordinal, dark -> light


def style():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#ececea",
                         "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.frameon": False, "legend.fontsize": 9})


def canonical_type(name):
    """Map a group or Type label ('type:Corner Kick', 'Kick Off (Start)', ...) to TYPE_ORDER,
    or return it unchanged."""
    t = str(name).split("type:", 1)[-1].strip()
    low = t.lower()
    for key in TYPE_ORDER:
        if low.startswith(key.lower()) or low.replace("-", " ").startswith(
                key.lower().replace("-", " ")):
            return key
    return t


def type_colour(name):
    """Colour for a group or Type: SET PIECES -> REAL, none -> NO_TYPE, unknown -> OTHER_TYPE."""
    s = str(name)
    if s.upper().endswith("SET PIECES"):
        return REAL
    if s in ("none", "(no Type)"):
        return NO_TYPE
    return TYPE_COLOURS.get(canonical_type(s), OTHER_TYPE)


def group_label(name):
    s = str(name)
    return "all set pieces" if s.upper().endswith("SET PIECES") else s.split("type:", 1)[-1]


def sort_groups(names):
    """SET PIECES first, then Types in TYPE_ORDER, then the rest alphabetically."""
    def key(n):
        if str(n).upper().endswith("SET PIECES"):
            return (0, 0, "")
        c = canonical_type(n)
        return (1, TYPE_ORDER.index(c), "") if c in TYPE_ORDER else (2, 0, str(n))
    return sorted(names, key=key)


def null_band(ax, x, p5, p95, mean, label="shifted tags (null): p5-p95, mean", step=False):
    """Null band, the same everywhere: light grey p5-p95 fill, dashed grey mean."""
    kw = {"step": "post"} if step else {}
    ax.fill_between(x, p5, p95, color=NULL_FILL, lw=0, label=label, **kw)
    if step:
        ax.step(x, mean, where="post", color=NULL_LINE, lw=1.2, ls="--")
    else:
        ax.plot(x, mean, color=NULL_LINE, lw=1.2, ls="--")


def null_interval(ax, y, p5, p95, mean, height=0.5):
    """Horizontal version for one category at y: band from p5 to p95, dashed tick at the mean."""
    ax.barh(y, p95 - p5, left=p5, height=height, color=NULL_FILL, lw=0)
    ax.plot([mean, mean], [y - height / 2, y + height / 2], color=NULL_LINE, lw=1.4, ls="--")


def fmt_p(p, k=None):
    """'p = 0.034', or 'p <= 0.001' at the resolution floor 1 / (K + 1)."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "-"
    if k:
        floor = 1.0 / (k + 1)
        if p <= floor + 1e-12:
            return "p ≤ {:g}".format(math.ceil(floor * 1000) / 1000.0)
    return "p = {:.3f}".format(p)


def bars_with_n(ax, labels, values, ns, colours=REAL, horizontal=False, fmt="{:.2f}",
                min_n=MIN_N, value_labels=True):
    """Bars with 'n = k' annotations. Bins with n < min_n are drawn as an empty grey stub
    labelled 'n < 5' instead of a value."""
    pos = np.arange(len(labels))
    colours = [colours] * len(labels) if isinstance(colours, str) else list(colours)
    vals = np.array([v if (n >= min_n and np.isfinite(v)) else 0.0
                     for v, n in zip(values, ns)], dtype=float)
    cols = [c if n >= min_n else SMALL_N for c, n in zip(colours, ns)]
    draw = ax.barh if horizontal else ax.bar
    draw(pos, vals, color=cols, edgecolor="white", linewidth=1.5,
         **({"height": 0.65} if horizontal else {"width": 0.65}))
    top = vals.max() if len(vals) and vals.max() > 0 else 1.0   # all bins small: plain 0-1 axis
    sep = "  " if horizontal else "\n"
    for i, (v, n) in enumerate(zip(vals, ns)):
        if n < min_n:
            text = "n < {}".format(min_n)
        elif value_labels:
            text = fmt.format(v) + sep + "n = {:,}".format(int(n))
        else:
            text = "n = {:,}".format(int(n))
        if horizontal:
            ax.text(v + top * 0.01, i, text, va="center", fontsize=8, color=INK_2)
        else:
            ax.text(i, v + top * 0.01, text, ha="center", va="bottom", fontsize=8, color=INK_2)
    if horizontal:
        ax.set_yticks(pos, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, top * 1.35)
    else:
        ax.set_xticks(pos, labels)
        ax.set_ylim(0, top * 1.3)
    return pos


def type_legend(ax, names, loc="best", extra=(), below=None, ncol=1):
    """Legend patches for groups/Types in sort order, plus (label, colour) extras. below=y puts
    it centred under the axes at that axes-fraction height, in ncol columns."""
    from matplotlib.patches import Patch
    handles = [Patch(color=type_colour(n), label=group_label(n)) for n in sort_groups(names)]
    handles += [Patch(color=c, label=l) for l, c in extra]
    if below is not None:
        return ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, below),
                         ncol=ncol)
    return ax.legend(handles=handles, loc=loc, ncol=ncol)


# ---------------------------------------------------------------- reading

def load_summary(eda_out, folder, csv_name):
    """Long-format summary CSV as a DataFrame (value numeric where possible, text kept in
    'text'), or None with a note if it is missing."""
    path = Path(eda_out) / folder / csv_name
    if not path.is_file():
        print("not found:", path)
        return None
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    df["text"] = df["value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def pick(df, analysis=None, scope=None, bin_=None, metric=None):
    sel = df
    for col, want in (("analysis", analysis), ("scope", scope), ("bin", bin_),
                      ("metric", metric)):
        if want is not None:
            sel = sel[sel[col] == want]
    return sel


def get(df, analysis, scope, bin_, metric, text=False):
    sel = pick(df, analysis, scope, bin_, metric)
    if not len(sel):
        return None if text else np.nan
    return sel["text"].iloc[0] if text else float(sel["value"].iloc[0])


def null_k(df):
    """K (number of shifts) from the params rows, or None."""
    k = get(df, "params", "all", "", "null_n")
    return int(k) if np.isfinite(k) and k > 0 else None


# ---------------------------------------------------------------- possession notebook

TEAM_COLOURS = {"H": "#2a78d6", "A": "#eb6834"}     # = TYPE_COLOURS slots 1 and 2 (CVD-checked)
TEAM_LABELS = {"H": "H (home; Barcelona on the match)", "A": "A (away; Athletic on the match)"}
DATASETS = ("match", "sg3", "sg3_deg")
DATASET_LABELS = {"match": "match (pseudo-labels)", "sg3": "Sample Game 3, clean",
                  "sg3_deg": "Sample Game 3, degraded"}
DATASET_STYLE = {"match": (REAL, "o"), "sg3": (INK_2, "s"), "sg3_deg": (GREYS[2], "^")}
THRESHOLD = "#1baf7a"        # verdict thresholds: dashed, labelled on the line


def team_colour(code):
    return TEAM_COLOURS[str(code)]


def dataset_style(name):
    """(colour, marker) for a dataset; lines are solid, so dashes stay reserved for thresholds
    and the null."""
    return DATASET_STYLE[name]


def threshold(ax, value, label, axis="y"):
    """A verdict threshold: dashed line with its label written at the end of the line."""
    if axis == "y":
        ax.axhline(value, color=THRESHOLD, ls="--", lw=1.2)
        ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8,
                    color=THRESHOLD)
    else:
        ax.axvline(value, color=THRESHOLD, ls="--", lw=1.2)
        ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                    textcoords="offset points", ha="left", va="top", fontsize=8,
                    color=THRESHOLD, rotation=90)


def log_seconds(ax, what, axis="x"):
    """Log axis for a duration that spans orders of magnitude; the label says so."""
    if axis == "x":
        ax.set_xscale("log")
        ax.set_xlabel("{} (s, log scale)".format(what))
    else:
        ax.set_yscale("log")
        ax.set_ylabel("{} (s, log scale)".format(what))


def annotated_heatmap(ax, data, row_labels, col_labels, fmt="{:.2f}", vmin=0.5, vmax=1.0,
                      cmap="cividis", nan_text="-", text=None):
    """Heatmap with every number written in its cell (dark text on light cells, light on dark).
    cividis is perceptually uniform and readable with colour-vision deficiencies. `text`
    (same shape) writes other numbers than the ones coloured, e.g. signed errors coloured by
    their size."""
    data = np.asarray(data, dtype=float)
    text = data if text is None else np.asarray(text, dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    cm = plt.get_cmap(cmap)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isfinite(v):
                ax.text(j, i, nan_text, ha="center", va="center", fontsize=8, color=INK_3)
                continue
            r, g, b, _ = cm((min(max(v, vmin), vmax) - vmin) / float(vmax - vmin))
            dark = 0.299 * r + 0.587 * g + 0.114 * b > 0.55
            ax.text(j, i, fmt.format(text[i, j]), ha="center", va="center", fontsize=8,
                    color=REAL if dark else "white")
    return im


def poss_folder(script, dataset):
    """Output folder of a poss_* script for a dataset: poss_<x>, poss_<x>_sg3, poss_<x>_sg3_deg
    (as in eda/README.md)."""
    return "poss_{}{}".format(script, "" if dataset == "match" else "_" + dataset)


def load_poss(eda_out, script, dataset):
    """The summary of poss_<script> for one dataset, or None (with a note) if not run."""
    return load_summary(eda_out, poss_folder(script, dataset),
                        "poss_{}_summary.csv".format(script))
