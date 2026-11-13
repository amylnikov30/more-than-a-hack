# %% [markdown]
# # Possession signals during live play: which are worth putting into the model?
#
# For each candidate signal for the live-play possession model this notebook answers three
# feasibility questions: is the signal **available** often enough, does it **separate** the two
# teams, and is it **sharp** enough in time? It does not build the model.
#
# Every number comes from the aggregate `poss_*_summary.csv` files of the `eda/scripts/poss_*.py`
# scripts (definitions in `eda/docs/analyses.md`). Three runs of each script sit side by side:
#
# - **match**: the competition match, labelled by **Smart Tagging pseudo-labels** (automatic
#   tactical phases mapped to "who has the ball"). They are noisy: a phase is not a possession,
#   so even a perfect signal scores well below 1 against them. That is why the match bar for a
#   Go (0.65) is lower than the reference bar (0.75).
# - **Sample Game 3, clean**: Metrica's open game, labelled from its open event data.
# - **Sample Game 3, degraded**: the same game with the ball dropped on 41% of live frames and
#   player tracks broken and re-slotted, to look like our automatic tracking.
#
# Nothing here is indexed by match time or frame: charts show windows, bins, contexts or
# seconds relative to an event. The first section is the summary; one section per analysis
# follows, each ending with an automatic verdict and a blank takeaway for the reader.

# %%
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EDA_DIR = Path(os.environ.get("EDA_DIR", Path.cwd()))
sys.path.insert(0, str(EDA_DIR))
import plot_helpers as ph  # noqa: E402

try:
    from IPython.display import display
except ImportError:           # plain-Python smoke test
    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") else obj)

EDA_OUT = Path(os.path.expanduser(os.environ.get("EDA_OUT", "~/eda_out")))
ph.style()

# ---- Feasibility criteria (edit here) ------------------------------------------------------
GO_AUC_DEGRADED = 0.75     # AUC on degraded Sample Game 3
GO_AUC_MATCH = 0.65        # AUC against the match pseudo-labels (noisy, so lower)
GO_COVERAGE = 0.20         # share of live frames where the signal exists
GO_TIMING_S = 1.0          # median |timing error|, only for signals used to place changes
MAYBE_AUC = 0.05           # "missed by a small margin"
MAYBE_COVERAGE = 0.05
MAYBE_TIMING_S = 0.5
GO_RESTART_ACCURACY = 0.90   # restarts: accuracy of the best rule on degraded Sample Game 3
# ---------------------------------------------------------------------------------------------

WINDOWS = [0.5, 1.0, 2.0, 5.0, 10.0]
ALL_WINDOWS = [0.0] + WINDOWS
DATASETS = ph.DATASETS
_cache = {}


def load(script, ds):
    key = (script, ds)
    if key not in _cache:
        _cache[key] = ph.load_poss(EDA_OUT, script, ds)
    return _cache[key]


def wlabel(w):
    return "frame" if w == 0 else "{:g} s".format(w)


def sep(script, ds, scope, w, metric="auc"):
    df = load(script, ds)
    return np.nan if df is None else ph.get(df, "separation", scope, "{:g}".format(w), metric)


def best_window(script, scope, ds="sg3_deg"):
    vals = [(sep(script, ds, scope, w), w) for w in ALL_WINDOWS]
    vals = [(v, w) for v, w in vals if np.isfinite(v)]
    return max(vals)[1] if vals else np.nan


def verdict(auc_degraded, auc_match, coverage, timing=None):
    """'Go' / 'Maybe' / 'No' from the criteria above; NaN inputs are not judged (noted)."""
    checks = [(auc_degraded, GO_AUC_DEGRADED, MAYBE_AUC, True),
              (auc_match, GO_AUC_MATCH, MAYBE_AUC, True),
              (coverage, GO_COVERAGE, MAYBE_COVERAGE, True)]
    if timing is not None:
        checks.append((timing, GO_TIMING_S, MAYBE_TIMING_S, False))
    misses, small, pending = 0, 0, []
    for value, bar, margin, higher in checks:
        if not np.isfinite(value):
            pending.append(bar)
            continue
        ok = value >= bar if higher else value <= bar
        if not ok:
            misses += 1
            gap = bar - value if higher else value - bar
            small += gap <= margin
    out = "Go" if misses == 0 else ("Maybe" if misses == 1 and small == 1 else "No")
    if pending:
        out += " (incomplete: {} criteria without data)".format(len(pending))
    return out


def fmt(v, f="{:.2f}"):
    return "-" if v is None or not np.isfinite(v) else f.format(v)


for script in ("ball_live", "carries", "pass_chains", "shape", "ball_field", "reversal",
               "contested", "restarts", "durations"):
    have = [ds for ds in DATASETS if load(script, ds) is not None]
    print("{:<12} {}".format(script, ", ".join(have) or "no runs found"))

# %% [markdown]
# ## 0. Summary
#
# One row per candidate signal. **AUC** is the separation of "home has the ball" from "away has
# the ball" at the signal's best smoothing window (picked on degraded Sample Game 3), with an
# absent signal counted as no evidence. **Coverage** is the share of live frames where the
# smoothed signal exists (degraded). **Timing** is the median |error| in seconds between a
# signal event and the true change (clean Sample Game 3); it counts toward the verdict only for
# signals meant to place changes (`places changes`). Restarts, ball availability and
# durations are not AUC signals and get their own lines below the table.

# %%
SIGNALS = [  # (idea, label, script, scope, timing scope or None, places changes)
    ("2 carries", "carry", "carries", "carry", "carry_switch", False),
    ("3 pass chains", "pass chain (>= 2 passes)", "pass_chains", "chain", "chain_switch", False),
    ("4 shape", "width", "shape", "width:all", None, False),
    ("4 shape", "hull area", "shape", "hull_area:all", None, False),
    ("4 shape", "length", "shape", "length:all", None, False),
    ("4 shape", "players goal-side (fewer)", "shape", "goal_side:all", None, False),
    ("4 shape", "team velocity", "shape", "centroid_velocity:all", None, False),
    ("4 shape", "centroid advance", "shape", "centroid_advance:all", None, False),
    ("4 shape", "line depth", "shape", "line_depth:all", None, False),
    ("5 ball field", "ball position", "ball_field", "ball_position", None, False),
    ("5 ball field", "ball direction", "ball_field", "ball_direction", None, False),
    ("5 ball field", "team drift", "ball_field", "team_drift", None, False),
    ("6 reversal", "implied by last reversal", "reversal", "reversal_state", "reversal", True),
    ("7 contested", "nearest team (baseline)", "contested", "nearest_team", None, False),
]


def timing_of(script, scope, ds="sg3"):
    df = load(script, ds)
    return np.nan if (df is None or scope is None) else \
        ph.get(df, "timing", scope, "", "median_abs_error")


rows = []
for idea, label, script, scope, tscope, places in SIGNALS:
    w = best_window(script, scope)
    a_clean = sep(script, "sg3", scope, w) if np.isfinite(w) else np.nan
    a_deg = sep(script, "sg3_deg", scope, w) if np.isfinite(w) else np.nan
    a_match = sep(script, "match", scope, w) if np.isfinite(w) else np.nan
    cov = sep(script, "sg3_deg", scope, w, "coverage") if np.isfinite(w) else np.nan
    t = timing_of(script, tscope)
    rows.append({"idea": idea, "signal": label, "coverage (deg)": fmt(cov),
                 "AUC SG3 clean / degraded": "{} / {}".format(fmt(a_clean), fmt(a_deg)),
                 "AUC match (pseudo)": fmt(a_match), "timing |err| s": fmt(t),
                 "best window": wlabel(w) if np.isfinite(w) else "-",
                 "verdict": verdict(a_deg, a_match, cov, t if places else None)})
summary = pd.DataFrame(rows)
display(summary)

# Context lines: ball availability, restarts, durations
lines = []
for ds in ("match", "sg3_deg"):
    df = load("ball_live", ds)
    if df is not None:
        lines.append("ball missing on live frames ({}): {} before, {} after interpolation".format(
            ph.DATASET_LABELS[ds],
            fmt(ph.get(df, "coverage", "all", "missing_before_interp", "share"), "{:.1%}"),
            fmt(ph.get(df, "coverage", "all", "missing", "share"), "{:.1%}")))
df = load("restarts", "sg3_deg")
if df is not None:
    acc = {r: ph.get(df, "rules", r, "", "accuracy") for r in
           ("geometric", "last_opposite", "label_opposite", "tag_team")}
    acc = {k: v for k, v in acc.items() if np.isfinite(v)}
    if acc:
        best = max(acc, key=acc.get)
        lines.append("restarts: best rule on degraded Sample Game 3 is {} ({:.1%}) -> {}".format(
            best, acc[best], "Go" if acc[best] >= GO_RESTART_ACCURACY else "No"))
df = load("durations", "sg3")
if df is not None:
    lines.append("possession length on Sample Game 3: median {} s, mean {} s; {} end in a "
                 "turnover".format(fmt(ph.get(df, "possessions", "all", "p50", "length"), "{:.1f}"),
                                   fmt(ph.get(df, "possessions", "all", "mean", "length"), "{:.1f}"),
                                   fmt(ph.get(df, "ends", "ended", "", "turnover_share"),
                                       "{:.0%}")))
print("\n".join(lines))

# %% [markdown]
# ### Scale map 1: at which window does each signal separate the teams?
#
# Rows are signals, columns smoothing windows (a centred rolling mean of the signal), cells
# the AUC on degraded Sample Game 3 (left) and against the match pseudo-labels (right, when
# run). Read along a row: a signal whose AUC climbs with the window is a slow, holding signal;
# one that peaks early is sharp. Colours run from 0.5 (chance) to 1.

# %%
labels_hm = [label for _, label, _, _, _, _ in SIGNALS]
panels = [ds for ds in ("sg3_deg", "match")
          if any(load(s[2], ds) is not None for s in SIGNALS)]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 0.42 * len(SIGNALS) + 1.6),
                             squeeze=False)
    for ax, ds in zip(axes[0], panels):
        data = [[sep(script, ds, scope, w) for w in WINDOWS]
                for _, _, script, scope, _, _ in SIGNALS]
        im = ph.annotated_heatmap(ax, data, labels_hm if ax is axes[0][0] else [""] * len(SIGNALS),
                                  [wlabel(w) for w in WINDOWS])
        ax.set_title("Which window separates the teams?\n" + ph.DATASET_LABELS[ds])
        ax.set_xlabel("smoothing window (s)")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label="AUC (home vs away possession)")
    plt.show()
else:
    print("no separation runs found under", EDA_OUT)

# %% [markdown]
# ### Scale map 2: how sharp are the signals that fire at changes?
#
# Left: the timing error of each change signal against the true changes of Sample Game 3
# (event minus change, in s; quartiles and the median absolute error), clean and degraded.
# Colour shows the size of the error (darker is closer to the change). Right: change
# detection, the AUC of the signal's density for "a change is near" against other live play
# (a different target from scale map 1), per window.

# %%
TIMED = [("carry switch", "carries", "carry_switch"), ("pass-chain switch", "pass_chains",
                                                     "chain_switch"),
         ("joint reversal", "reversal", "reversal")]
CHANGE_AUC = [("reversal density", "reversal", "reversal_change"),
              ("nearest-team flips vs mid", "contested", "contested_change"),
              ("nearest-team flips vs loose ball", "contested", "contested_loose")]
cols = ["p25", "p50", "p75", "median |err|"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
for ax, ds in zip(axes[:2], ("sg3", "sg3_deg")):
    signed = []
    for _, script, scope in TIMED:
        df = load(script, ds)
        signed.append([np.nan] * 4 if df is None else
                      [ph.get(df, "timing", scope, b, "error") for b in ("p25", "p50", "p75")] +
                      [ph.get(df, "timing", scope, "", "median_abs_error")])
    signed = np.array(signed, dtype=float)
    ph.annotated_heatmap(ax, np.abs(signed), [t[0] for t in TIMED] if ds == "sg3" else
                         [""] * len(TIMED), cols, fmt="{:+.2f}", vmin=0, vmax=3,
                         cmap="cividis_r", text=signed)
    ax.set_title("How far from the change does it fire?\n" + ph.DATASET_LABELS[ds])
    ax.set_xlabel("timing error statistic (s)")
data = [[sep(script, "sg3_deg", scope, w) for w in WINDOWS] for _, script, scope in CHANGE_AUC]
ph.annotated_heatmap(axes[2], data, [c[0] for c in CHANGE_AUC], [wlabel(w) for w in WINDOWS])
axes[2].set_title("Does it mark that a change is near?\n" + ph.DATASET_LABELS["sg3_deg"])
axes[2].set_xlabel("smoothing window (s)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 1. Ball availability on live play (`poss_ball_live.py`)
#
# **Question.** How much of live play has no ball, and in how long gaps? This sets how much the
# ball-free signals (shape, reversal) must carry in the model.

# %%
# Run commands (team-repo root; see eda/README.md):
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_ball_live.py "$HOME/eda_out/poss_ball_live" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_ball_live.py "$HOME/eda_out/poss_ball_live_sg3" --raw ... --events ... (Sample Game 3)
#   ... poss_ball_live.py "$HOME/eda_out/poss_ball_live_sg3_deg" ... --degrade
BINS = ["<0.5s", "0.5-2s", "2-5s", ">=5s"]
present = [ds for ds in ("match", "sg3_deg") if load("ball_live", ds) is not None]
if present:
    fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], present):
        df = load("ball_live", ds)
        x = np.arange(len(BINS))
        for k, (stage, colour) in enumerate((("before", ph.INK_3), ("after", ph.REAL))):
            v = [ph.get(df, "gaps_" + stage, "all", b, "share") for b in BINS]
            ax.bar(x + (k - 0.5) * 0.38, v, width=0.38, color=colour,
                   label="{} interpolation".format(stage))
        ax.set_xticks(x, BINS)
        ax.set_xlabel("ball gap length")
        ax.set_ylabel("share of missing live frames")
        ax.set_title("Is the missing ball in short or long gaps?\n" + ph.DATASET_LABELS[ds])
        ax.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **How to read it.** Bars show where the missing live frames sit by gap length, before (light)
# and after (dark) short gaps are interpolated. Mass in the `>=5s` bin is time the model must
# label without a ball, from players alone.

# %%
for ds in DATASETS:
    df = load("ball_live", ds)
    if df is None:
        continue
    print("{}: missing {} of live frames before, {} after interpolation; within 2 s of a change "
          "{} vs {} elsewhere".format(
              ph.DATASET_LABELS[ds],
              fmt(ph.get(df, "coverage", "all", "missing_before_interp", "share"), "{:.1%}"),
              fmt(ph.get(df, "coverage", "all", "missing", "share"), "{:.1%}"),
              fmt(ph.get(df, "change", "around", "before", "missing_share"), "{:.1%}"),
              fmt(ph.get(df, "change", "rest", "before", "missing_share"), "{:.1%}")))
print("Verdict: context, not a signal; the after-interpolation share is what the ball-free "
      "signals must cover.")

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 2. Carries (`poss_carries.py`)
#
# **Question.** When the ball moves with one player, is that player's team in possession, and
# how much live time does it cover? Feeds the per-frame emission of the model.

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_carries.py "$HOME/eda_out/poss_carries" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_carries.py "$HOME/eda_out/poss_carries_sg3" (and _sg3_deg with --degrade)


def auc_by_window(script, scope, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(ALL_WINDOWS))
    for ds in DATASETS:
        vals = [sep(script, ds, scope, w) for w in ALL_WINDOWS]
        if np.all(~np.isfinite(vals)):
            continue
        colour, marker = ph.dataset_style(ds)
        ax.plot(x, vals, color=colour, marker=marker, label=ph.DATASET_LABELS[ds])
    ph.threshold(ax, GO_AUC_DEGRADED, "Go bar, degraded SG3 ({:g})".format(GO_AUC_DEGRADED))
    ph.threshold(ax, GO_AUC_MATCH, "Go bar, match ({:g})".format(GO_AUC_MATCH))
    ax.set_xticks(x, [wlabel(w) for w in ALL_WINDOWS])
    ax.set_xlabel("smoothing window (s)")
    ax.set_ylabel("AUC, home vs away possession")
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


auc_by_window("carries", "carry", "Does the carrying team have the ball?")
plt.show()

# %% [markdown]
# **How to read it.** Each line is the AUC of the carry signal (+1 home carry, -1 away) after
# smoothing over the window on the x axis; dashed lines are the Go bars. A line that keeps
# rising with the window means carries are right but rare, and smoothing spreads them over
# the gaps between them.


# %%
def signal_verdict(script, scope, tscope=None, places=False):
    w = best_window(script, scope)
    a_deg, a_match = sep(script, "sg3_deg", scope, w), sep(script, "match", scope, w)
    cov = sep(script, "sg3_deg", scope, w, "coverage")
    t = timing_of(script, tscope)
    print("best window {}: AUC degraded {}, match {}, coverage {}, timing |err| {} s".format(
        wlabel(w), fmt(a_deg), fmt(a_match), fmt(cov), fmt(t)))
    print("Verdict:", verdict(a_deg, a_match, cov, t if places else None))


signal_verdict("carries", "carry", "carry_switch")
for ds in DATASETS:
    df = load("carries", ds)
    if df is not None:
        print("  {}: precision (carry team = label) {}".format(
            ph.DATASET_LABELS[ds], fmt(ph.get(df, "precision", "all", "", "precision"), "{:.3f}")))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 3. Pass chains (`poss_pass_chains.py`)
#
# **Question.** Does "team T completes two passes in a row" (the CHALLENGE.md proxy) mean T has
# the ball, and how much of live play does it cover? Feeds the emission model as an anchor.

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_pass_chains.py "$HOME/eda_out/poss_pass_chains" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_pass_chains.py "$HOME/eda_out/poss_pass_chains_sg3" (and _sg3_deg with --degrade)
auc_by_window("pass_chains", "chain", "Does the passing team have the ball?")
plt.show()

# %% [markdown]
# **How to read it.** As for carries, for chains of at least two passes. A flat line means the
# proxy is already as good as it gets on single frames; its limit is coverage.

# %%
signal_verdict("pass_chains", "chain", "chain_switch")
for ds in DATASETS:
    df = load("pass_chains", ds)
    if df is not None:
        print("  {}: precision by chain length {}".format(ph.DATASET_LABELS[ds], ", ".join(
            "{} {}".format(b, fmt(ph.get(df, "precision", "length", b, "precision"), "{:.2f}"))
            for b in ("1", "2", "3", "4", "5+"))))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 4. Team shape (`poss_shape.py`)
#
# **Question.** Does the team in possession spread out, push up and keep fewer players behind
# the ball, clearly enough to tell who has it without the ball? Feeds the ball-free emission.

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_shape.py "$HOME/eda_out/poss_shape" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_shape.py "$HOME/eda_out/poss_shape_sg3" (and _sg3_deg with --degrade)
FEATURES = [("width", "width"), ("hull_area", "hull area"), ("length", "length"),
            ("goal_side", "players goal-side"), ("centroid_velocity", "team velocity"),
            ("centroid_advance", "centroid advance"), ("line_depth", "line depth")]
fig, ax = plt.subplots(figsize=(7, 3.8))
y = np.arange(len(FEATURES))
for k, ds in enumerate(DATASETS):
    if load("shape", ds) is None:
        continue
    colour, marker = ph.dataset_style(ds)
    vals = [sep("shape", ds, f + ":all", best_window("shape", f + ":all")) for f, _ in FEATURES]
    ax.plot(vals, y + (k - 1) * 0.18, ls="none", marker=marker, color=colour,
            label=ph.DATASET_LABELS[ds])
ph.threshold(ax, GO_AUC_DEGRADED, "Go, degraded", axis="x")
ph.threshold(ax, GO_AUC_MATCH, "Go, match", axis="x")
ax.set_yticks(y, [name for _, name in FEATURES])
ax.invert_yaxis()
ax.set_xlim(0.3, 1.0)
ax.set_xlabel("AUC at the best window, home vs away possession")
ax.set_title("Which shape features tell who has the ball?")
ax.legend(loc="lower left", fontsize=8)
plt.show()

# %% [markdown]
# **How to read it.** One row per feature, one marker per run, at each feature's best window
# (chosen on degraded Sample Game 3); right of the dashed lines passes the Go bars. The table
# below adds the no-ball split and each team's defensive-line depth, which tests the "low
# block" assumption instead of assuming it.

# %%
for f, name in FEATURES:
    scope = f + ":all"
    w = best_window("shape", scope)
    a_deg, a_match = sep("shape", "sg3_deg", scope, w), sep("shape", "match", scope, w)
    cov = sep("shape", "sg3_deg", scope, w, "coverage")
    print("{:<18} window {:<6} degraded {} (no ball {}), match {} (no ball {}) -> {}".format(
        name, wlabel(w), fmt(a_deg), fmt(sep("shape", "sg3_deg", f + ":no_ball", w)),
        fmt(a_match), fmt(sep("shape", "match", f + ":no_ball", w)),
        verdict(a_deg, a_match, cov)))
depth = []
for ds in DATASETS:
    df = load("shape", ds)
    if df is None:
        continue
    for team in ("H", "A"):
        for state in ("defending", "attacking"):
            depth.append({"run": ph.DATASET_LABELS[ds], "team": team, "state": state,
                          "line depth p50 (m from own goal)":
                          fmt(ph.get(df, "line_depth", "{}:{}".format(team, state), "p50",
                                     "depth"), "{:.1f}")})
if depth:
    display(pd.DataFrame(depth))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 5. Ball position vs ball direction vs team drift (`poss_ball_field.py`)
#
# **Question.** Which tells who has the ball better: where the ball is, which way it moves, or
# which way both teams drift? If position underperforms, it is confounded (pressing, build-up).

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_ball_field.py "$HOME/eda_out/poss_ball_field" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_ball_field.py "$HOME/eda_out/poss_ball_field_sg3" (and _sg3_deg with --degrade)
FIELD = [("ball_position", "ball position", ph.GREYS[0], "o"),
         ("ball_direction", "ball direction", ph.GREYS[1], "s"),
         ("team_drift", "team drift", ph.GREYS[2], "^")]
present = [ds for ds in ("match", "sg3_deg") if load("ball_field", ds) is not None]
if present:
    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 3.6), squeeze=False,
                             sharey=True)
    x = np.arange(len(ALL_WINDOWS))
    for ax, ds in zip(axes[0], present):
        for scope, name, colour, marker in FIELD:
            ax.plot(x, [sep("ball_field", ds, scope, w) for w in ALL_WINDOWS], color=colour,
                    marker=marker, label=name)
        ph.threshold(ax, GO_AUC_DEGRADED if ds != "match" else GO_AUC_MATCH, "Go bar")
        ax.axhline(0.5, color=ph.INK_3, lw=0.8)
        ax.set_xticks(x, [wlabel(w) for w in ALL_WINDOWS])
        ax.set_xlabel("smoothing window (s)")
        ax.set_ylabel("AUC, home vs away possession")
        ax.set_title("Position, direction or drift?\n" + ph.DATASET_LABELS[ds])
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **How to read it.** Three scores per panel; the solid grey line is chance (0.5). Below it, the
# score points the wrong way: on Sample Game 3 a team has the ball more often when the ball is
# in its *own* half. The table compares the two teams by the ball's third of the pitch.

# %%
for scope, name, _, _ in FIELD:
    print("{:<15}".format(name), end=" ")
    signal_verdict("ball_field", scope)
thirds = []
for ds in DATASETS:
    df = load("ball_field", ds)
    if df is None:
        continue
    for team in ("H", "A"):
        thirds.append({"run": ph.DATASET_LABELS[ds], "team": team, **{
            third: fmt(ph.get(df, "thirds", "{}:{}".format(team, third), "",
                              "share_team_has_ball"))
            for third in ("defensive", "middle", "attacking")}})
if thirds:
    print("share of ball-present frames the team has the ball, by the ball's third in its "
          "attacking frame:")
    display(pd.DataFrame(thirds))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 6. Joint reversals (`poss_reversal.py`)
#
# **Question.** Do both teams turn around at a turnover, so a change can be placed without the
# ball, and how late? Feeds the change detector, where spellF1 is won or lost.

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_reversal.py "$HOME/eda_out/poss_reversal" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_reversal.py "$HOME/eda_out/poss_reversal_sg3" (and _sg3_deg with --degrade)
present = [ds for ds in DATASETS if load("reversal", ds) is not None]
if present:
    fig, axes = plt.subplots(1, len(present), figsize=(4.8 * len(present), 3.4), squeeze=False,
                             sharey=True)
    for ax, ds in zip(axes[0], present):
        df = load("reversal", ds)
        g = ph.pick(df, "profile", "gainer").copy()
        bins = sorted(set(float(b) for b in g["bin"]))
        x = np.array(bins) + 0.5
        ph.null_band(ax, x, [ph.get(df, "profile", "gainer", "{:g}".format(b), "null_p5")
                             for b in bins],
                     [ph.get(df, "profile", "gainer", "{:g}".format(b), "null_p95")
                      for b in bins],
                     [ph.get(df, "profile", "gainer", "{:g}".format(b), "null_mean")
                      for b in bins], label="random anchors (null): p5-p95, mean")
        for role, colour in (("gainer", ph.REAL), ("loser", ph.GREYS[2])):
            ax.plot(x, [ph.get(df, "profile", role, "{:g}".format(b), "value") for b in bins],
                    color=colour, marker="o", ms=3, label="team {} the ball".format(
                        "gaining" if role == "gainer" else "losing"))
        ax.axhline(0, color=ph.INK_3, lw=0.8)
        ax.axvline(0, color=ph.INK_3, lw=0.8)
        ax.set_xlabel("seconds from the possession change")
        ax.set_ylabel("velocity toward the gainer's goal (m/s)")
        ax.set_title("When do both teams turn around?\n" + ph.DATASET_LABELS[ds])
        ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# **How to read it.** Each team's velocity in the attacking direction of the team gaining the
# ball, averaged over all changes, against the same profile around random live moments (grey
# band). Where the dark line crosses zero is how long after the change the gaining team turns:
# that lag is how late a reversal-based change would land.

# %%
signal_verdict("reversal", "reversal_state", "reversal", places=True)
for ds in DATASETS:
    df = load("reversal", ds)
    if df is not None:
        print("  {}: {} joint reversals per live minute vs null mean {}; {} of changes matched, "
              "median error {} s".format(
                  ph.DATASET_LABELS[ds], fmt(ph.get(df, "reversals", "all", "", "per_min")),
                  fmt(ph.get(df, "reversals", "null", "", "null_mean"), "{:.1f}"),
                  fmt(ph.get(df, "timing", "reversal", "", "matched_share"), "{:.0%}"),
                  fmt(ph.get(df, "timing", "reversal", "p50", "error"))))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 7. Contested ball (`poss_contested.py`)
#
# **Question.** When the nearest team flips back and forth, is a turnover happening or just a
# contest? If the two differ, contested play could suppress false changes.

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_contested.py "$HOME/eda_out/poss_contested" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_contested.py "$HOME/eda_out/poss_contested_sg3" (and _sg3_deg with --degrade)
CTX = [("around_change", "around a change"), ("loose_no_change", "loose ball, no change"),
       ("mid", "mid-possession")]
present = [ds for ds in DATASETS if load("contested", ds) is not None]
if present:
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(CTX))
    width = 0.8 / len(present)
    for k, ds in enumerate(present):
        df = load("contested", ds)
        colour, _ = ph.dataset_style(ds)
        ax.bar(x + (k - (len(present) - 1) / 2) * width,
               [ph.get(df, "contexts", c, "", "flip_rate") for c, _ in CTX], width=width,
               color=colour, label=ph.DATASET_LABELS[ds])
    ax.set_xticks(x, [name for _, name in CTX])
    ax.set_ylabel("nearest-team flips per second")
    ax.set_title("Is play more contested around turnovers than elsewhere?")
    ax.legend(fontsize=8)
    plt.show()

# %% [markdown]
# **How to read it.** Flip rates of the nearest team in three kinds of moment. If "around a
# change" is well above "loose ball, no change", flips can tell a turnover from a contest; if
# they are close (as on Sample Game 3), flips only say that something is happening.

# %%
print("nearest team (the baseline signal):", end=" ")
signal_verdict("contested", "nearest_team")
for key, name in (("contested_change", "turnover vs mid-possession"),
                  ("contested_loose", "turnover vs loose ball")):
    print("flip density, {}: ".format(name) + ", ".join(
        "{} {}".format(wlabel(w), fmt(sep("contested", "sg3_deg", key, w))) for w in ALL_WINDOWS))
print("Verdict for suppressing false changes: {}".format(
    "Go" if sep("contested", "sg3_deg", "contested_loose", 2.0) >= GO_AUC_DEGRADED else "No"))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 8. Restart team after a dead ball (`poss_restarts.py`)
#
# **Question.** Which rule names the team restarting play? Feeds the entry state of every
# possession after a `D` stretch.

# %%
# Run commands:
#   cd ~/more-than-a-hack && PYTHONPATH="$PWD" python3 eda/scripts/poss_restarts.py "$HOME/eda_out/poss_restarts" --dead "$HOME/eda_out/dead_ball/dead_ball.csv"
#   ... poss_restarts.py "$HOME/eda_out/poss_restarts_sg3" (and _sg3_deg with --degrade)
RULES = ["geometric", "last_opposite", "label_opposite", "tag_team"]
present = [ds for ds in ("sg3", "sg3_deg") if load("restarts", ds) is not None]
if present:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(RULES))
    for k, ds in enumerate(present):
        df = load("restarts", ds)
        colour, _ = ph.dataset_style(ds)
        ax.bar(x + (k - 0.5) * 0.38, [ph.get(df, "rules", r, "", "accuracy") for r in RULES],
               width=0.38, color=colour, label=ph.DATASET_LABELS[ds])
    ph.threshold(ax, GO_RESTART_ACCURACY, "Go ({:g})".format(GO_RESTART_ACCURACY))
    ax.set_xticks(x, RULES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy (share of restarts)")
    ax.set_title("Which rule names the restarting team?")
    ax.legend(loc="lower left", fontsize=8)
    plt.show()

# %% [markdown]
# **How to read it.** Accuracy of each rule against the true restart team of Sample Game 3 (no
# Smart Tagging there, so `tag_team` is empty). The open data puts the ball at the taker's
# feet, so `geometric` is optimistic; the match table below shows how the rules agree on the
# match, where there is no truth.

# %%
df = load("restarts", "match")
if df is not None:
    m = ph.pick(df, "agreement_matrix", metric="agree_share")
    n = ph.pick(df, "agreement_matrix", metric="n")
    table = pd.DataFrame({"pair": m["scope"].values, "agree": m["value"].round(2).values,
                          "restarts": n["value"].values})
    display(table)
df = load("restarts", "sg3_deg")
if df is not None:
    acc = {r: ph.get(df, "rules", r, "", "accuracy") for r in RULES}
    acc = {k: v for k, v in acc.items() if np.isfinite(v)}
    best = max(acc, key=acc.get) if acc else None
    print("Verdict: {} (best rule on degraded Sample Game 3: {} at {})".format(
        "Go" if best and acc[best] >= GO_RESTART_ACCURACY else "No", best,
        fmt(acc.get(best, np.nan), "{:.1%}")))

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_

# %% [markdown]
# ## 9. Possession duration prior (`poss_durations.py`, Sample Game 3 only)
#
# **Question.** How long do live possessions last and how do they end? This is the duration
# prior of a hidden semi-Markov model, and the possession count to expect per live minute.

# %%
# Run command (reference game only; the degraded copy has the same labels):
#   cd ~/more-than-a-hack && SG3=/path/to/Sample_Game_3 && python3 eda/scripts/poss_durations.py "$HOME/eda_out/poss_durations_sg3" --raw ... --events ...
# drawn edges: the open-ended bins are drawn over a comparable log width (0.5-2 s, 40-160 s)
DUR_BINS = [("0-2s", 0.5, 2), ("2-5s", 2, 5), ("5-10s", 5, 10), ("10-20s", 10, 20),
            ("20-40s", 20, 40), (">=40s", 40, 160)]
df = load("durations", "sg3")
if df is not None:
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for k, team in enumerate(("H", "A")):
        for label, lo, hi in DUR_BINS:
            left = lo * (hi / lo) ** (0.5 * k)
            right = lo * (hi / lo) ** (0.5 * (k + 1))
            ax.bar(left, ph.get(df, "bins", team, label, "n"), width=right - left, align="edge",
                   color=ph.team_colour(team), edgecolor="white",
                   label=ph.TEAM_LABELS[team].split(";")[0] + ")" if label == "0-2s" else None)
    ph.log_seconds(ax, "possession length")
    ax.set_ylabel("possessions (count)")
    ax.set_title("How long do live possessions last?")
    ax.legend(fontsize=8)
    plt.show()

# %% [markdown]
# **How to read it.** Possessions per length bin (bin edges at 2, 5, 10, 20 and 40 s on a log
# axis), home and away side by side within each bin. The first bin holds everything under
# 2 s and the last everything from 40 s; they are drawn from 0.5 s and up to 160 s so their
# width matches the others. Event-derived labels count every
# recovery as a change, so the shortest bin is likely an overcount of what the ground truth
# calls a possession.

# %%
if df is not None:
    print("median {} s, mean {} s; log-normal fit: mean ln(s) {}, sd {}".format(
        fmt(ph.get(df, "possessions", "all", "p50", "length"), "{:.1f}"),
        fmt(ph.get(df, "possessions", "all", "mean", "length"), "{:.1f}"),
        fmt(ph.get(df, "fit", "lognormal", "", "log_mean")),
        fmt(ph.get(df, "fit", "lognormal", "", "log_sd"))))
    print("hazard per bin: " + ", ".join("{} {}".format(label, fmt(
        ph.get(df, "hazard", "all", label, "hazard"))) for label, _, _ in DUR_BINS))
    print("Verdict: use as the duration prior (a rising hazard favours a semi-Markov model over "
          "a plain HMM).")

# %% [markdown]
# **Takeaway.**
#
# _(to fill in)_
