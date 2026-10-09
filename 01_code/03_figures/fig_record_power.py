#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   fig_record_power.pdf, fig_record_power.png
  FEEDS      fig:record_power (ssec:record)
  INPUTS     none of its own; it imports 01_calibration/record_detectability.py, which reads
             02_output/record_study/horizon_attente.json and horizon_np.json
  SEED       none. It simulates nothing and interpolates nothing.
  RUNTIME    under 5 s
  IMPLEMENTS no equation: it plots the Neyman-Pearson bound the study computed

fig_record_power.py -- what the record can establish, drawn.

Two curves of the power of the best test (the Neyman-Pearson bound against the least
favourable null) for a fold, against the share of the remaining budget already spent:

  SOLID    the waiting diagonal: the record lengthens by one year for every year the
           threshold draws nearer (horizon_attente.json, block "diagonal", fold).
  DASHED   the record held at its present length L (horizon_np.json, fold, L = L).

The study evaluates both on a grid of t_c; the grid points are joined by straight segments,
with no smoothing. Years to the threshold are converted to the share of the budget spent by
the conversion record_detectability.py applies to the \recHorizonFold*SpentPct macros:
share = 1 - t_c / t_c0, t_c0 the diagonal's starting distance (\recNPtcNow), i.e. at constant
emissions the share of the budget spent is the share of the time.

NOTHING IS RECOMPUTED HERE. The macro strings written in the figure, the record length and
t_c0 all come from record_detectability.compute(), the evaluation that collect_computed.py
parses for the \rec* macros, so the figure and the macros are two views of one evaluation.
The 50% and 80% markers sit at h50 and h80 as the study states them; the joined diagonal
passes through them because the study obtained them by linear interpolation on the same grid,
and that is asserted below rather than assumed.

Usage:  python 01_code/03_figures/fig_record_power.py    (writes into the current directory)
"""
# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
FIGURE_MACROS = (
    "recNPFoldNow",
    "recHorizonFoldFiftySpentPct",
    "recHorizonFoldEightySpentPct",
    "recNPFoldNear",
    "recRecordYears",
)

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "01_calibration"))
from record_detectability import compute  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

C_FOLD = "#2b7bba"     # the fold, as in fig_bifurcation_intro.py
C_FIXED = "#8c8c8c"
C_TIP = "#d73027"      # markers, red-orange
C_TXT, C_GREY = "#111111", "#555555"

out, ctx = compute(printing=False)
L, tc0 = ctx["L"], float(ctx["tc0"])


def spent(tc):
    """Share of the remaining budget spent, in percent: the conversion of the SpentPct macros."""
    return 100.0 * (1.0 - np.asarray(tc, float) / tc0)


# ── the two curves, as the study evaluated them ─────────────────────────────
diag = {d["model"]: d for d in ctx["att"]["diagonal"]}["lin-1/2"]
cells = sorted(diag["cells"], key=lambda c: -c["tc"])
x_diag = spent([c["tc"] for c in cells])
y_diag = np.array([c["bound"] for c in cells])

fixed = sorted((c for c in ctx["hnp"]["cells"]
                if c["model"] == "lin-1/2" and c["L"] == L and c["tc"] <= tc0),
               key=lambda c: -c["tc"])
x_fix = spent([c["tc"] for c in fixed])
y_fix = np.array([c["bound"] for c in fixed])

# the two horizons, where the study put them
x50, x80 = spent(diag["h50"]), spent(diag["h80"])
for x, lvl in ((x50, 0.5), (x80, 0.8)):
    if abs(np.interp(x, x_diag, y_diag) - lvl) > 1e-9:
        raise SystemExit("[ERROR] fig_record_power.py: the joined diagonal does not pass "
                         "through the study's horizon at power %.1f" % lvl)
for key, x in (("recHorizonFoldFiftySpentPct", x50), ("recHorizonFoldEightySpentPct", x80)):
    if "%.0f" % x != out[key]:
        raise SystemExit("[ERROR] fig_record_power.py: marker at %.2f%% but %s = %s"
                         % (x, key, out[key]))

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.grid(True, axis="y", lw=0.4, alpha=0.3)
ax.set_axisbelow(True)

ax.plot(x_diag, y_diag, color=C_FOLD, lw=1.8, marker="o", ms=3,
        label="record lengthens as the threshold nears")
ax.plot(x_fix, y_fix, color=C_FIXED, lw=1.5, ls="--", marker="o", ms=2.5,
        label="record held at %s years" % out["recRecordYears"])

# today
ax.plot([0], [ctx["b_fold_now"]], "o", color=C_TXT, ms=6, zorder=5, clip_on=False)
ax.annotate("today: %s" % out["recNPFoldNow"], xy=(0, ctx["b_fold_now"]),
            xytext=(3.0, 0.02), fontsize=10, color=C_TXT, ha="left", va="bottom")

# the two horizons, projected on both axes
for x, lvl, txt, dx in ((x50, 0.5, "even odds: %s%% spent" % out["recHorizonFoldFiftySpentPct"], -3),
                        (x80, 0.8, "four in five: %s%% spent" % out["recHorizonFoldEightySpentPct"], -3)):
    ax.plot([x, x], [0, lvl], color=C_TIP, lw=0.8, ls=":")
    ax.plot([0, x], [lvl, lvl], color=C_TIP, lw=0.8, ls=":")
    ax.plot([x], [lvl], "o", color=C_TIP, ms=6, zorder=5)
    ax.text(x + dx, lvl + 0.03, txt, fontsize=10, color=C_TXT, ha="right", va="bottom")

# the ceiling on the present record, to the right of the dashed curve's end
ax.text(x_fix[-1] + 1.5, y_fix[-1], "%s on the\npresent record" % out["recNPFoldNear"],
        fontsize=10, color=C_GREY, ha="left", va="center", clip_on=False)

ax.set_xlim(0, 100)
ax.set_ylim(0, 1.0)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("share of the remaining carbon budget already spent")
ax.set_ylabel("power of the best test")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=10)

fig.savefig("fig_record_power.pdf", dpi=200, bbox_inches="tight")
fig.savefig("fig_record_power.png", dpi=150, bbox_inches="tight")
print("fig_record_power.py: diagonal %d grid points, fixed record %d grid points (L = %d); "
      "t_c0 = %g; h50 -> %.2f%%, h80 -> %.2f%% spent"
      % (len(x_diag), len(x_fix), L, tc0, x50, x80))
print("  origin: diagonal %.4f, fixed %.4f, fine grid %.4f (the macro); end of fixed %.4f at "
      "%.1f%%, fine grid near %.4f (the macro)"
      % (y_diag[0], y_fix[0], ctx["b_fold_now"], y_fix[-1], x_fix[-1], ctx["b_fold_near"]))
print("saved fig_record_power.pdf/.png")
