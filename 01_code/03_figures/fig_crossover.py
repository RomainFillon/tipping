#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   fig_crossover.pdf, fig_crossover.png
  FEEDS      the crossover figure of the climate application (no LaTeX label yet: the
             manuscript does not display it until that subsection is written)
  INPUTS     none of its own; it imports 01_calibration/crossover.py, which reads
             inputs_literature.DERIVATION_INPUTS, and 01_calibration/eps_dyn.py,
             which reads the ews_calibration and calibration_ci captures for the shaded
             region below eps_dyn -- so the figure now runs after the network step
  SEED       none (deterministic)
  RUNTIME    under 5 s
  IMPLEMENTS the same power law SCC ~ eps^{p-1} that crossover.py evaluates

fig_crossover.py -- the figure the paper's thesis fits into.

Two curves against the proximity eps, on a log axis:

  LOCATION (flat)      what the WHOLE published disagreement over the AMOC threshold does
                       to the tipping price, at the fold exponent p = 1/2, with the budget to
                       each threshold moving with it (a ratio of budgets,
                       ((DT*_high - DT_now)/(DT*_low - DT_now))^(1/2)). It does not depend
                       on eps.
  CLASS (divergent)    what moving the exponent from p = 1/2 to p = 1 does at that same
                       proximity. It rises as eps -> 0; below eps_dyn (shaded, dashed) the
                       budget drains faster than the circulation recovers and the power law
                       no longer applies (the appendix on where the elements sit).

They cross just above eps_dyn (in an earlier version the flat line was computed on the
proximity and the crossing fell inside the published range). Left of the crossing, the
parameter nobody estimates moves the price more than the parameter everybody argues about.

NOTHING IS RECOMPUTED HERE. Every quantity comes from crossover.py, imported below, which
is also what collect_computed.py parses for the five \xover* macros. The figure and the
macros are therefore two views of ONE evaluation rather than two evaluations that agree
until they stop agreeing.

Usage:  python 01_code/03_figures/fig_crossover.py    (writes into the current directory)
"""
# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
FIGURE_MACROS = (
    "xoverEps",
    "xoverEpsLow",
    "xoverEpsHigh",
    "xoverRangeFactor",
    # The shaded region below the lower edge of the fixed-budget regime.
    "epsDynAMOC",
)

import sys
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "01_calibration"))
sys.path.insert(0, str(CODE))
from crossover import p_FOLD, p_SMOOTH, price, proximity, inp  # noqa: E402
DTlo, DThi = inp("DTstar_AMOC_p5"), inp("DTstar_AMOC_p95")
import eps_dyn as ed  # noqa: E402  (the formula behind the eps_dyn macro, imported, not copied)
import re  # noqa: E402

# The two numbers printed on the figure, the location factor and eps*, are the very
# strings crossover.py hands to \xoverRangeFactor and \xoverEps: read from the XOVER lines of
# its capture (run_all.sh runs crossover.py first), as collect_computed.py reads them, and
# checked below against the figure's own evaluation. Never typed.
_xcap = Path(ed.RAW) / "crossover.out"
if not _xcap.exists():
    raise SystemExit("[ERROR] fig_crossover.py: missing capture %s; run run_all.sh "
                     "(crossover.py runs before this figure)." % _xcap)
XOVER = dict(re.findall(r"^XOVER\s+(xover\w+)\s*=\s*(\S+)\s*$",
                        _xcap.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))
for _k in ("xoverRangeFactor", "xoverEps"):
    if _k not in XOVER:
        raise SystemExit("[ERROR] fig_crossover.py: %s not found in %s." % (_k, _xcap))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

C_LOC = "#1f6fb4"   # location: the disagreement the literature has
C_CLS = "#d73027"   # class: the one it does not
C_BAND = "#9e9e9e"

# ── the same inputs, through the same functions ──────────────────────────────
DTnow = inp("DT_current")
eps_lo = proximity(inp("DTstar_AMOC_p5"), DTnow)
eps_hi = proximity(inp("DTstar_AMOC_p95"), DTnow)
eps_mid = proximity(inp("DTstar_AMOC_best"), DTnow)

mu_ratio = (DTlo - DTnow) / (DThi - DTnow)      # mu_low / mu_high, as crossover.py
R_loc = price(mu_ratio, p_FOLD)
R_cls = lambda e: price(e, p_FOLD) / price(e, p_SMOOTH)
eps_star = mu_ratio

# The crossing is the figure's whole point, so it is asserted rather than eyeballed.
if abs(R_cls(eps_star) - R_loc) > 1e-9:
    raise SystemExit("[ERROR] fig_crossover.py: the marked crossing is not one.")
# The printed strings must be the figure's own numbers at the macros' precision.
if ("%.2f" % R_loc, "%.3f" % eps_star) != (XOVER["xoverRangeFactor"], XOVER["xoverEps"]):
    raise SystemExit("[ERROR] fig_crossover.py: the capture gives xoverRangeFactor = %s and "
                     "xoverEps = %s, the figure draws %.6f and %.6f."
                     % (XOVER["xoverRangeFactor"], XOVER["xoverEps"], R_loc, eps_star))

eps = np.logspace(math.log10(0.01), math.log10(1.0), 800)

# Below eps_dyn the budget drains faster than the circulation recovers and the power
# law no longer applies (the appendix on where the elements sit). Read through eps_dyn.py from the same captures as
# collect_computed.py, which checks the value printed below against the eps_dyn macro.
eps_dyn = ed.eps_dyn(*ed.amoc_inputs())
if not 0.01 < eps_dyn < eps_star < eps_lo:
    raise SystemExit("[ERROR] fig_crossover.py: expected 0.01 < eps_dyn < eps* < eps_low; the "
                     "shading and the crossing are drawn as though it held.")

fig, ax = plt.subplots(figsize=(8.2, 5.2))

# published range, hatched on the axis
band_fill = ax.axvspan(eps_lo, eps_hi, color=C_BAND, alpha=0.16, lw=0)
ax.axvspan(eps_lo, eps_hi, facecolor="none", edgecolor=C_BAND, hatch="////", lw=0.0,
           alpha=0.55)

loc_line, = ax.plot(eps, np.full_like(eps, R_loc), color=C_LOC, lw=2.6)
ax.axvspan(0.01, eps_dyn, color="#000000", alpha=0.10, lw=0)
ax.text(math.sqrt(0.01 * eps_dyn), 0.30, "budget drains\nfaster than\nrecovery",
        ha="center", va="center", fontsize=10, color="#444444",
        transform=ax.get_xaxis_transform())
_in = eps >= eps_dyn
ax.plot(eps[_in], [R_cls(e) for e in eps[_in]], color=C_CLS, lw=2.6)
ax.plot(eps[~_in], [R_cls(e) for e in eps[~_in]], color=C_CLS, lw=2.0, ls=(0, (3, 2)))

# The legend is replaced by labels on the lines, the location factor and eps* as
# their macros print them (XOVER above).
# Both sit above their line, inside the hatched band, left of the central threshold: the gap
# between eps* and the band is too narrow for either.
EPS_LBL = 0.20
loc_txt = ax.text(EPS_LBL, R_loc * 1.07, r"location: $\times$%s" % XOVER["xoverRangeFactor"],
                  ha="left", va="bottom", fontsize=12, color=C_LOC)
cls_txt = ax.text(EPS_LBL, R_cls(EPS_LBL) * 1.10, r"class: $\varepsilon^{-1/2}$",
                  ha="left", va="bottom", fontsize=12, color=C_CLS)

ax.plot([eps_star], [R_loc], "o", ms=8, color="#111111", zorder=5)
star_txt = ax.annotate(r"$\varepsilon^{\ast}=%s$" % XOVER["xoverEps"],
                       xy=(eps_star, R_loc), xytext=(eps_star * 1.35, R_loc * 1.55),
                       arrowprops=dict(arrowstyle="-", color="#111111", lw=1.0),
                       fontsize=12, color="#111111")

# The budget is spent from right to left; an arrow under the axis says so.
ax.annotate("", xy=(0.30, -0.255), xytext=(0.70, -0.255), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2, mutation_scale=14),
            annotation_clip=False)
spent_txt = ax.text(0.50, -0.275, "budget being spent", ha="center", va="top", fontsize=11.5,
                    color="#333333", transform=ax.transAxes, clip_on=False)

central_line = ax.axvline(eps_mid, color="#111111", lw=0.9, ls=":")
# The label of the central threshold sits above the frame, at the line, so that it
# touches neither the hatched band nor the blue line (checked below by the box rule).
central_txt = ax.text(eps_mid, 1.06, r"central $\Delta T^{*}$",
                      ha="center", va="bottom", fontsize=10.5, color="#444444",
                      transform=ax.get_xaxis_transform(), clip_on=False)

# Candidate places for the range label, tried in order once the layout is final;
# the first whose box touches neither the dotted line nor its label is kept (the box rule of
# fig_forest_tc.py).
RANGE_CANDS = [(math.sqrt(eps_lo * eps_hi), "center"),
               (eps_lo * 1.04, "left")]       # inside the hatched band, from its left edge
range_txt = ax.text(RANGE_CANDS[0][0], 0.055, "published threshold range",
                    ha=RANGE_CANDS[0][1], va="bottom", fontsize=10.5, color="#555555",
                    transform=ax.get_xaxis_transform())
# The hatched band is where the circulation stands today.
today_txt = ax.text(math.sqrt(eps_lo * eps_hi), 0.90, "today", ha="center", va="center",
                    fontsize=12, color="#333333", transform=ax.get_xaxis_transform())

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.01, 1.0)
ax.set_ylim(1.0, 12.0)
ax.set_yticks([1, 1.5, 2, 3, 5, 8, 12])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel(r"proximity to the threshold  $\varepsilon=(\Delta T^{*}-\Delta T_{0})/\Delta T^{*}$")
ax.set_ylabel(r"factor on the tipping price")
ax.grid(True, which="both", axis="y", lw=0.4, alpha=0.25)

fig.tight_layout()

_renderer = fig.canvas.get_renderer()
_PAD = 6.0   # pixels: a visible gap, not only no contact


def _box(a):
    return a.get_window_extent(_renderer).padded(_PAD)


def _touches(t):
    return any(_box(t).overlaps(_box(o)) for o in (central_line, central_txt))


for x, ha in RANGE_CANDS:
    range_txt.set_x(x); range_txt.set_ha(ha)
    if not _touches(range_txt):
        break
else:
    raise SystemExit("[ERROR] fig_crossover.py: the range label touches the central threshold "
                     "line or its label in every candidate place")
print("  range label placed at eps = %.4f (%s)" % (range_txt.get_position()[0], range_txt.get_ha()))
# The label of the central threshold may touch neither the hatched band nor the blue
# line. Nor may the line labels touch the band, each other or the eps* label, and
# "today" may touch neither line of the band's neighbourhood nor the central threshold.
_checks = [("central threshold label", central_txt, (("hatched band", band_fill),
                                                     ("location line", loc_line))),
           ("location label", loc_txt, (("class label", cls_txt), ("eps* label", star_txt),
                                        ("central threshold line", central_line))),
           ("class label", cls_txt, (("location line", loc_line), ("eps* label", star_txt),
                                     ("central threshold line", central_line),
                                     ("range label", range_txt))),
           ("today label", today_txt, (("location line", loc_line),
                                       ("central threshold line", central_line),
                                       ("central threshold label", central_txt),
                                       ("range label", range_txt), ("location label", loc_txt)))]
for name, a, others in _checks:
    for oname, o in others:
        if _box(a).overlaps(_box(o)):
            raise SystemExit("[ERROR] fig_crossover.py: the %s touches the %s" % (name, oname))
# the class label sits above the red line, never across it: the line falls to the right, so
# at the label's left end it must still be below the label's bottom
_bb = cls_txt.get_window_extent(_renderer)
_xl = ax.transData.inverted().transform((_bb.x0, _bb.y0))[0]
if ax.transData.transform((_xl, R_cls(_xl)))[1] > _bb.y0 - _PAD / 2:
    raise SystemExit("[ERROR] fig_crossover.py: the class label crosses the class line")

fig.savefig("fig_crossover.pdf", dpi=200, bbox_inches="tight")
fig.savefig("fig_crossover.png", dpi=150, bbox_inches="tight")

print("fig_crossover.py -- location against class, drawn from crossover.py's own functions")
print("  eps range [%.6f, %.6f], central %.6f" % (eps_lo, eps_hi, eps_mid))
print("  location factor (flat)     = %.6f" % R_loc)
print("  class factor at eps_low    = %.6f" % R_cls(eps_lo))
print("  crossing at eps*           = %.6f   (inside the range)" % eps_star)
print("  eps_dyn (AMOC, E_0)        = %.9f   (shaded below)" % eps_dyn)
print("Saved fig_crossover.pdf, fig_crossover.png")
