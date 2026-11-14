#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   fig_hjb_stop.pdf, fig_hjb_stop.png
  FEEDS      none yet (step 54, F2: the section on the planner's problem is not written)
  INPUTS     02_output/planner/hjb_stop.json, written by 02_solvers/hjb_stop.py (which reads the AMOC
             series through ews_calibration.py)
  SEED       none (deterministic)
  RUNTIME    under 5 s
  IMPLEMENTS no equation: it plots the solution of the full control problem hjb_stop.py computed

fig_hjb_stop.py -- the full control problem for the fold, at eta*_now = 2 and eta*_now = F.

TOP     the stopping curve eps_bar(x): stop when the budget is at or below eps_bar(x) with the state at x
        (state in widths l of the noise layer). The branch x = x*(eps) = eps^(1/2) is dashed; it meets
        each curve at the stop point of the branch, (x_b, eps_b), marked.
BOTTOM  the price -dc/dM at x*(eps) along the branch, in units of rho L/E_0: controlled (one curve per g),
        against the frozen price (the budget does not move) and the price under the flow (emit for ever).
        The dash-dotted line is the edge of the noise layer, eta*(eps) = 1 (eps = l^2); it is eps_dyn on the
        panel F only (l^2 = eps_now/F^2), and lies above eps_dyn at eta*_now = 2.
        The stop point is marked twice: a filled dot at the controlled price there, read from the emitting
        side, and an open ring at the abatement cost g. The rule dc/dM = gamma makes the two coincide; the
        largest gap over the cases is \hjbStopPriceMatchPct percent.

NOTHING IS RECOMPUTED HERE. Every curve and point is read from hjb_stop.json, written by the run that
printed the \hjb* macros; the gap printed in the figure is recomputed from the saved stop points and
checked against the macro string, rather than assumed.

Usage:  python 01_code/03_figures/fig_hjb_stop.py    (writes into the current directory)
"""
# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
FIGURE_MACROS = (
    "hjbStopPriceMatchPct",
)

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "02_solvers"))
import moving_budget as mb  # noqa: E402  (fmt_sig, the formatting of the macro)

DATA = CODE.parent / "02_output" / "planner" / "hjb_stop.json"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

COLS = {0.02: "#1f4e79", 0.1: "#c0504d", 0.5: "#6a9a2a"}   # one colour per g, as fig_pas3_hjb (step 53)
C_GREY, C_TXT = "#555555", "#111111"

if not DATA.exists():
    raise SystemExit("[ERROR] fig_hjb_stop.py: %s is missing; run 02_solvers/hjb_stop.py first" % DATA)
D = json.loads(DATA.read_text(encoding="utf-8"))
e_now, e_dyn, F = D["eps_now"], D["eps_dyn"], D["F"]
cases = D["cases"]
flows = {f["eta"]: f for f in D["flows"]}
etas = sorted(flows)

# the gap at the stop point, recomputed from the saved points and checked against the macro
stops = [c for c in cases if c["eps_b"] > 0]
gap = max(abs(c["P_stop"] / c["g"] - 1) for c in stops)
lF = [c["l"] for c in cases if c["eta"] == F][0]
if abs(lF ** 2 / e_dyn - 1) > 1e-9:
    raise SystemExit("[ERROR] fig_hjb_stop.py: on the panel F, l^2 = %.6g is not eps_dyn = %.6g" % (lF ** 2, e_dyn))
if mb.fmt_sig(100 * gap) != D["macros"]["hjbStopPriceMatchPct"]:
    raise SystemExit("[ERROR] fig_hjb_stop.py: largest |P/g - 1| = %.4g%%, hjbStopPriceMatchPct = %s"
                     % (100 * gap, D["macros"]["hjbStopPriceMatchPct"]))

fig, axs = plt.subplots(2, len(etas), figsize=(10.0, 7.6), squeeze=False)
for i, eta in enumerate(etas):
    ax, bx = axs[0, i], axs[1, i]
    lab = r"$F$" if eta == F else f"{eta:g}"
    mine = [c for c in cases if c["eta"] == eta]
    l = mine[0]["l"]
    for c in mine:
        col = COLS[c["g"]]
        X, ebx = np.array(c["X"]), np.array(c["ebx"])
        m = ebx > 0
        ax.semilogy(X[m] / l, ebx[m], color=col, lw=1.7, label=f"$g={c['g']:g}$")
        if c["eps_b"] > 0:
            ax.plot(np.sqrt(c["eps_b"]) / l, c["eps_b"], "o", color=col, ms=5.5, mec="white", mew=0.8, zorder=5)
        bx.loglog(c["EP"], c["Pc"], color=col, lw=1.7)
        bx.axhline(c["g"], color=col, lw=0.6, ls=":")
        if c["eps_b"] > 0:
            bx.plot(c["eps_b"], c["P_stop"], "o", color=col, ms=5.5, zorder=6)
            bx.plot(c["eps_b"], c["g"], "o", mfc="none", mec=C_TXT, ms=10, mew=1.0, zorder=7)
            bx.annotate(f"$g={c['g']:g}$", (c["eps_b"], c["g"]), xytext=(-8, 7), textcoords="offset points",
                        ha="right", va="bottom", fontsize=9, color=col)
    xx = np.linspace(1e-4, np.sqrt(e_now), 300)
    ax.semilogy(xx / l, xx ** 2, color=C_GREY, ls="--", lw=1.0, label=r"branch $x=x^*(\varepsilon)$")
    edge = l ** 2                      # the edge of the noise layer, eta*(eps) = 1; eps_dyn on the panel F only
    edge_lab = r"$\eta^*=1$, $\varepsilon=\varepsilon_{dyn}$" if eta == F else r"$\eta^*=1$"
    ax.axhline(edge, color=C_GREY, ls="-.", lw=0.8)
    ax.text(0.01, edge, edge_lab, transform=ax.get_yaxis_transform(), ha="left", va="top", fontsize=9,
            color=C_GREY)
    ax.set_xlabel(r"state $x/\ell$")
    ax.set_ylabel(r"stopping curve $\bar\varepsilon(x)$")
    ax.set_title(rf"fold, $\eta^*_{{now}}={lab.strip('$')}$", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    fl = flows[eta]
    bx.loglog(fl["EP"], fl["Pf"], color=C_TXT, lw=1.0, ls="--")
    bx.loglog(fl["EP"], np.maximum(fl["Pa"], 1e-300), color=C_GREY, lw=1.0, ls="-.")
    bx.axvline(edge, color=C_GREY, ls="-.", lw=0.8)
    bx.text(edge, 0.45, edge_lab + " ", transform=bx.get_xaxis_transform(), ha="right", va="center", fontsize=9,
            color=C_GREY)
    bx.set_ylim(1e-5, 1.5)
    bx.set_xlabel(r"budget $\varepsilon$ along the branch")
    bx.set_ylabel(r"price at $x^*(\varepsilon)$, units of $\rho L/E_0$")
    bx.legend(handles=[Line2D([], [], color=C_GREY, lw=1.7, label="controlled (colour: $g$)"),
                       Line2D([], [], color=C_TXT, lw=1.0, ls="--", label="frozen budget"),
                       Line2D([], [], color=C_GREY, lw=1.0, ls="-.", label="under the flow (emit for ever)"),
                       Line2D([], [], ls="none", marker="o", color=C_GREY, ms=5.5, label="controlled price at the stop"),
                       Line2D([], [], ls="none", marker="o", mfc="none", mec=C_TXT, ms=10, label="the cost $g$")],
              fontsize=8, loc="lower right", frameon=False)
fig.text(0.5, -0.005, r"At the stop point the controlled price equals $g$ to within "
         + D["macros"]["hjbStopPriceMatchPct"] + r"$\,$%", ha="center", va="top", fontsize=10)
fig.tight_layout()

fig.savefig("fig_hjb_stop.pdf", dpi=200, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig("fig_hjb_stop.png", dpi=150, bbox_inches="tight")
print("fig_hjb_stop: stop points (eta*_now, g, eps_b, P, |P/g - 1|): " + "; ".join(
    f"({c['eta']:.3g}, {c['g']:g}, {c['eps_b']:.4g}, {c['P_stop']:.5g}, {100 * abs(c['P_stop'] / c['g'] - 1):.3f}%)"
    for c in stops) + f"; largest {100 * gap:.3f}% (hjbStopPriceMatchPct = {D['macros']['hjbStopPriceMatchPct']})")
