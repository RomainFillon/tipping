#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   fig_intro_two_boundaries.pdf, fig_intro_two_boundaries.png
  FEEDS      fig:two_boundaries (the introduction)
  INPUTS     none: closed forms only
  SEED       none (deterministic)
  RUNTIME    under 5 s
  IMPLEMENTS the two margin laws of tab:geometries, d = mu (fixed boundary) and
             d = 2 sqrt(mu) (fold), and the price exponents they give, eps^0 and eps^{-1/2}

fig_intro_two_boundaries.py -- two ways a safety margin closes.

Left, a fixed boundary: the stable state walks to a boundary that stands still, so the
margin is linear in the budget left and the price of restraint stays bounded. Right, a fold
x' = mu - x^2 with mu = 1 - (budget spent): the stable state +sqrt(mu) and the boundary
-sqrt(mu) meet at mu = 0, the boundary moves as sqrt(mu), and the price rises as
(budget left)^{-1/2} until the budget drains faster than the system recovers (the
shaded end of the lower panel, where the price leaves the power law and flattens; drawn, not
computed). Both lower panels share the upper panels' axis, budget spent.
Nothing is calibrated; the axes carry no units. The same colours as fig_bifurcation_intro.py:
the fold in blue, the bounded class in green, the boundary in red-orange.

Usage:  python 01_code/03_figures/fig_intro_two_boundaries.py   (writes into the cwd)
"""
# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
FIGURE_MACROS = ()   # closed forms only, no number from values.tex

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
})

# same palette as fig_bifurcation_intro.py
C_FOLD, F_FOLD = "#2b7bba", "#c6dbef"   # fold: blue line, light-blue fill
C_BND, F_BND = "#41ab5d", "#c7e9c0"     # bounded class: green line, light-green fill
C_TIP = "#d73027"                        # tipping boundary
C_TXT, C_GREY = "#111111", "#555555"
LW = 1.5

YLIM = (-1.15, 1.25)          # common to both panels
s = np.linspace(0.0, 1.0, 400)   # budget spent
X_MID = 0.5                   # where the margin is drawn

fig = plt.figure(figsize=(8.2, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.15], hspace=0.72, wspace=0.16,
                      left=0.05, right=0.98, top=0.93, bottom=0.08)
axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])


def frame(ax, title, note):
    ax.set_xlim(0, 1.04)
    ax.set_ylim(*YLIM)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", color=C_TXT)
    ax.set_xlabel(r"budget spent $\rightarrow$", color=C_TXT)
    ax.set_ylabel("state of the system", color=C_TXT)
    ax.text(0.0, -0.17, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color=C_GREY)


def margin(ax, y_lo, y_hi):
    ax.annotate("", xy=(X_MID, y_lo), xytext=(X_MID, y_hi),
                arrowprops=dict(arrowstyle="<->", color=C_TXT, lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(X_MID - 0.02, 0.5 * (y_lo + y_hi), "safety margin", ha="right", va="center",
            fontsize=9.5, color=C_TXT)


# ── A. fixed boundary: state 1 - s, boundary 0 ───────────────────────────────
stable_A = 1.0 - s
axA.fill_between(s, 0.0, stable_A, color=F_BND, alpha=0.55, lw=0)
axA.plot(s, stable_A, color=C_BND, lw=LW, label="stable state")
axA.plot(s, np.zeros_like(s), color=C_TIP, lw=LW, label="tipping boundary")
margin(axA, 0.0, 1.0 - X_MID)
frame(axA, "A. Fixed boundary",
      "the state walks to a boundary that stands still")
axA.legend(frameon=False, loc="upper right", fontsize=9)

# ── B. fold: mu = 1 - s, stable +sqrt(mu), boundary -sqrt(mu) ─────────────────
mu = 1.0 - s
stable_B, bound_B = np.sqrt(mu), -np.sqrt(mu)
axB.fill_between(s, bound_B, stable_B, color=F_FOLD, alpha=0.55, lw=0)
axB.plot(s, stable_B, color=C_FOLD, lw=LW, label="stable state")
axB.plot(s, bound_B, color=C_TIP, lw=LW, label="tipping boundary")
margin(axB, -np.sqrt(1 - X_MID), np.sqrt(1 - X_MID))
frame(axB, "B. Fold (overturning circulation, Amazon,\nWest Antarctic ice sheet)",
      "the boundary comes to meet the state")
axB.legend(frameon=False, loc="upper right", fontsize=9)


# ── insets: price of restraint against the budget spent (same axis as above) ──
def inset(cell, y, colour, label, xy, shade=None):
    ax = fig.add_subplot(cell)
    # a small framed box, centred under its panel
    pos = ax.get_position()
    w = pos.width * 0.6
    ax.set_position([pos.x0 + 0.5 * (pos.width - w), pos.y0, w, pos.height])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6); sp.set_color(C_GREY)
    if shade is not None:
        ax.axvspan(shade, 1.0, color="#e0e0e0", lw=0)
        # the legend of the shaded zone sits outside the box, to its right, with an arrow
        ax.annotate("budget drains\nfaster than the\nsystem recovers",
                    xy=(0.5 * (shade + 1.0), 0.25 * Y_TOP), xycoords="data",
                    xytext=(1.06, 0.5), textcoords="axes fraction",
                    ha="left", va="center", fontsize=8.5, color=C_GREY,
                    arrowprops=dict(arrowstyle="-", color=C_GREY, lw=0.6))
    ax.plot(SPENT, y, color=colour, lw=LW)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, Y_TOP)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(r"budget spent $\rightarrow$", fontsize=9, color=C_TXT, labelpad=2)
    ax.set_ylabel("price of restraint", fontsize=9, color=C_TXT, labelpad=2)
    ax.text(*xy, label, transform=ax.transAxes, fontsize=9.5, color=C_TXT,
            ha="left", va="center")


SPENT = np.linspace(0.0, 1.0, 400)
Y_TOP = 2.6
S_DYN = 0.85                     # illustrative end of the law, not a calibrated eps_dyn
TAU = 0.04                       # how fast the drawn price flattens past it
L_NOISE = 0.55                   # illustrative noise layer, in units of the initial margin


def fold_price(b):
    """Negligible while the margin sqrt(b) is large against the noise layer, then the
    inverse square root of the budget left b. Drawn, not computed."""
    return b ** -0.5 * np.exp(-(np.sqrt(b) / L_NOISE) ** 3)


b_left = 1.0 - np.minimum(SPENT, S_DYN)
y_law = fold_price(b_left)
y_d = float(fold_price(1.0 - S_DYN))
h = 1e-5
slope_d = float((fold_price(1.0 - S_DYN - h) - fold_price(1.0 - S_DYN + h)) / (2 * h))
past = np.clip(SPENT - S_DYN, 0.0, None)
y_fold = np.where(SPENT <= S_DYN, y_law, y_d + slope_d * TAU * (1.0 - np.exp(-past / TAU)))
# The boundary stands still; the price stays low and falls to zero near the boundary.
y_bnd = 0.6 * np.sqrt(1.0 - SPENT)
inset(gs[1, 0], y_bnd, C_BND, "bounded", (0.40, 0.60))
inset(gs[1, 1], y_fold, C_FOLD, r"rises as (budget left)$^{-1/2}$", (0.04, 0.86), shade=S_DYN)

fig.savefig("fig_intro_two_boundaries.pdf", dpi=200, bbox_inches="tight")
fig.savefig("fig_intro_two_boundaries.png", dpi=150, bbox_inches="tight")
print("saved fig_intro_two_boundaries.pdf/.png")
