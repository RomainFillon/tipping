"""
REPLICATION HEADER
  PRODUCES   fig_bifurcation_intro.pdf
  FEEDS      fig:bifurcation_intro
  INPUTS     none (synthetic landscapes)
  SEED       none (deterministic)
  RUNTIME    4 to 10 s
  IMPLEMENTS the three canonical geometries of tab:geometries

fig_bifurcation_intro.py
1×3 figure: the three canonical tipping geometries (cleaned landscapes). The two
margin-law panels that once formed a right column are now the insets of
fig_intro_two_boundaries.py.

For economists: each landscape panel shows the "energy landscape" of the climate
system as the remaining carbon budget μ shrinks (light → dark curves).
The ● is the stable climate state; the red ■/○ is the tipping boundary.
"""

# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
FIGURE_MACROS = ()   # closed forms only, no number from values.tex

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

# ── Shared axis limits (same for all 3 panels) ───────────────────────────────
XLIM = [-2.2, 2.5]
YLIM = [-1.5, 1.6]   # Low enough for the stable state of panel C at mu = 2

# ── Colour palettes ───────────────────────────────────────────────────────────
PAL_DIV = ["#c6dbef", "#4393c3", "#08306b"]   # the fold, whose boundary moves (blue)
PAL_BND = ["#c7e9c0", "#41ab5d", "#00441b"]   # boundary fixed, bounded class (green): pitchfork, transcritical
C_STABLE = "#111111"   # stable state marker
C_TIP    = "#d73027"   # tipping boundary marker
C_MARGIN = "#d73027"   # safety-margin arrow (now red, matching tipping boundary)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=False, sharey=False)
ax_fold, ax_pf, ax_tc = axes


def draw_margin(ax, x1, x2, y, label):
    """Double-headed arrow for the safety margin (red)."""
    ax.annotate("", xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle="<->", color=C_MARGIN, lw=2.3))
    ax.text((x1 + x2) / 2, y + 0.13, label,
            ha="center", va="bottom", fontsize=13, color=C_MARGIN)


# ════════════════════════════════════════════════════════════════════════════
# (A) FOLD  —  U = X³/3 − μX,  stable at +√μ, tipping at −√μ
#             safety margin d = 2√μ  →  d ~ √μ  →  SCC diverges
# ════════════════════════════════════════════════════════════════════════════
ax = ax_fold
X = np.linspace(*XLIM, 700)
MU_A = [1.1, 0.45, 0.08]

for mu, col in zip(MU_A, PAL_DIV):
    U = X**3/3 - mu*X                      # Not clipped; leaves the frame
    ax.plot(X, U, color=col, lw=2.8)
    xs, xu = np.sqrt(mu), -np.sqrt(mu)
    ax.plot(xs, xs**3/3 - mu*xs, "o", color=C_STABLE, ms=10, zorder=5)
    ax.plot(xu, xu**3/3 - mu*xu, "s", color=C_TIP,    ms=10, zorder=5)

ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel(r"Climate state  $X$")
ax.set_ylabel(r"Potential  $U(X;\,\mu)$")
ax.set_title(r"(A)  Fold   —   $d(\mu)\propto\sqrt{\mu}$",
             loc="left", fontsize=14)


# ════════════════════════════════════════════════════════════════════════════
# (B) PITCHFORK  —  U = X⁴/4 − μX²/2,  stable at ±√μ, tipping at 0
#                  safety margin d = √μ, but the boundary 0 does not move  →  bounded
# ════════════════════════════════════════════════════════════════════════════
ax = ax_pf
MU_B = [1.2, 0.55, 0.08]

for mu, col in zip(MU_B, PAL_BND):
    U = X**4/4 - mu*X**2/2
    ax.plot(X, U, color=col, lw=2.8)
    xs = np.sqrt(mu)
    U_s = -mu**2/4
    ax.plot([ xs, -xs], [U_s, U_s], "o", color=C_STABLE, ms=10, zorder=5)
    ax.plot([0], [0], "s", color=C_TIP, ms=10, zorder=5)

ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel(r"Climate state  $X$")
ax.set_title(r"(B)  Pitchfork   —   $d(\mu)\propto\sqrt{\mu}$",
             loc="left", fontsize=14)


# ════════════════════════════════════════════════════════════════════════════
# (C) TRANSCRITICAL  —  U = X³/3 − μX²/2,  stable at μ, tipping at 0
#                       safety margin d = μ  →  SCC bounded
# ════════════════════════════════════════════════════════════════════════════
ax = ax_tc
MU_C = [2.0, 1.0, 0.30]

for mu, col in zip(MU_C, PAL_BND):
    U = X**3/3 - mu*X**2/2
    ax.plot(X, U, color=col, lw=2.8)
    U_s = -mu**3/6
    ax.plot([mu], [U_s], "o", color=C_STABLE, ms=10, zorder=5)
    ax.plot([0],  [0],   "s", color=C_TIP,    ms=10, zorder=5)

ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel(r"Climate state  $X$")
ax.set_title(r"(C)  Transcritical   —   $d(\mu)\propto\mu$",
             loc="left", fontsize=14)


# The right column (two margin-law panels, Class A d ~ sqrt(mu) and Class B d ~ mu) was
# removed: it duplicated the insets of fig_intro_two_boundaries.py, the
# introduction's figure, which now carries the two margin laws and the price each gives.


# ── Shared legend ─────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=C_STABLE, ms=11,
           label="Stable state"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=C_TIP, ms=11,
           label="Tipping boundary"),
    Line2D([0],[0], color=PAL_DIV[0], lw=3.5, label=r"Budget $\mu$ large $\to$ small"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3,
           fontsize=11.5, frameon=True, framealpha=0.95,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1], h_pad=2.5, w_pad=2.0)
plt.savefig("fig_bifurcation_intro.pdf", dpi=200, bbox_inches="tight")
plt.savefig("fig_bifurcation_intro.png", dpi=150, bbox_inches="tight")
print("Saved.")
