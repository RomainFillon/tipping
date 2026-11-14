#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   fig_ceiling_map.pdf, fig_ceiling_map.png
  FEEDS      none yet (step 54, F1: the section on the planner's problem is not written)
  INPUTS     02_output/planner/fixed_ceiling.json, written by 02_solvers/fixed_ceiling.py (which reads
             the AMOC series through ews_calibration.py); imports fixed_ceiling.analyse and eta_at
  SEED       none (deterministic)
  RUNTIME    under 10 s
  IMPLEMENTS no equation: it classifies the best fixed ceiling fixed_ceiling.py computed

fig_ceiling_map.py -- where the best fixed ceiling lies, on the plane (g, eta*_now).

Each cell is one (g, eta*_now) and one of three outcomes of the best fixed ceiling eps_bar*:
    stop today            eps_bar* = eps_now;
    interior ceiling      stop more than 5 years before t_c;
    exhaustion            stop within 5 years of t_c, or never stop.
The two regimes of eta*_now are two panels: outside the noise layer today (eta*_now >= 1.25, F the largest
margin, F = (eps_now/eps_dyn)^(1/2)) and inside it (eta*_now <= 1), where on the six g of the sweep the
choice is binary, stop today or never.

THE EDGE OF THE LAYER. Outside the layer a line marks where the best ceiling sits at the edge of the noise
layer of its row, eta*(eps_bar*) = 1, i.e. eps_bar* = l^2: to its left the planner stops before the budget
enters the layer, to its right inside it. eps_dyn is a constant of the calibration while l varies with
eta*_now; the edge is eps_dyn only on the row F (l^2 = eps_now/F^2 = eps_dyn, checked below), the one row
where eps_dyn is marked: for eta*_now < F, eps_dyn lies inside the noise layer.

BETWEEN THE SWEEP POINTS. Inside the layer the choice is not exactly binary on a continuum of g: at
eta*_now = 1, a band of about 2% in g around g_switch has an early interior ceiling (a stop within a decade),
worth under 0.1% of c. The fine axis is densified around g_switch so that the map draws that band as it is
(a sliver at this scale).

NOTHING IS RECOMPUTED HERE. c(eps_bar) = g U1 + (1 - g) U2 is linear in g, and fixed_ceiling.py saved U1
and U2 on the doubled grids; this script evaluates fixed_ceiling.analyse on a fine g axis. On the six g of
the sweep the classification is the one fixed_ceiling.py made: the interior count and the switch at
eta*_now = 1 are checked against the macro strings it wrote, rather than assumed.

Usage:  python 01_code/03_figures/fig_ceiling_map.py    (writes into the current directory)
"""
# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
FIGURE_MACROS = (
    "ceilInteriorCount",
    "ceilGSwitchEtaOne",
)

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE / "02_solvers"))
import fixed_ceiling as fc  # noqa: E402

DATA = CODE.parent / "02_output" / "planner" / "fixed_ceiling.json"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

C_TODAY, C_INTER, C_EXH = "#1f4e79", "#9ecae1", "#d9d9d9"   # the map of step 53 (fig_pas2bis_map)
C_EDGE = "#d73027"                                           # markers, red-orange, as fig_record_power.py
CATS = {"< 5 yr": 0, "interior > 5 yr": 1, "today": 2}

if not DATA.exists():
    raise SystemExit("[ERROR] fig_ceiling_map.py: %s is missing; run 02_solvers/fixed_ceiling.py first" % DATA)
D = json.loads(DATA.read_text(encoding="utf-8"))
Y, GAP, e_dyn, F = D["years_per_eps"], D["YEARS_GAP"], D["eps_dyn"], D["F"]
R = {c["eta"]: dict(ebar=np.array(c["ebar"]), U1=np.array(c["U1"]), U2=np.array(c["U2"]), never=c["never"])
     for c in D["cases"]}
ELL = {c["eta"]: c["l"] for c in D["cases"]}
ETAS_IN, ETAS_OUT, GS = D["ETAS_IN"], D["ETAS_OUT"], D["GS"]
if abs(ELL[F] ** 2 / e_dyn - 1) > 1e-9:
    raise SystemExit("[ERROR] fig_ceiling_map.py: on the row F the edge of the layer l^2 = %.6g is not "
                     "eps_dyn = %.6g" % (ELL[F] ** 2, e_dyn))


def category(a):
    """fixed_ceiling.category, with the years per unit of eps of the run that wrote the data."""
    if a["at_now"]:
        return "today"
    return "interior > 5 yr" if a["e_star"] * Y > GAP else "< 5 yr"


# ── the sweep: the classification fixed_ceiling.py made, checked against its macros ──
n_inter = sum(1 for eta in ETAS_OUT for g in GS if category(fc.analyse(R[eta], g)) == "interior > 5 yr")
if str(n_inter) != D["macros"]["ceilInteriorCount"]:
    raise SystemExit("[ERROR] fig_ceiling_map.py: %d interior ceilings on the sweep, ceilInteriorCount = %s"
                     % (n_inter, D["macros"]["ceilInteriorCount"]))
n_in = sum(1 for eta in ETAS_IN for g in GS if category(fc.analyse(R[eta], g)) == "interior > 5 yr")
if n_in:
    raise SystemExit("[ERROR] fig_ceiling_map.py: %d interior ceilings inside the layer on the sweep" % n_in)
gsw = D["g_switch"]["1"]
if fc.fmt_sig(gsw) != D["macros"]["ceilGSwitchEtaOne"]:
    raise SystemExit("[ERROR] fig_ceiling_map.py: g_switch %.4g against ceilGSwitchEtaOne = %s"
                     % (gsw, D["macros"]["ceilGSwitchEtaOne"]))
a_sw = fc.analyse(R[1.0], gsw)      # g_switch: "stop today" and "never stop" cost the same
if abs(a_sw["cnow"] - a_sw["c0"]) > 1e-9 * a_sw["c0"]:
    raise SystemExit("[ERROR] fig_ceiling_map.py: at g_switch, c(today) = %.9g and c(never) = %.9g differ"
                     % (a_sw["cnow"], a_sw["c0"]))

# ── the map: a fine g axis, geometric, through the six g of the sweep, dense around g_switch ──
G = np.unique(np.concatenate([np.geomspace(0.004, 1.0, 141), GS, np.geomspace(0.97 * gsw, 1.03 * gsw, 61)]))
edges = np.concatenate([[G[0] ** 1.5 / G[1] ** 0.5], np.sqrt(G[1:] * G[:-1]), [G[-1] ** 1.5 / G[-2] ** 0.5]])


def grid(etas):
    return np.array([[CATS[category(fc.analyse(R[eta], g))] for g in G] for eta in etas])


def g_at_edge(eta):
    """The g at which the best ceiling sits at the edge of the noise layer, eta*(eps_bar*) = 1 (eps_bar*
    decreases in g): bisection in log g on the interval of the fine axis where eta*(eps_bar*) - 1 changes
    sign among interior ceilings; None if there is no such interval."""
    def f(g):
        a = fc.analyse(R[eta], g)
        return fc.eta_at("fold", {"l": ELL[eta]}, a["e_star"]) - 1 if category(a) == "interior > 5 yr" else None
    v = [f(g) for g in G]
    for k in range(len(G) - 1):
        if v[k] is not None and v[k + 1] is not None and v[k] > 0 >= v[k + 1]:
            lo, hi = np.log(G[k]), np.log(G[k + 1])
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fm = f(np.exp(mid))
                if fm is None:
                    break
                lo, hi = (mid, hi) if fm > 0 else (lo, mid)
            return float(np.exp(0.5 * (lo + hi)))
    return None


cmap = ListedColormap([C_EXH, C_INTER, C_TODAY])
fig, (ax_out, ax_in) = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True,
                                    gridspec_kw={"height_ratios": [len(ETAS_OUT), len(ETAS_IN)], "hspace": 0.32})
for ax, etas, title in ((ax_out, ETAS_OUT, r"outside the noise layer today ($\eta^*_{now}\geq 1.25$)"),
                        (ax_in, ETAS_IN, r"inside the noise layer today ($\eta^*_{now}\leq 1$)")):
    ax.pcolormesh(edges, np.arange(len(etas) + 1), grid(etas), cmap=cmap, vmin=-0.5, vmax=2.5, shading="flat")
    ax.set_xscale("log")
    ax.set_yticks(np.arange(len(etas)) + 0.5, [r"$F$" if eta == F else f"{eta:g}" for eta in etas])
    for g in GS:
        ax.axvline(g, color="k", lw=0.5, ls=":", alpha=0.7)
    ax.set_ylabel(r"$\eta^*_{now}$")
    ax.set_title(title, loc="left", fontsize=10)
    ax.tick_params(axis="y", length=0)

# the edge of the layer: where the best ceiling sits at eta*(eps_bar*) = 1 (outside the layer only)
pts = []
for i, eta in enumerate(ETAS_OUT):
    g = g_at_edge(eta)
    print(f"eta*(eps_bar*) = 1: eta*_now = {eta:.4g}, " + ("no crossing on the axis" if g is None else f"g = {g:.4g}"))
    if g is not None:
        pts.append((g, i + 0.5, eta))
if pts:
    ax_out.plot([p[0] for p in pts], [p[1] for p in pts], color=C_EDGE, lw=1.3, marker="o", ms=4.5,
                mfc="white", mew=1.3, zorder=5)
    for g, y, eta in pts:
        if eta == F:
            ax_out.annotate(r"$\bar\varepsilon^*=\varepsilon_{dyn}$", (g, y), xytext=(-7, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=9, color=C_EDGE)

ax_in.set_xlabel(r"$g=\gamma E_0/(\rho L)$, the abatement cost relative to the loss (dotted: the sweep)")
ax_in.set_xlim(edges[0], edges[-1])
handles = [Patch(color=C_TODAY, label="stop today"),
           Patch(color=C_INTER, label=f"interior ceiling, more than {GAP:g} years before $t_c$"),
           Patch(color=C_EXH, label=f"exhaustion: within {GAP:g} years of $t_c$, or never stop"),
           Line2D([], [], color=C_EDGE, lw=1.3, marker="o", ms=4.5, mfc="white", mew=1.3,
                  label=r"ceiling at the edge of the layer, $\eta^*(\bar\varepsilon^*)=1$")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False, fontsize=9)

fig.savefig("fig_ceiling_map.pdf", dpi=200, bbox_inches="tight", metadata={"CreationDate": None})
fig.savefig("fig_ceiling_map.png", dpi=150, bbox_inches="tight")
print(f"fig_ceiling_map: {n_inter} interior ceilings on the sweep (ceilInteriorCount), switch at eta*_now = 1: "
      f"g = {gsw:.4g} (ceilGSwitchEtaOne = {D['macros']['ceilGSwitchEtaOne']}); fine axis {len(G)} values of g")
