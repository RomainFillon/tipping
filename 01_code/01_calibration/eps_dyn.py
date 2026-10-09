# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   nothing on disk; a module imported by collect_computed.py and fig_crossover.py
  FEEDS      \\epsDyn*, \\amp*, \\classFactorAtEpsDynAMOC (through collect_computed.py), and
             the shaded region of fig:crossover (fig_crossover.py)
  INPUTS     for amoc_inputs(): the captures 02_output/values/raw_runs/ews_calibration.out
             (theta, the recovery rate) and calibration_ci.out (M*_P50), and
             inputs_literature.py (E_0, DT*, DT_0)
  SEED       none
  RUNTIME    under 1 s
  IMPLEMENTS the lower edge of the fixed-budget regime, in the appendix on where the elements sit

The lower edge eps_dyn is the proximity at which the budget, in relative terms, drains as
fast as the element recovers. Near a fold the recovery rate falls as the square root of the
remaining budget, theta(eps) = theta (eps/eps_now)^{1/2}, and the budget drains at the
relative rate E/(M* eps); equating the two gives

    eps_dyn = ( E eps_now^{1/2} / (M* theta) )^{2/3},

and the largest factor by which the geometry can lift the price between today and eps_dyn is
F = (eps_now/eps_dyn)^{1/2}, or 1 when eps_dyn >= eps_now.

The formula lives here once, so that the number in values.tex and the shaded region of
fig:crossover are two views of one evaluation; collect_computed.py checks the value the figure
prints against its own.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
RAW = os.path.join(ROOT, "02_output", "values", "raw_runs")
sys.path.insert(0, CODE)
from inputs_literature import DERIVATION_INPUTS, LITERATURE_MACROS  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_solvers"))
from price import fold_lead_price  # noqa: E402  (the one price and its leading law)


def eps_dyn(e_rate, mstar, theta, eps_now):
    """Lower edge of the fixed-budget regime (the appendix on where the elements sit)."""
    return ((e_rate / (mstar * theta)) * eps_now ** 0.5) ** (2.0 / 3.0)


def amp(eps_dyn_value, eps_now):
    """F, the factor by which the leading law lifts the price between eps_now and eps_dyn: the
    ratio of the fold's leading law at fixed state (price.py) at the two proximities, which is
    (eps_now/eps_dyn)^(1/2); 1 when eps_dyn >= eps_now."""
    if not eps_dyn_value < eps_now:
        return 1.0
    return float(fold_lead_price(1.0, 1.0, eps_dyn_value) / fold_lead_price(1.0, 1.0, eps_now))


def amoc_inputs():
    """(E_0, M*, theta, eps_now) of the AMOC, read as collect_computed.py reads them:
    theta at full precision from the ews capture, M*_P50 from the calibration capture,
    eps_now = (DT* - DT_0)/DT* at two decimals (\\epsAMOCnat)."""
    def _read(key):
        p = os.path.join(RAW, key + ".out")
        if not os.path.exists(p):
            sys.exit(f"[ERROR] eps_dyn.py: missing capture {p}; run run_all.sh "
                     "(the recovery rate needs the network step).")
        return open(p, encoding="utf-8", errors="replace").read()
    lam = re.search(r"\|lambda\|\s*\(full-sample\)\s*=\s*([\d.]+)", _read("ews_calibration"))
    ms = re.search(r"^\s*AMOC:\s*M\*_P50=(\d+)", _read("calibration_ci"), re.MULTILINE)
    if not (lam and ms):
        sys.exit("[ERROR] eps_dyn.py: could not parse theta or M*_P50 from the captures.")
    dts = float(LITERATURE_MACROS["DTstarAMOC"][0])
    eps_now = float(f"{(dts - DERIVATION_INPUTS['DT_current'][0]) / dts:.2f}")
    return float(LITERATURE_MACROS["EglobalRate"][0]), float(ms.group(1)), float(lam.group(1)), eps_now
