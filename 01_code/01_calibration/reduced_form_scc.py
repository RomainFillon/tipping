#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   the MACROS block parsed into computed.json
  FEEDS      \sccReducedCoefAMOC \reducedExponentAnchor \dphysBreakevenIAM \dphysValidityMax and the eta* family
  INPUTS     the cached ews_calibration run (lambda, Var); inputs_literature.py
  SEED       none of its own; inherits the seeded EWS estimates
  RUNTIME    14 to 20 s
  IMPLEMENTS the reduced-form price (not reported in the paper) and the price family it indexes

reduced_form_scc.py — the tipping SCC in reduced form (not reported in the paper).

WHY THIS SCRIPT EXISTS
----------------------
calibration_ci.py evaluates

    SCC = (L/GDP) x (GDP/E0) x Phi(rho_tilde) / (sigma^(2/3) sqrt(M*)) x eps^(-1/2)

with sigma entered in the element's own state-variable units (K for the AMOC) and M* in
GtCO2. Those are not the units of the normal form f = mu - X^2, in which mu carries the
dimensions of an inverse squared time; a carbon budget in GtCO2 does not. The pairing is
therefore dimensionally inconsistent and the LEVEL it produces is not defensible (the
exponent, the noise sign, the cross-partial and the eps-ratios are unaffected: they are
ratios in eps and sigma).

Restoring the missing conversion removes parameters instead of adding them. Identifying
the observed restoring rate with the linearised rate of the normal form, lambda = 2 sqrt(mu)
(a structural assumption of an earlier AMOC appendix, not reported in the paper), the budget, the noise and the margin
stop entering separately and the price collapses to

    SCC_tip = 2 * L * Omega(rho T) * Phi(rho_tilde) * eta_star / lambda^2                (A)
    (the leading law at fixed state; it read 4 along the stable state)
    eta_star   = 0.5 * ( d_phys^2 / (2 Var) )^(1/3)                                  (B)
    rho_tilde  = 2 rho eta_star / lambda                                             (C)

Two element-specific numbers remain: the restoring rate lambda read off the record, and
the physical margin d_phys expressed in units of the observed variability through (B).
d_phys is a unit conversion, NOT a proximity: at fixed (lambda, Var) a larger margin means
the same fluctuations span a smaller fraction of the distance to the edge, i.e. a quieter
system in model units, hence -- by dSCC/dsigma < 0 -- a HIGHER price. Since eta_star ~
d_phys^(2/3) and Phi(x) ~ Phi'(0) x at small x, (A) grows as d_phys^(4/3) in the
small-rho_tilde regime. The script reports the LOCAL exponent so that the 4/3 written in
the paper is checked rather than asserted.

INPUTS: lambda and Var are parsed from the cached ews_calibration.py run (the same source
collect_computed.py uses), so the chain HadISST -> ews_calibration.py -> here is explicit.
Nothing is transcribed by hand.

Usage:  python reduced_form_scc.py
"""
import os
import re
import sys
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_solvers"))
from price import fold_lead_price  # noqa: E402  (the one price and its leading law)
sys.path.insert(0, CODE)
from phi_fold import phi_true_scalar, realization_factor, self_check  # noqa: E402
from inputs_literature import LITERATURE_MACROS, DERIVATION_INPUTS  # noqa: E402

RAW = os.path.join(ROOT, "02_output", "values", "raw_runs", "ews_calibration.out")

# ── sourced constants (same values as calibration_ci.py / inputs_literature.py) ──
rho = 0.03            # discount rate (modelling choice, rhoBase)
T_AMOC = 50.0         # yr, realization timescale (Armstrong McKay et al. 2022, TAMOC)
GDP = 100e12          # $ world annual GDP
# E0 imported (EglobalRate, GtCO2/yr), not retyped.
E0 = float(LITERATURE_MACROS["EglobalRate"][0]) * 1e9      # tCO2/yr annual emissions
gdp_per_ton = GDP / E0                      # $/tCO2 ~ 2500 (used only by an earlier version of the level)

# The budget today in tonnes, at central inputs, as the rest of the AMOC-price appendix:
# mu_now = M* eps_now = (DT* - DT0) x 1000 / TCRE GtCO2.
DT_STAR = float(DERIVATION_INPUTS["DTstar_AMOC_best"][0])  # degC, central AMOC threshold
DT_NOW = float(DERIVATION_INPUTS["DT_current"][0])         # degC, current warming
TCRE = float(DERIVATION_INPUTS["TCRE"][0])                 # degC per 1000 GtCO2, central
MSTAR_T = DT_STAR * 1000.0 / TCRE * 1e9                    # tCO2, the budget to the threshold
EPS_NOW = (DT_STAR - DT_NOW) / DT_STAR                     # current proximity
MU_NOW_T = MSTAR_T * EPS_NOW                               # tCO2, budget left today
L_LO = 0.0                                  # L/GDP uniform robustness sweep, floored at 0
L_HI = float(LITERATURE_MACROS["Lmax"][0])  # Lmax, sourced once in inputs_literature.py
L_MED = 0.5 * (L_LO + L_HI)                 # median of the damage prior


def read_ews():
    """lambda (yr^-1) and detrended Var (K^2) from the cached ews_calibration.py run."""
    if not os.path.exists(RAW):
        sys.exit(f"[ERROR] missing {RAW} — run ews_calibration.py first (needs network).")
    txt = open(RAW, encoding="utf-8", errors="replace").read()
    lam = re.search(r"\|lambda\|\s*\(full-sample\)\s*=\s*([\d.]+)", txt)
    var = re.search(r"Var\(X_t\) detrended\s*=\s*([\d.]+)", txt)
    if not (lam and var):
        sys.exit("[ERROR] could not parse lambda / Var from the cached ews run.")
    return float(lam.group(1)), float(var.group(1))


def eta_star(d_phys, var):
    """(B) margin in units of the observed variability; dimensionless."""
    return 0.5 * (d_phys ** 2 / (2.0 * var)) ** (1.0 / 3.0)


def scc_reduced_pre35(d_phys, lam, var, L_frac=L_MED, T=T_AMOC):
    """COMPARISON ONLY: the earlier level, L (L/GDP x GDP/E0, $/tCO2) times the leading
    law in model units (yr^2): its result carries yr^2 too, since nothing converted the model's
    budget (yr^-2) into tonnes."""
    eta = eta_star(d_phys, var)
    rt = 2.0 * rho * eta / lam
    Phi = phi_true_scalar(rt)
    mu = lam ** 2 / 4.0
    return L_frac * gdp_per_ton * float(realization_factor(rho, T)) * fold_lead_price(Phi, np.sqrt(mu) / eta, mu)


def scc_reduced(d_phys, lam, var, L_frac=L_MED, T=T_AMOC):
    """(A) SCC in $/tCO2 at margin d_phys (K), at the median of the damage prior.

    The units made explicit. At the normal form theta = 2 sqrt(mu_model), so the fold
    parameter today is theta^2/4 (yr^-2); by (H1) the budget moves it in proportion,
    mu_model = (theta^2/4) mu/mu_now with mu, mu_now in tCO2, so one tonne lowers it by
    theta^2/(4 mu_now). The price per tonne is the loss in dollars (L/GDP x GDP, with the
    realization factor) times |d phi/d mu_model| (yr^2) times theta^2/(4 mu_now) (yr^-2/tCO2):
    dollars per tonne."""
    eta = eta_star(d_phys, var)
    rt = 2.0 * rho * eta / lam
    Phi = phi_true_scalar(rt)
    L_dollars = L_frac * GDP                 # $ (L/GDP is a multiple of annual GDP)
    f = float(realization_factor(rho, T))
    # The leading law at FIXED state, Phi/(2 l sqrt(mu)) with mu = lambda^2/4 and
    # l = sqrt(mu)/eta*, i.e. 2 Phi eta*/lambda^2 (it was 4 Phi eta*/lambda^2 along x*).
    mu = lam ** 2 / 4.0                      # yr^-2, the fold parameter today
    per_tonne = mu / MU_NOW_T                # yr^-2 per tCO2: theta^2/(4 mu_now)
    return L_dollars * f * fold_lead_price(Phi, np.sqrt(mu) / eta, mu) * per_tonne


def local_exponent(d, lam, var, h=1e-3):
    """dlog SCC / dlog d_phys at d (should approach 4/3 as rho_tilde -> 0)."""
    a = np.log(scc_reduced(d * (1 - h), lam, var))
    b = np.log(scc_reduced(d * (1 + h), lam, var))
    return (b - a) / (np.log(d * (1 + h)) - np.log(d * (1 - h)))


def breakeven(target, lam, var):
    """margin d_phys (K) at which the reduced-form level equals `target` $/tCO2; nan when the
    level stays below the target over (1e-5, 50) K (the converted level is small)."""
    g = lambda d: scc_reduced(d, lam, var) - target
    if g(1e-5) * g(50.0) > 0:
        return float("nan")
    return brentq(g, 1e-5, 50.0, xtol=1e-10)


def main():
    lam, var = read_ews()
    sd = np.sqrt(var)
    f = float(realization_factor(rho, T_AMOC))
    p0, dp0, dp0_exact = self_check()

    print("=" * 72)
    print("REDUCED-FORM TIPPING SCC (AMOC)")
    print("=" * 72)
    print(f"  lambda (full-sample, HadISST) = {lam:.4f} yr^-1   [ews_calibration.py]")
    print(f"  Var detrended                 = {var:.4f} K^2     [ews_calibration.py]")
    print(f"  sd of the record              = {sd:.4f} K")
    print(f"  rho = {rho}  |  T = {T_AMOC:.0f} yr  |  f(rhoT) = {f:.4f}")
    print(f"  L/GDP median = {L_MED:.3f} (uniform sweep on [{L_LO},{L_HI}])"
          f"  |  GDP/E0 = ${gdp_per_ton:.0f}/tCO2")
    print(f"  Phi self-check: Phi(0)={p0:.2e} (exact 0) ; "
          f"Phi'(0)={dp0:.5f} vs exact {dp0_exact:.5f}")
    print()

    # ── the coefficient of the price family ──
    # The family is INDEXED BY THE RECORD'S STANDARD DEVIATION, not by the kelvin, because
    # every quantity in its neighbourhood is: the validity window is d_phys < 4 SD, and the
    # IAM breakeven and the margin the identification deduces are both of the order of an
    # SD (\dphysBreakevenSdRatio, \dphysSdRatioLambda; their values moved with the
    # annual-mean correction of lambda, which is why they are not restated here). The
    # coefficient must therefore be the price AT d_phys = SD, so that the
    # displayed law is exact at its own anchor. Anchoring at 1 K instead (what this script
    # did until now) leaves the printed law overstating the price by SD^{-4/3} ~ 6.
    s_sd = scc_reduced(sd, lam, var)
    # Two significant figures (the converted level is far below $10, the old rounding).
    coef = float(f"{s_sd:.2g}")
    # ── The conversion, checked ──
    s_old = scc_reduced_pre35(sd, lam, var)
    T_now = MU_NOW_T / E0                    # yr: the budget left at today's emissions
    factor = 4.0 * T_now / lam ** 2          # yr^2 / ... : old / new should equal it
    print("  UNIT CHECK")
    print("    old   : L [$/tCO2] x Omega [1] x Phi/(2 l sqrt(mu)) [yr^2]  ->  $ yr^2 / tCO2 (not $/tCO2)")
    print("    now   : L [$]      x Omega [1] x Phi/(2 l sqrt(mu)) [yr^2] x theta^2/(4 mu_now) [yr^-2/tCO2]  ->  $/tCO2")
    print(f"    mu_now = M* eps_now = {MSTAR_T/1e9:.1f} GtCO2 x {EPS_NOW:.4f} = {MU_NOW_T/1e9:.1f} GtCO2 ; "
          f"E0 = {E0/1e9:g} GtCO2/yr ; T_now = mu_now/E0 = {T_now:.2f} yr ; theta = {lam:.4f}/yr")
    print(f"    level at d_phys = SD: old {s_old:.6g}, now {s_sd:.6g} ; old/new = {s_old/s_sd:.6g} ; "
          f"4 T_now/theta^2 = {factor:.6g}")
    if abs(s_old / s_sd / factor - 1.0) > 1e-9:
        raise SystemExit("[ERROR] reduced_form_scc.py: old/new = %.9g is not 4 T_now/theta^2 = %.9g"
                         % (s_old / s_sd, factor))
    # ── The observed-variables price as written, L Omega Phi eta*/(2 mu_now), and eq:fold_scc, A/sqrt(eps_now) with
    # A = L Omega Phi theta_1/(4 l M*), theta_1 = theta/sqrt(eps_now), l = sqrt(mu)/eta* = theta/(2 eta*).
    eta_sd = eta_star(sd, var)
    rt_sd = 2.0 * rho * eta_sd / lam
    Phi_sd = phi_true_scalar(rt_sd)
    L_dol = L_MED * GDP
    p_obs = L_dol * f * Phi_sd * eta_sd / (2.0 * MU_NOW_T)
    ell_sd = lam / (2.0 * eta_sd)
    theta1 = lam / np.sqrt(EPS_NOW)
    A_fold = L_dol * f * Phi_sd * theta1 / (4.0 * ell_sd * MSTAR_T)
    print(f"    the observed-variables price, L Omega Phi eta*/(2 mu_now) = {p_obs:.9g} ; eq:fold_scc A/sqrt(eps_now) = {A_fold/np.sqrt(EPS_NOW):.9g} ; "
          f"code = {s_sd:.9g}")
    for name, v in (("the observed-variables price", p_obs), ("eq:fold_scc", A_fold / np.sqrt(EPS_NOW))):
        if abs(v / s_sd - 1.0) > 1e-9:
            raise SystemExit("[ERROR] reduced_form_scc.py: %s gives %.9g, the code %.9g" % (name, v, s_sd))
    if f"{p_obs:.2g}" != f"{coef:.2g}":
        raise SystemExit("[ERROR] reduced_form_scc.py: the observed-variables price does not give \\sccReducedCoefAMOC")
    s1 = scc_reduced(1.0, lam, var)
    print(f"  COEFFICIENT @ d_phys=SD={sd:.4f} K: SCC = ${s_sd:.4g}/tCO2  -> reported ${coef:.2g}")
    print(f"  (for reference, at d_phys=1 K: ${s1:.2f}; the ratio {s1/s_sd:.3f} is below the")
    print(f"   pure-power-law SD^(-4/3)={sd**(-4/3):.3f} because the local exponent is")
    print(f"   {local_exponent(sd, lam, var):.3f} at the anchor, not the asymptotic 4/3)")
    print(f"  local exponent dlogSCC/dlogd @ SD  = {local_exponent(sd, lam, var):.4f} "
          f"(small-rho_tilde limit 1.3333)")
    print(f"  rho_tilde @ SD = {2*rho*eta_star(sd, var)/lam:.4f} ; "
          f"eta_star @ SD = {eta_star(sd, var):.4f}")
    print(f"  validity ceiling eta*=1  <=>  d_phys = 4 SD = {4*sd:.4f} K "
          f"(eta* there = {eta_star(4*sd, var):.4f})")
    print()

    print("  family SCC(d_phys) = 4 L f Phi(2 rho eta*/lambda) eta*/lambda^2 :")
    for d in (0.02, 0.05, 0.10, 0.26, 0.50, 1.00, 2.00):
        s = scc_reduced(d, lam, var)
        law = coef * d ** (4.0 / 3.0)
        print(f"    d={d:5.2f} K ({d/sd:5.2f} sd)  SCC=${s:9.2f}   "
              f"4/3-law=${law:9.2f}   ratio={s/law:5.3f}   n_loc={local_exponent(d, lam, var):.3f}")
    print()

    # break-even margins: what d_phys would be needed to come DOWN to a given level
    for label, target in (("retracted published aggregate ($10)", 10.0),
                          ("Dietz et al. 2021 central ($25)", 25.0),
                          ("top of the Dietz et al. 2021 range ($69)", 69.0)):
        d = breakeven(target, lam, var)
        print(f"  break-even d_phys for ${target:.0f}/tCO2 = {d:.4f} K "
              f"({d/sd:.3f} sd, i.e. record sd / {sd/d:.1f})   [{label}]")
    print()
    print(f"  damage sweep is linear in L: SCC(d) spans [0, {2*1:.0f}x] the median value")
    print("=" * 72)

    # ── machine-readable block parsed by 04_tables_values/collect_computed.py ──
    d_pub = breakeven(10.0, lam, var)     # margin needed to fall back to the retracted level
    d_iam = breakeven(69.0, lam, var)     # margin needed to fall to the TOP of the Dietz range
    print("MACROS")
    # emitted with the manuscript's thousands separator (cf. 1{,}000 GtCO2, 100{,}000 draws)
    print(f"  sccReducedCoefAMOC = {coef:.2g}")
    print(f"  sdAMOCrecord = {sd:.2f}")
    print(f"  dphysBreakevenPub = {d_pub:.2f}")
    print(f"  dphysBreakevenIAM = {d_iam:.2f}")
    # The three below replace numbers the text used to state in words. A verbal form that
    # reproduces a computed quantity ("about a third of the standard deviation", "about one
    # kelvin") is a hard-coded number in disguise: if the calibration moves, the sentence
    # goes wrong silently. They are quoted in the units the surrounding text uses.
    print(f"  dphysBreakevenSdRatio = {d_iam/sd:.2f}")     # breakeven margin, in record SDs
    print(f"  etaStarBreakeven = {eta_star(d_iam, var):.2f}")   # eta* there
    print(f"  dphysValidityMax = {4*sd:.2f}")              # eta* = 1  <=>  d_phys = 4 SD
    # the local slope at the anchor: 4/3 is only the asymptote, so the displayed law and
    # the exact family separate away from d_phys = SD (about 8% by one kelvin)
    print(f"  reducedExponentAnchor = {local_exponent(sd, lam, var):.2f}")
    # the margin the identification lambda = 2 sqrt(mu) deduces, in record standard
    # deviations; reported only in the footnote to ssec on the level, since carrying
    # d = lambda into kelvin fixes a unit convention rather than measuring anything
    print(f"  dphysSdRatioLambda = {lam/sd:.2f}")
    print("END MACROS")


if __name__ == "__main__":
    main()
