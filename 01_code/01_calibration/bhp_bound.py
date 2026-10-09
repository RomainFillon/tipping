#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   stdout only (captured by run_all.sh to 02_output/values/raw_runs/bhp_bound.out)
  FEEDS      the sixteen computed \bhp* macros of values.tex, via collect_computed.py
  INPUTS     inputs_literature.DERIVATION_INPUTS, keys prefixed bhp_ (Hambel, van der Ploeg
             and van den Bremer, replication package, Zenodo 20075290, 2026-05-07)
  SEED       none; the script is a closed-form evaluation
  RUNTIME    under 1 s
  IMPLEMENTS eq:ceiling_body and the elasticity of section ssec:ceiling

bhp_bound.py — what the hazard specification in use delivers, and what its class permits.

The hazard of the first tipping transition is floored: lambda_1(T) = h_1T max(0, T - T*).
Writing s = (T - T*)/T for the share of current warming above the threshold and
Lam = lambda_1/rho, and taking the loss to grow as L ~ T^nu, the complete price
d_M(L phi) with phi = lambda_1/(rho + lambda_1) is

    SCC = L chi / (T - T*) . G(Lam),      G(Lam) = Lam [1 + nu s (1 + Lam)] / (1 + Lam)^2
    sup_Lambda SCC = (1 + nu s)^2 / 4 . L chi / (T - T*)   at   Lambda* = (1 + nu s)/(1 - nu s)

which returns the familiar L chi / (4 T) at s = 1, nu = 0. NOTHING below is hard-coded:
every printed quantity is derived from the sourced inputs, so moving a parameter in
inputs_literature.py moves the manuscript.

Both nu and T* are parameters here rather than constants, because ssec:ceiling varies
them: nu for the sign result, T* for the statement that the supremum is unbounded in it.

Usage:  python 01_code/01_calibration/bhp_bound.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inputs_literature import DERIVATION_INPUTS  # noqa: E402


def inp(key):
    """Sourced value only. A missing key is fatal and named: no default, no fallback."""
    if key not in DERIVATION_INPUTS:
        raise SystemExit("[ERROR] bhp_bound.py: DERIVATION_INPUTS has no '%s'. Declare it "
                         "with its source rather than defaulting it here." % key)
    return float(DERIVATION_INPUTS[key][0])


def price(T, nu, h1, Tstar, rho):
    """d_M(L phi) / (chi L_0 T_0^-nu), the complete price up to a positive constant."""
    lam = h1 * max(0.0, T - Tstar)
    Lam = lam / rho
    if Lam == 0.0:
        return 0.0
    s = (T - Tstar) / T
    return (T ** nu) / (T - Tstar) * Lam * (1.0 + nu * s * (1.0 + Lam)) / (1.0 + Lam) ** 2


def elasticity(T, nu, h1, Tstar, rho, rel=1e-6):
    """dlog SCC / dlog T, by symmetric difference in log T."""
    import math
    lo, hi = T * (1 - rel), T * (1 + rel)
    return (math.log(price(hi, nu, h1, Tstar, rho)) -
            math.log(price(lo, nu, h1, Tstar, rho))) / (math.log(hi) - math.log(lo))


def bisect(f, a, b, n=200):
    fa = f(a)
    for _ in range(n):
        m = 0.5 * (a + b)
        if fa * f(m) <= 0:
            b = m
        else:
            a, fa = m, f(m)
    return 0.5 * (a + b)


def main():
    h1    = inp("bhp_h1T")        # /degC/yr, marginal arrival rate of the first tipping point
    Tstar = inp("bhp_Tstar")      # degC, tipping threshold (floor)
    T0    = inp("bhp_T0")         # degC, initial temperature anomaly
    rho   = inp("bhp_rho")        # /yr, pure rate of time preference
    lam2  = inp("bhp_lambda2")    # /yr, arrival rate of subsequent tipping
    chipre= inp("bhp_chi_pre")    # degC/GtC, TCRE pre-tip
    E0    = inp("bhp_E0")         # GtC, initial cumulative emissions
    F0    = inp("bhp_F0")         # GtC/yr, initial emissions
    rungs = inp("bhp_rungs")      # transitions in the cascade
    nu    = inp("bhp_nu")         # loss elasticity in warming, L ~ T^nu
    Tup   = inp("bhp_T_upper")    # degC, upper end of the range the premium is quoted over

    lam1 = h1 * max(0.0, T0 - Tstar)
    Lam    = lam1 / rho
    s    = (T0 - Tstar) / T0
    G    = Lam * (1.0 + nu * s * (1.0 + Lam)) / (1.0 + Lam) ** 2
    Lam_star= (1.0 + nu * s) / (1.0 - nu * s)
    Gsup = Lam_star * (1.0 + nu * s * (1.0 + Lam_star)) / (1.0 + Lam_star) ** 2

    nustar = bisect(lambda v: elasticity(T0, v, h1, Tstar, rho), 0.0, 1.0)

    out = {
        "bhpLambda":           "%.5f" % Lam,
        "bhpS":               "%.4f" % s,
        "bhpLambdaStar":           "%.2f" % Lam_star,
        "bhpShare":           "%.2f" % (100.0 * G / Gsup),
        "bhpPriceRatio":      "%.1f" % (Gsup / G),
        # In percent per degC per year, like \bhpHone, and carrying that unit: the model's
        # h_1T is a rate per degC per year, 100 times smaller than the number shown.
        # Math mode only; no space, since collect_computed.py reads the value up to one.
        "bhpHoneStar":        r"%.2f\%%\,{}^{\circ}\mathrm{C}^{-1}\,\mathrm{yr}^{-1}"
                              % (100.0 * rho * Lam_star / (T0 - Tstar)),
        "bhpHoneRatio":       "%.0f" % (Lam_star / Lam),
        "bhpWait":            "%.0f" % (1.0 / lam1),
        "bhpCascadeFactor":   "%.4f" % (lam2 / (rho + lam2)),
        "bhpUnravel":         "%.0f" % ((rungs - 1.0) / lam2),
        "bhpKinkE":           "%.2f" % (Tstar / chipre),
        "bhpKinkYears":       "%.1f" % ((E0 - Tstar / chipre) / F0),
        "bhpNuStar":          "%.4f" % nustar,
        "bhpElasticityNuZero": "%+.4f" % elasticity(T0, 0.0, h1, Tstar, rho),
        # Three significant figures (it printed +1.6944).
        "bhpElasticityNuOne":  "%+.3g" % elasticity(T0, 1.0, h1, Tstar, rho),
        # C states the discount rate and lambda_2 behind \bhpCascadeFactor,
        # read from the same inputs, never retyped.
        "bhpRho":             "%.4f" % rho,
        "bhpLamTwo":          "%.2f" % lam2,
        "bhpPremiumRatio":    "%.2f" % (price(Tup, nu, h1, Tstar, rho) /
                                        price(T0, nu, h1, Tstar, rho)),
    }

    # bhpNuStar is typeset as an exponent, T^{\bhpNuStar}. It must therefore be a bare
    # number: no unit, no sign, no percent, no non-breaking space. Asserted here rather
    # than trusted, because the failure is silent in the pdf.
    import re
    if not re.fullmatch(r"\d+[.,]\d+", out["bhpNuStar"]):
        raise SystemExit("[ERROR] bhp_bound.py: bhpNuStar = %r is not a bare decimal, and "
                         "it is typeset as an exponent." % out["bhpNuStar"])

    print("bhp_bound.py — hazard with a floor: what it delivers, what its class permits")
    print("  lambda_1(T0) = %.6f /yr   mean wait = %.0f yr   Lam = %.5f   s = %.4f"
          % (lam1, 1.0 / lam1, Lam, s))
    print("  G = %.5f   sup_Lambda G = %.4f (at Lambda* = %.2f)   share = %.2f %%"
          % (G, Gsup, Lam_star, 100.0 * G / Gsup))
    print("  elasticity in T:  nu=0 %+.4f   nu=1 %+.4f   zero at nu* = %.4f"
          % (elasticity(T0, 0.0, h1, Tstar, rho),
             elasticity(T0, 1.0, h1, Tstar, rho), nustar))
    print("  supremum is unbounded in T*, in units of L.chi/T:"
          "  sup = (1+nu.s)^2/(4s) -> infinity as s -> 0")
    for sv in (0.5, 0.2, s, 0.02, 0.005):
        print("     s = %.4f   sup SCC / (L.chi/T) = %8.2f"
              % (sv, (1.0 + nu * sv) ** 2 / (4.0 * sv)))
    print()
    for k in sorted(out):
        print("BHP %-22s = %s" % (k, out[k]))


if __name__ == "__main__":
    main()
