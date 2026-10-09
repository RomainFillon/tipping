#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   stdout only (captured by run_all.sh to 02_output/values/raw_runs/crossover.out)
  FEEDS      the five computed \xover* macros of values.tex, via collect_computed.py
  INPUTS     inputs_literature.DERIVATION_INPUTS -- DT_current (Forster et al. 2024), and the ends of
             the AMOC threshold range DTstar_AMOC_p5 / DTstar_AMOC_p95 (Armstrong McKay
             et al. 2022, Table 1)
  SEED       none; the script is a closed-form evaluation
  RUNTIME    under 1 s
  IMPLEMENTS the power law SCC ~ eps^{p-1} of the theorem (ssec:gen_thm), read at two exponents

crossover.py -- which of the two disagreements moves the tipping price more.

THE QUESTION. Climate economics argues at length about WHERE a threshold sits and settles
in a line HOW the risk rises on the way in. This script puts the two on one axis. The
price of restraint follows a power law in the proximity,

    SCC(eps) = A eps^{p-1},        eps = (DT* - DT_now)/DT*,

so there are exactly two ways to move it: move eps, or move p.

  LOCATION.  Hold the exponent at the fold value p = 1/2 and walk the threshold from one
             end of the published AMOC range to the other. The leading law eq:gen_master is
             a power of the BUDGET left, mu^{p-1} with mu = M* eps, and M* = DT* x 1000/TCRE
             moves with the threshold. The ratio of the two prices is therefore a ratio of
             budgets, in which the TCRE cancels (it read (eps_high/eps_low)^{1-p} until step
             34, which held M* fixed while moving DT*):

                 R_loc = (mu_low / mu_high)^{p-1}
                       = ((DT*_high - DT_now) / (DT*_low - DT_now))^{1-p}.

  CLASS.     Hold eps and move the exponent from the fold value p = 1/2 to p = 1:

                 R_cls(eps) = eps^{-(1 - 1/2)} = eps^{-1/2}.

             THIS ONE CARRIES A CONVENTION and the paper must say so. A ratio of two
             different power laws is only a number once their prefactors are tied
             together; here they are tied at eps = 1, the moment the budget is untouched,
             which is the only proximity at which the two specifications describe the same
             undisturbed system. The UNBOUNDEDNESS of the class factor is free of that
             choice -- eps^{-1/2}/const diverges as eps -> 0 whatever A is -- but the
             numerical crossover below is not, and it is quoted as what it is.

TWO READINGS OF THE SAME NUMBER, and they are not the same claim. R_cls(eps) is the factor
between p = 1/2 and p = 1 at fixed proximity, and p = 1 arises in two ways:

  (a) the TRANSCRITICAL bifurcation, whose phase portrait is the fold's, so the factor
      measures what the phase portrait does not determine;
  (b) any SMOOTH hazard rate, whichever functional form, so the factor measures the gap
      between the class the structural literature assigns to the AMOC and the class its
      economic representation implies.

Reading (b) is the one the introduction and the climate application carry, in the
subsection that asks which parameter decides, because it says that the applied literature
prices a fold in the bounded class for an element the structural literature has already
classified as a fold. Reading (a) stays where it is, in the section that shows the phase
portrait does not decide. The number is identical and is computed once, here, so that the
two passages cannot drift apart.

THE CROSSOVER is the proximity at which the second disagreement overtakes the first:

    R_cls(eps*) = R_loc   <=>   eps* = (DT*_low - DT_now) / (DT*_high - DT_now),

a closed form that this script re-derives numerically and checks against, rather than
printing one of the two.

Usage:  python 01_code/01_calibration/crossover.py
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inputs_literature import DERIVATION_INPUTS  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_solvers"))
from price import lead_price_fixed, boundary_power_speed  # noqa: E402  (the leading law)

# The two exponents the comparison runs between. p_FOLD is the closing rate of the margin
# at a saddle-node, which the exponent section of the paper proves; p_SMOOTH is what a
# smooth hazard delivers, whatever its functional form, and is also the transcritical
# exponent.
p_FOLD, p_SMOOTH = 0.5, 1.0


def inp(key):
    """Sourced value only. A missing key is fatal and named: no default, no fallback."""
    if key not in DERIVATION_INPUTS:
        raise SystemExit("[ERROR] crossover.py: DERIVATION_INPUTS has no '%s'. Declare it "
                         "with its source rather than defaulting it here." % key)
    return float(DERIVATION_INPUTS[key][0])


def proximity(DTstar, DTnow):
    """eps = (DT* - DT_now)/DT*, the fraction of the budget to the threshold still unspent.

    This is the definition fig_forest_tc.py and eps_inf_ci.py use; it is repeated here as
    one line rather than imported, because those two are Monte Carlo scripts that would
    have to run for it.
    """
    if not 0.0 < DTnow < DTstar:
        raise SystemExit("[ERROR] crossover.py: eps is only defined for 0 < DT_now < DT*, "
                         "and DT_now = %g, DT* = %g." % (DTnow, DTstar))
    return (DTstar - DTnow) / DTstar


def price(eps, p):
    """SCC(eps) up to the prefactor A, which is common to the two exponents (see header): the
    leading law at fixed state of a boundary that moves as mu^p (price.py), normalised at eps = 1,
    which is eps^(p - 1)."""
    return float(lead_price_fixed(1.0, 1.0, boundary_power_speed(1.0, p, eps))
                 / lead_price_fixed(1.0, 1.0, boundary_power_speed(1.0, p, 1.0)))


def dlog_dlog_eps(eps, p, rel=1e-6):
    """dlog SCC / dlog eps, by symmetric difference in log eps. Should return p - 1."""
    lo, hi = eps * (1 - rel), eps * (1 + rel)
    return (math.log(price(hi, p)) - math.log(price(lo, p))) / (math.log(hi) - math.log(lo))


def dlog_dp(eps, p, h=1e-6):
    """dlog SCC / dp at fixed eps, by symmetric difference in p. Should return log eps."""
    return (math.log(price(eps, p + h)) - math.log(price(eps, p - h))) / (2.0 * h)


def main():
    DTnow = inp("DT_current")           # degC above pre-industrial, IPCC AR6
    DTlo  = inp("DTstar_AMOC_p5")       # degC, low end of the AMOC threshold range
    DThi  = inp("DTstar_AMOC_p95")      # degC, high end
    DTmid = inp("DTstar_AMOC_best")     # degC, central threshold (reported, not a macro here)

    eps_lo = proximity(DTlo, DTnow)     # NEAREST threshold -> SMALLEST eps -> HIGHEST price
    eps_hi = proximity(DThi, DTnow)
    eps_mid = proximity(DTmid, DTnow)

    if not 0.0 < eps_lo < eps_hi < 1.0:
        raise SystemExit("[ERROR] crossover.py: expected 0 < eps_low < eps_high < 1, got "
                         "%.6f and %.6f. A threshold range that does not bracket the "
                         "current warming is not what the factors below assume."
                         % (eps_lo, eps_hi))

    # --- the two factors -------------------------------------------------------------
    # The location factor on the budget. price(x, p) is x^(p-1) normalised at x = 1,
    # so price(mu_low/mu_high, p) is the ratio of the leading laws at the two budgets.
    mu_ratio = (DTlo - DTnow) / (DThi - DTnow)          # mu_low / mu_high, the TCRE cancels
    R_loc = price(mu_ratio, p_FOLD)
    def R_cls(e):
        return price(e, p_FOLD) / price(e, p_SMOOTH)

    R_cls_at_lo = R_cls(eps_lo)

    # --- the crossover, derived numerically and checked against its closed form --------
    # R_cls(eps) = eps^{-(p_SMOOTH - p_FOLD)} is strictly decreasing in eps, so bisection
    # on log R_cls(eps) - log R_loc has exactly one root and no sign ambiguity.
    def f(e):
        return math.log(R_cls(e)) - math.log(R_loc)
    a, b = 1e-12, 1.0
    if f(a) * f(b) > 0:
        raise SystemExit("[ERROR] crossover.py: no crossover on (0,1]; the class factor "
                         "does not cross the location factor.")
    for _ in range(300):
        m = 0.5 * (a + b)
        if f(a) * f(m) <= 0:
            b = m
        else:
            a = m
    eps_star = 0.5 * (a + b)
    eps_star_closed = mu_ratio
    if abs(eps_star - eps_star_closed) > 1e-9:
        raise SystemExit("[ERROR] crossover.py: the bisected crossover %.12f and the closed "
                         "form (DT*_low - DT_now)/(DT*_high - DT_now) = %.12f disagree. One of the two is wrong "
                         "and the paper quotes the number."
                         % (eps_star, eps_star_closed))

    # --- the two elasticities the 'which parameter decides' subsection states ---------
    # Both are exact here; they are recomputed by finite difference so that the statement
    # in the paper is checked against the power law rather than asserted next to it.
    for e in (eps_lo, eps_star, eps_mid, eps_hi):
        for p in (p_FOLD, p_SMOOTH):
            if abs(dlog_dlog_eps(e, p) - (p - 1.0)) > 1e-6:
                raise SystemExit("[ERROR] crossover.py: dlog SCC/dlog eps is not p-1 at "
                                 "eps=%g, p=%g." % (e, p))
        if abs(dlog_dp(e, p_FOLD) - math.log(e)) > 1e-6:
            raise SystemExit("[ERROR] crossover.py: dlog SCC/dp is not log eps at eps=%g."
                             % e)

    inside = eps_lo <= eps_star <= eps_hi

    out = {
        "xoverEpsLow":           "%.3f" % eps_lo,
        "xoverEpsHigh":          "%.3f" % eps_hi,
        "xoverRangeFactor":      "%.2f" % R_loc,
        "xoverEps":              "%.3f" % eps_star,
        # The share of the budget to the threshold still unspent at the crossing,
        # in percent, two significant figures.
        "xoverEpsPct":           ("%.1f" % (100 * eps_star) if 100 * eps_star < 10
                                  else "%.0f" % (100 * eps_star)),
        "xoverClassFactorAtLow": "%.2f" % R_cls_at_lo,
    }

    # Every one of the six is typeset as a bare number -- three of them inside "a factor
    # of ...", two as a proximity -- so none may carry a unit, a sign or a percent sign.
    # Asserted here rather than trusted, because the failure is silent in the pdf.
    import re
    for k, v in out.items():
        if not re.fullmatch(r"\d+(\.\d+)?", v):
            raise SystemExit("[ERROR] crossover.py: %s = %r is not a bare decimal, and it "
                             "is typeset as one." % (k, v))

    print("crossover.py -- location against class, on the AMOC threshold range")
    print("  DT_now = %.2f degC;  DT* range [%.2f, %.2f] degC, central %.2f"
          % (DTnow, DTlo, DThi, DTmid))
    print("  eps at the range ends: low %.6f (nearest threshold), high %.6f; central %.6f"
          % (eps_lo, eps_hi, eps_mid))
    print()
    print("  LOCATION factor, p = %.2f held fixed, across the whole published range:" % p_FOLD)
    print("      R_loc = (mu_low/mu_high)^(p-1) = ((DT*_high-DT_now)/(DT*_low-DT_now))^(1-p) = %.6f" % R_loc)
    print("  CLASS factor, eps held fixed, p from %.2f to %.2f, prefactors tied at eps = 1:"
          % (p_FOLD, p_SMOOTH))
    print("      R_cls(eps) = eps^-(p_smooth - p_fold) = eps^-%.2f" % (p_SMOOTH - p_FOLD))
    print("      at eps_low  %.6f     at eps_central %.6f     at eps_high %.6f"
          % (R_cls_at_lo, R_cls(eps_mid), R_cls(eps_hi)))
    print("      -> unbounded as eps -> 0, for any prefactor; the LEVEL below is not.")
    print()
    print("  CROSSOVER  R_cls(eps*) = R_loc  at  eps* = %.6f" % eps_star)
    print("      closed form eps* = (DT*_low-DT_now)/(DT*_high-DT_now) = %.6f  (agree to %.1e)"
          % (eps_star_closed, abs(eps_star - eps_star_closed)))
    print("      inside the published range [%.6f, %.6f]? %s"
          % (eps_lo, eps_hi, "YES" if inside else "NO"))
    if not inside:
        print("      (on the budget, the crossing lies below the published range, in the")
        print("       last stretch of the window; ssec:which_parameter says so, and collect_computed.py asserts it")
        print("       lies above eps_dyn)")
    print()
    print("  ELASTICITIES of the which-parameter-decides subsection, checked above:")
    print("      dlog SCC / dlog eps = p - 1 = %+.2f, INDEPENDENT of eps" % (p_FOLD - 1.0))
    print("      dlog SCC / dp       = log eps, which diverges as eps -> 0:")
    for e in (eps_hi, eps_mid, eps_star, eps_lo, 0.05, 0.01):
        print("          eps = %.4f   dlog SCC/dp = %+.4f" % (e, math.log(e)))
    print()
    print("  THE SAME class factor reads two ways (see header): p = 1 is the")
    print("  transcritical exponent, and p = 1 is what every smooth hazard delivers.")
    print()
    for k in sorted(out):
        print("XOVER %-22s = %s" % (k, out[k]))


if __name__ == "__main__":
    main()
