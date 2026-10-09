# The level A and the floor ε_∞ are not computed: in dollars per tonne they require η*, which no element identifies (the appendix on where the elements sit); the deadline bound eq:tc_bound drops the floor.
# -*- coding: utf-8 -*-
# The docstring is raw: it lists macro names, and \j in \jointSamplePct is not a valid
# escape. Python warned about it on every run -- the second line a replicator saw -- and
# a future version makes it an error.
r"""
REPLICATION HEADER
  PRODUCES   fig_forest_tc.pdf and the MACROS block parsed into computed.json
  FEEDS      fig:forest_tc, \fracWAISfirst, \fracAmazonBeforeAMOC,
             \fracWAISalreadyPct, \fracWAISfirstIncl, \jointSamplePct, \tcAMOCPten,
             \tcAmazonPten, \tcWAISPten; the bound on SSP2-4.5 and SSP3-7.0,
             \tc<element>SSPmid, \tc<element>SSPhigh with Pten, Pninety, NotReachedPct,
             \EglobalRateSSPmid, \EglobalRateSSPhigh, \emissionPathStartYear, LastYear
  INPUTS     inputs_literature.py; the same draws as eps_inf_ci.py; the
             emission paths of emission_paths.py (RCMIP v5.1.0 extract, 00_data/emissions/)
  SEED       numpy seed 42
  RUNTIME    4 to 10 min (three cold runs; the spread is machine load, not the work)
  IMPLEMENTS eq:tc_bound, plotted as a forest of median bounds with [P10,P90]

fig_forest_tc.py — urgency-ranking forest plot (Figure in the policy section; replaces the
former eps_inf/t_c body tables, now in the appendix).

Replicates EXACTLY the Monte Carlo of ../01_calibration/eps_inf_ci.py (seed=42, same draw
order and same common priors, incl. the harmonized sigma prior [0.05,0.45] for Amazon and
WAIS). Draws the deadline BOUND t_c <= (M*/E0) eps_0 (median dot + P10-P90 bar) per element,
ordered by urgency (WAIS < Amazon < AMOC), x on a log scale.

WHY THE BOUND AND NOT THE FLOOR-BASED DATE.  The floor eps_inf = (A/c_bar)^2 needs the level A,
which in dollars per tonne requires eta*, which no element identifies; it was withdrawn (see
the line above). The bound needs neither: (M*/E0) eps_0 = mu_now/E0 exactly, the remaining
warming divided by TCRE divided by the emission rate.

M* IS THE PRE-INDUSTRIAL-TO-THRESHOLD BUDGET, DT*/TCRE, because eps_0 = (DT*-DT_now)/DT* is
the fraction of that budget still unspent. This script used to pair eps_0 with the REMAINING
budget, which applies eps_0 twice and shortens every deadline by exactly 1/eps_0 (x1.43 AMOC,
x1.52 Amazon, x5.0 WAIS).

Writes fig_forest_tc.pdf/.png in the current directory (run_all.sh runs it from
02_output/figures, then copies the PDF to manuscript/figures/).
"""
# The macros whose values this figure prints or draws. make_tables_values.py
# reads this tuple (as text, without running the script) and refuses to drop any of them
# from values.tex: a number on a figure must stay traceable to a macro.
# A centile that the path does not reach before its last year
# is drawn as an arrow at the axis edge and gets no macro, so it is NOT declared here. The
# script checks at run time that every name below was emitted, and stops otherwise: a
# centile that ceases to be reached must be taken out of this tuple by hand, never dropped
# silently. \tcAmazonSSPhighPninety left it for that reason.
FIGURE_MACROS = (
    "tcWAIS",
    "tcWAISSSPmid",
    "tcWAISSSPhigh",
    "tcAmazon",
    "tcAmazonSSPhigh",
    "tcWAISPten",
    "tcWAISSSPmidPten",
    "tcWAISSSPhighPten",
    "tcAmazonPten",
    "tcAmazonSSPmidPten",
    "tcAmazonSSPhighPten",
    "tcWAISSSPmidPninety",
    "tcWAISSSPhighPninety",
    "tcWAISSSPmidNotReachedPct",
    "tcWAISSSPhighNotReachedPct",
    "tcAmazonSSPmidNotReachedPct",
    "tcAmazonSSPhighNotReachedPct",
    "fracWAISalreadyPct",
)

import os
import sys
import numpy as np
from scipy.stats import norm as spnorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lmax from its single home (inputs_literature.py), like eps_inf_ci.py whose element
# dicts this figure reproduces draw for draw.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "01_calibration"))
from inputs_literature import LITERATURE_MACROS, DERIVATION_INPUTS  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_solvers"))
from tcre_draw import draw_tcre  # noqa: E402

L_MAX = float(LITERATURE_MACROS["Lmax"][0])   # 0.67


def _lit(key):
    """A sourced constant, by name. A missing key is fatal and named: no default here."""
    if key not in DERIVATION_INPUTS:
        raise SystemExit("[ERROR] fig_forest_tc.py: DERIVATION_INPUTS has no '%s'. "
                         "Declare it with its source rather than writing it out here."
                         % key)
    return float(DERIVATION_INPUTS[key][0])


def _DTstar(element):
    """(central, low end, high end) of an element's threshold, from the one sourced home.
    Same reason as the Lmax note above: these nine numbers had three copies."""
    return tuple(_lit("DTstar_%s_%s" % (element, k)) for k in ("best", "p5", "p95"))


np.random.seed(42)
N = int(DERIVATION_INPUTS["N_MC"][0])
DT_current = _lit("DT_current")
E0_Gt = 40.0                      # GtCO2/yr


def fit_lognormal(p5, best, p95):
    return np.log(best), (np.log(p95)-np.log(p5))/(2*spnorm.ppf(0.95))
def sample_lognormal(p5, best, p95, n, lower=0.0, upper=np.inf):
    mu, sig = fit_lognormal(p5, best, p95)
    return np.clip(np.exp(np.random.normal(mu, sig, n)), lower, upper)
def sample_uniform(lo, hi, n, floor=0.0):
    return np.clip(np.random.uniform(lo, hi, n), floor, None)
def sample_loguniform(lo, hi, n):
    return np.exp(np.random.uniform(np.log(lo), np.log(hi), n))
def draw_L(p, n):
    return sample_uniform(p["L_lo"], p["L_hi"], n, floor=0.0)
def draw_sigma(p, n):
    if "sigma_lo" in p:
        return sample_loguniform(p["sigma_lo"], p["sigma_hi"], n)
    return np.clip(np.random.lognormal(np.log(p["sigma_mu"]), p["sigma_cv"], n), 0.02, 2.0)
def draw_budget(DT_best, DT_p5, DT_p95, TCRE):
    """Returns (M_star, mu_now, eps0) per draw, all consistent with one another.

    M* is the budget from the PRE-INDUSTRIAL to the threshold, DT*/TCRE, because the
    paper's proximity eps_0 = (DT* - DT_now)/DT* is the fraction of THAT budget still
    unspent. Pairing eps_0 with the remaining budget instead applies (DT*-DT_now)/DT*
    twice: once inside the budget and once in eps_0. The remaining budget is then
    mu_now = M* eps_0, which is what the deadline actually runs on.
    """
    DT = sample_lognormal(DT_p5, DT_best, DT_p95, len(TCRE), lower=0.1)
    M_star = DT / TCRE * 1000.0
    mu_now = np.maximum(0.0, DT - DT_current) / TCRE * 1000.0
    return M_star, mu_now, mu_now / M_star


# element dicts identical to eps_inf_ci.py (common L prior [0,Lmax]; common sigma prior
# [0.05,0.45] for Amazon and WAIS).
# The DT* triples are read from inputs_literature.py, their one home. "eps0" is NOT
# derived from them although it is (DT_best - DT_current)/DT_best: the Amazon entry is the
# ROUNDED 0.66 where the definition gives 0.657143, and eps_inf_ci.py carries the same
# rounded dict, so the two scripts agree with each other and both round. Deriving it here
# would move the Amazon deadline t_c, silently and for a reason unrelated to this change.
# Recorded, not repaired.
elements = {
    "AMOC":   {"DT_best":_DTstar("AMOC")[0],  "DT_p5":_DTstar("AMOC")[1],  "DT_p95":_DTstar("AMOC")[2],  "sigma_mu":_lit("sigma_mu_AMOC"),"sigma_cv":_lit("sigma_cv_AMOC"),"L_lo":0.0,"L_hi":L_MAX,"eps0":0.70},
    "Amazon": {"DT_best":_DTstar("Amazon")[0],"DT_p5":_DTstar("Amazon")[1],"DT_p95":_DTstar("Amazon")[2],"sigma_lo":DERIVATION_INPUTS["sigma_prior_lo"][0],"sigma_hi":DERIVATION_INPUTS["sigma_prior_hi"][0],"L_lo":0.0,"L_hi":L_MAX,"eps0":0.66},
    "WAIS":   {"DT_best":_DTstar("WAIS")[0],  "DT_p5":_DTstar("WAIS")[1],  "DT_p95":_DTstar("WAIS")[2],  "sigma_lo":DERIVATION_INPUTS["sigma_prior_lo"][0],"sigma_hi":DERIVATION_INPUTS["sigma_prior_hi"][0],"L_lo":0.0,"L_hi":L_MAX,"eps0":0.20},
}

# reproduce the EXACT draw order of eps_inf_ci.py
TCRE = draw_tcre(np.random, N)

stats = {}
for name, p in elements.items():
    M, mu, eps0_draws = draw_budget(p["DT_best"], p["DT_p5"], p["DT_p95"], TCRE)
    s = draw_sigma(p, N)
    L = draw_L(p, N)
    # Condition on the element not having tipped already, exactly as before: the filter is
    # on the REMAINING budget, so draws with DT* below present-day warming are excluded
    # rather than entered as a deadline of zero. Filtering on M* instead would silently
    # change the sample as well as the pairing.
    # s and L entered only the level A, which is no longer computed; they are still
    # drawn, in the same order, so that the random stream is the one every number was computed
    # on. The sample was mu_now > 5 and Phi finite; Phi was finite and positive on every draw,
    # so the sample is unchanged.
    valid = mu > 5.0
    Mok, muok, eps0 = M[valid], mu[valid], eps0_draws[valid]
    # The deadline is reported on the BOUND t_c <= (M*/E0) eps_0 = mu_now/E0 exactly, i.e. the
    # remaining warming over TCRE over the emission rate, with no unmeasured quantity.
    tc = muok / E0_Gt
    M_med, mu_med = np.percentile(Mok, 50), np.percentile(muok, 50)
    eps0_med = float(np.percentile(eps0, 50))
    tc_central = (M_med / E0_Gt) * eps0_med
    # Retain t_c PER DRAW, realigned onto the full 0..N-1 index, so the ordering between
    # elements can be evaluated draw by draw. The elements share the TCRE draws, so the
    # question "in what share of draws does the ordering hold?" is meaningful; it cannot be
    # answered from the marginal P10/P90 alone, which overlap heavily for Amazon vs AMOC.
    # No new random numbers are drawn here: the draw order is unchanged.
    tc_full = np.full(N, np.nan)
    tc_full[np.where(valid)[0]] = tc
    # Draws whose threshold lies below present warming have mu_now = 0 exactly (the max(0,.)
    # of draw_budget), so the remaining budget is nil and the deadline has passed rather than
    # being short. They are EXCLUDED rather than entered at t_c = 0, and the reason is that the
    # model no longer applies to them: SCC_tip = A/sqrt(eps) prices a boundary not yet reached,
    # so such a draw is not one in which the element leads but one the formula does not
    # describe. The t_c = 0 variant retained below is a CONTROL on that choice, never the
    # reported number — it says how much the exclusion costs the ordering share, and the answer
    # is that it costs it in the conservative direction.
    tc_full_incl = tc_full.copy()
    tc_full_incl[mu <= 0.0] = 0.0
    stats[name] = {
        "eps0": float(np.percentile(eps0, 50)), "tc_central": tc_central,
        "tc_p10": np.percentile(tc, 10), "tc_p90": np.percentile(tc, 90),
        # The median OF THE BOUND over the draws, the quantity the text quotes as \tc*
        # (eps_inf_ci.py: median of mu_now over the same draws, over E0). tc_central, the
        # product of the medians, is a different number (160.6 against 159.7 for the AMOC).
        "tc_med": float(np.percentile(tc, 50)),
        "Mstar_med": float(M_med), "mu_med": float(mu_med),
        "tc_draws": tc_full,
        # share of draws already past threshold, and the t_c = 0 control
        "frac_already": float((mu <= 0.0).mean()),
        "tc_draws_incl": tc_full_incl,
        # mu_now over the plotted sample, for the emission paths
        "mu_draws": muok,
    }

print("forest t_c on the BOUND  (median-param central, P10-P90 from distribution)")
for name in elements:
    d = stats[name]
    print(f"  {name:6s}: median of the bound over the draws = {d['tc_med']:.2f} yr (the plotted point)")
    print(f"  {name:6s}: t_c={d['tc_central']:6.1f} yr  [P10 {d['tc_p10']:5.1f}, P90 {d['tc_p90']:6.1f}]"
          f"  M*_med={d['Mstar_med']:.0f}  mu_med={d['mu_med']:.0f}  eps0_med={d['eps0']:.3f}")

# ── draw-by-draw ordering: what the caption is allowed to claim ──────────────
tW, tAm, tAM = (stats[k]["tc_draws"] for k in ("WAIS", "Amazon", "AMOC"))
both = np.isfinite(tW) & np.isfinite(tAm) & np.isfinite(tAM)
n_ok = int(both.sum())
f_wais_first = float(((tW[both] <= tAm[both]) & (tW[both] <= tAM[both])).mean())
f_am_before_amoc = float((tAm[both] < tAM[both]).mean())
f_full = float(((tW[both] <= tAm[both]) & (tAm[both] < tAM[both])).mean())
print(f"\nordering across draws (n={n_ok}):")
print(f"  WAIS first                 = {f_wais_first:.1%}")
print(f"  Amazon before AMOC         = {f_am_before_amoc:.1%}")
print(f"  full order WAIS<Amazon<AMOC= {f_full:.1%}")

# ── the conditioning the three shares above are computed under ────────────────
# The joint sample is the intersection of the three per-element filters, and it is
# selected overwhelmingly by ONE of them: a quarter of the WAIS draws place its threshold
# below present-day warming. So the lead, the median horizon and the P10-P90 bar are all
# conditional on no element having crossed already, and the paper has to say so.
#
# The control: entering the excluded draws at t_c = 0 instead. It RAISES the ice sheet's
# lead, because the draws removed are exactly those in which it is furthest ahead, so the
# exclusion cannot be what produces the lead. It is reported for that reason and not as an
# alternative estimate — see the note at the retention of tc_draws_incl above.
frac_wais_already = stats["WAIS"]["frac_already"]
joint_sample = n_ok / N
iW, iAm, iAM = (stats[k]["tc_draws_incl"] for k in ("WAIS", "Amazon", "AMOC"))
both_i = np.isfinite(iW) & np.isfinite(iAm) & np.isfinite(iAM)
f_wais_first_incl = float(((iW[both_i] <= iAm[both_i]) & (iW[both_i] <= iAM[both_i])).mean())
f_am_before_amoc_incl = float((iAm[both_i] < iAM[both_i]).mean())
print("\nconditioning of the joint sample:")
for name in elements:
    print(f"  {name:6s}: DT* below present warming in {stats[name]['frac_already']:.3%} of draws")
print(f"  joint sample (all three finite) = {n_ok}/{N} = {joint_sample:.3%}")
print(f"  CONTROL, excluded draws entered at t_c=0 (n={int(both_i.sum())}):")
print(f"    WAIS first         = {f_wais_first_incl:.3%}  (reported: {f_wais_first:.3%})")
print(f"    Amazon before AMOC = {f_am_before_amoc_incl:.3%}  (reported: {f_am_before_amoc:.3%})")

# ── The ice sheet against the Amazon only ──────────────────────────
# The body ranks two elements, so note 4 quotes the share in which the ice sheet comes before
# the Amazon. Same draws and same convention as fracWAISfirst: the sample is the draws in which
# both bounds are finite (t_c > 0, the AMOC playing no part), ties counted as the ice sheet
# first (<=, as above); the control enters the excluded draws at t_c = 0.
pair = np.isfinite(tW) & np.isfinite(tAm)
f_wais_before_am = float((tW[pair] <= tAm[pair]).mean())
pair_i = np.isfinite(iW) & np.isfinite(iAm)
f_wais_before_am_incl = float((iW[pair_i] <= iAm[pair_i]).mean())
f_wais_before_am_joint = float((tW[both] <= tAm[both]).mean())
print("\nice sheet before the Amazon:")
print(f"  pairwise sample (n={int(pair.sum())}) = {f_wais_before_am:.3%}")
print(f"  CONTROL, excluded draws at t_c=0 (n={int(pair_i.sum())}) = {f_wais_before_am_incl:.3%}")
print(f"  for reference, on the three-element joint sample (n={n_ok}) = {f_wais_before_am_joint:.3%}")


# ── The bound on two emission paths, SSP2-4.5 and SSP3-7.0 ──────────────────
# Same draws, same sample (mu_now > 5), same mu_now: only the path along
# which the budget is spent changes. The constant path goes through the same module and
# must give back the median and P10 above to the last digit; the run stops otherwise.
# Draws the path never brings to mu_now are "not reached" and count as +inf in the centiles.
import emission_paths as ep   # noqa: E402  (01_code/01_calibration, on sys.path below)
PATHS = {"now": None}
PATHS.update({k: ep.annual_path(v)[1] for k, v in ep.SCENARIOS.items()})
path_stats = {}
for name in elements:
    mu_d = stats[name]["mu_draws"]
    path_stats[name] = {}
    for key, e in PATHS.items():
        t = ep.deadline_constant(mu_d, E0_Gt) if e is None else ep.deadline(mu_d, e)
        p10, p50, p90 = ep.quantiles_with_unreached(t)
        path_stats[name][key] = {"p10": p10, "p50": p50, "p90": p90,
                                 "unreached": float(np.isinf(t).mean()), "t": t}
    c = path_stats[name]["now"]
    if not (np.isclose(c["p50"], stats[name]["tc_med"], rtol=0, atol=1e-9)
            and np.isclose(c["p10"], stats[name]["tc_p10"], rtol=0, atol=1e-9)):
        raise SystemExit("[ERROR] fig_forest_tc.py: the constant path does not give back "
                         "the bound for %s (%.6f vs %.6f)" % (name, c["p50"], stats[name]["tc_med"]))

print("\ndeadline bound on three emission paths, from T0 = %d (years; inf = not reached by %d)"
      % (ep.T0, ep.LAST_YEAR))
for key, e in PATHS.items():
    if e is not None:
        print(f"  {key:7s} ({ep.SCENARIOS[key]}): E({ep.T0}) = {e[0]:.2f} GtCO2/yr against "
              f"E_0 = {E0_Gt:.0f}; cumulative {ep.T0}-{ep.LAST_YEAR} = {e.sum():.0f} GtCO2; "
              f"peak {e.max():.1f} GtCO2/yr in {ep.T0 + int(np.argmax(e))}")
for name in elements:
    for key in PATHS:
        d = path_stats[name][key]
        print(f"  {name:6s} {key:7s}: P10 {d['p10']:7.2f}  P50 {d['p50']:7.2f}  "
              f"P90 {d['p90']:7.2f}  not reached {100*d['unreached']:5.1f}%")

# The lower edge of the fixed-budget regime, eps_dyn ~ E^{2/3} (collect_computed.py), moves
# with the emission rate. Printed for the check of the appendix on where the elements sit, on each path: the factor by
# which the highest rate met before the median date raises eps_dyn above its E_0 value.
print("\nregime lower edge on each path: (E_max before the median date / E_0)^(2/3)")
for name in elements:
    for key, e in PATHS.items():
        if e is None:
            continue
        t50 = path_stats[name][key]["p50"]
        horizon = len(e) if not np.isfinite(t50) else int(np.ceil(t50))
        emax = float(e[:max(horizon, 1)].max())
        print(f"  {name:6s} {key:7s}: E_max = {emax:5.1f}  factor on eps_dyn = "
              f"{(emax/E0_Gt)**(2/3):.2f}; emissions positive up to the median date: "
              f"{bool((e[:max(horizon, 1)] > 0).all())}")

print("MACROS")
# floorShorteningMax is no longer emitted: the floor needs the amplitude A,
# evaluated here with sigma in the observable's units against M* in gigatonnes, i.e. with
# the physical margin set to one, which ssec:window declares unmeasured.
print(f"  fracWAISfirst = {round(f_wais_first*100)}")
print(f"  fracAmazonBeforeAMOC = {round(f_am_before_amoc*100)}")
print(f"  fracFullOrder = {round(f_full*100)}")
print(f"  fracWAISalreadyPct = {frac_wais_already*100:.1f}")
print(f"  fracWAISfirstIncl = {f_wais_first_incl*100:.1f}")
print(f"  fracWAISbeforeAmazon = {round(f_wais_before_am*100)}")
print(f"  fracWAISbeforeAmazonIncl = {f_wais_before_am_incl*100:.1f}")
print(f"  jointSamplePct = {joint_sample*100:.1f}")
# The P10 of the bound, the left end of the bar the figure draws: the same quantity as the
# plotted median (tc = mu_now/E0 over the draws with mu_now > 5), at the 10th centile.
# Conditional on t_c > 0 like the median; no floor enters it.
print(f"  tcAMOCPten = {round(stats['AMOC']['tc_p10'])}")
print(f"  tcAmazonPten = {round(stats['Amazon']['tc_p10'])}")
# The figure labels the P10 of every bar, so the third one is exposed too.
print(f"  tcWAISPten = {round(stats['WAIS']['tc_p10'])}")
# The bound on the two emission paths, same rounding as \tc<element> and its P10.
# A centile that falls among the draws the path never reaches is +inf and gets NO macro:
# the text says "not reached before \emissionPathLastYear" there, and a later run in which
# that centile becomes finite emits a macro the text does not call yet, which the per-macro
# diff reports as NEW rather than letting a number change under an unchanged sentence.
print(f"  emissionPathStartYear = {ep.T0}")
print(f"  emissionPathLastYear = {ep.LAST_YEAR}")
for key in ep.SCENARIOS:
    print(f"  EglobalRate{key} = {PATHS[key][0]:.1f}")
# The year from which SSP2-4.5 emissions stay at zero in its published extension
# (Meinshausen et al. 2020 ramp fossil CO2 down to zero by 2250), read off the data rather than
# typed, so the text can say why the path never reaches the Amazon's median budget.
_e_mid = PATHS["SSPmid"]
_nz = np.nonzero(_e_mid > 0)[0]
if len(_nz) == 0 or _nz[-1] + 1 >= len(_e_mid):
    raise SystemExit("[ERROR] fig_forest_tc.py: SSP2-4.5 emissions do not reach zero by %d"
                     % ep.LAST_YEAR)
print(f"  emissionPathSSPmidZeroYear = {ep.T0 + _nz[-1] + 1}")
for name in elements:
    for key in ep.SCENARIOS:
        d = path_stats[name][key]
        for suffix, q in (("", "p50"), ("Pten", "p10"), ("Pninety", "p90")):
            if np.isfinite(d[q]):
                print(f"  tc{name}{key}{suffix} = {round(d[q])}")
        print(f"  tc{name}{key}NotReachedPct = {100*d['unreached']:.1f}")
print("END MACROS")

# Every macro this figure declares must have been emitted above (a centile not reached has
# no macro and must not be declared). Checked against the same values the block printed.
_emitted = {"tc%s%s%s" % (n, k, s_) for n in elements for k in ep.SCENARIOS
            for s_, q in (("", "p50"), ("Pten", "p10"), ("Pninety", "p90"))
            if np.isfinite(path_stats[n][k][q])}
_emitted |= {"tc%s%sNotReachedPct" % (n, k) for n in elements for k in ep.SCENARIOS}
_emitted |= {"tc%s" % n for n in elements} | {"tc%sPten" % n for n in elements}
_emitted |= {"fracWAISalreadyPct"}
_missing = [m for m in FIGURE_MACROS if m not in _emitted]
if _missing:
    raise SystemExit("[ERROR] fig_forest_tc.py: FIGURE_MACROS declares %s, which this run does "
                     "not emit (a centile not reached on its path). Take it out of "
                     "FIGURE_MACROS; the figure draws it as an arrow." % ", ".join(_missing))

COL = {"WAIS":"#c0392b", "Amazon":"#1a9850", "AMOC":"#2c3e50"}
plt.rcParams.update({"font.family":"serif", "font.size":10, "mathtext.fontset":"cm"})

# Two elements only, the Amazon and the West Antarctic ice sheet; the overturning
# circulation's bounds are in the AMOC-price appendix (its loss has a contested sign, ssec:comparable).
# One colour per element, one marker and one intensity per emission path. Every number
# written on the figure is a macro printed above, with the same rounding.
SHOWN = ["WAIS", "Amazon"]                                   # most urgent on top
PATH_STYLE = [("now",     "today's rate held constant", "o", 1.00),
              ("SSPmid",  "SSP2-4.5",                   "s", 0.70),
              ("SSPhigh", "SSP3-7.0",                   "^", 0.45)]
NAME = {"WAIS": "West Antarctic\nice sheet", "Amazon": "Amazon"}
X_LO, X_HI = 1.0, 1000.0
fig, ax = plt.subplots(figsize=(6.4, 3.6))
yticks = []
# Every label is recorded with the row it belongs to, so that the placement can be
# checked once the layout is final (see RESOLVE below). The "not reached" label has a list of
# candidate places, tried in order; the first is its original place.
LABELS = []          # (Text, row id)
MOVABLE = []         # (Text, row id, [(x, y, va, ha), ...])
for g, name in enumerate(SHOWN):
    y0 = (len(SHOWN) - 1 - g) * 4.0
    yticks.append((y0, NAME[name]))
    for j, (key, _lab, mk, alpha) in enumerate(PATH_STYLE):
        d = path_stats[name][key]
        y = y0 + 1.0 - j
        # A centile the path does not reach before its last year is +inf. The bar then runs
        # to the edge of the axis and ends in an arrow (explained in the legend), for any
        # centile and any element. A FINITE centile beyond the axis would be drawn as if it
        # were not reached, so it stops the run instead.
        for q in ("p10", "p50", "p90"):
            if np.isfinite(d[q]) and d[q] > X_HI:
                raise SystemExit("[ERROR] fig_forest_tc.py: %s %s %s = %.0f yr lies beyond the "
                                 "axis (%.0f yr); widen X_HI" % (name, key, q, d[q], X_HI))
        reached_hi = np.isfinite(d["p90"])
        lo = max(d["p10"], X_LO) if np.isfinite(d["p10"]) else X_HI / 1.6
        hi = d["p90"] if reached_hi else X_HI
        if np.isfinite(d["p10"]):
            ax.hlines(y, lo, hi, color=COL[name], lw=4, alpha=0.40*alpha + 0.05, zorder=1)
        if not reached_hi:
            ax.annotate("", xy=(X_HI, y), xytext=(X_HI/1.6, y), zorder=2,
                        arrowprops=dict(arrowstyle="->", color=COL[name], alpha=0.8, lw=1.2))
        if np.isfinite(d["p50"]):
            ax.plot(d["p50"], y, mk, color=COL[name], alpha=alpha, ms=8, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)
            LABELS.append((ax.text(d["p50"], y + 0.28, f"{round(d['p50'])}", va="bottom",
                                   ha="center", fontsize=7, color="#333333"), (name, key)))
        # The P10, at the bar's left end, rounded like the macros.
        if np.isfinite(d["p10"]):
            LABELS.append((ax.text(lo/1.12, y, f"P10: {round(d['p10'])}", va="center",
                                   ha="right", fontsize=6.5, color="#333333"), (name, key)))
        # Shown when the share, at the macro's one decimal, is positive.
        if round(100*d["unreached"], 1) > 0:
            # First choice: under the bar when it ends in an arrow (it cannot meet the row's own
            # median label there), to its right otherwise. Second choice: above the bar, at its
            # right end. RESOLVE below moves a label only if the first choice touches another.
            if not reached_hi:
                cands = [(hi/1.6, y - 0.28, "top", "right"), (hi/1.6, y + 0.28, "bottom", "right")]
            else:
                cands = [(hi*1.12, y, "center", "left"), (hi, y + 0.28, "bottom", "right")]
            x0, y0_, va0, ha0 = cands[0]
            t = ax.text(x0, y0_, f"not reached in {100*d['unreached']:.1f}% of draws",
                        va=va0, ha=ha0, fontsize=6.5, color="#555555")
            LABELS.append((t, (name, key)))
            MOVABLE.append((t, (name, key), cands))
    if name == "WAIS":
        # The draws the bars exclude: threshold below present warming, bound zero on every
        # path. Same format as the fracWAISalreadyPct macro printed above.
        # Placed at the level of the block, under the element's name in the left
        # margin, since it concerns the three paths alike and not one bar.
        LABELS.append((ax.text(-0.02, y0 - 1.25,
                               f"already past in\n{stats['WAIS']['frac_already']*100:.1f}% of draws",
                               va="top", ha="right", fontsize=6.5, color="#555555",
                               transform=ax.get_yaxis_transform()),
                       (name, "note")))
ax.set_yticks([t for t, _ in yticks]); ax.set_yticklabels([l for _, l in yticks])
ax.tick_params(axis="y", length=0)
for t, name in zip(ax.get_yticklabels(), SHOWN):
    t.set_color(COL[name])
ax.set_xscale("log"); ax.set_xlim(X_LO/1.0, X_HI)
ax.set_ylim(-1.8, (len(SHOWN) - 1) * 4.0 + 1.8)
ax.set_xlabel(r"deadline bound $t_c$ (years, log scale)")
handles = [plt.Line2D([], [], ls="", marker=mk, color="#555555", alpha=a, ms=7, label=lab)
           for _k, lab, mk, a in PATH_STYLE]
# What the arrow means: the centile is not reached on that path before its last year.
handles.append(plt.Line2D([], [], ls="", marker=r"$\rightarrow$", color="#555555", ms=10,
                          label="centile not reached on the path"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
          frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.2)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()

# RESOLVE. The layout is final, so the labels' boxes on the canvas are too. A
# "not reached" label that touches any other label is moved to its next candidate place that
# touches none; if none is free the run stops. Then no two labels may touch at all: the rule
# is checked on the figure as drawn, not assumed from the offsets.
_renderer = fig.canvas.get_renderer()
_PAD = 1.0   # pixels


def _box(t):
    return t.get_window_extent(_renderer).expanded(1.0, 1.0).padded(_PAD)


def _touches(t):
    b = _box(t)
    return any(o is not t and b.overlaps(_box(o)) for o, _ in LABELS)


for t, row, cands in MOVABLE:
    if not _touches(t):
        continue
    for (x, yy, va, ha) in cands[1:]:
        t.set_position((x, yy)); t.set_va(va); t.set_ha(ha)
        if not _touches(t):
            sys.stderr.write(f"label of {row} moved to its next place, "
                             f"{'above' if va == 'bottom' else 'below or beside'} the bar\n")
            break
    else:
        raise SystemExit(f"[ERROR] fig_forest_tc.py: the 'not reached' label of {row} touches "
                         "another label in every candidate place")
for i, (t, row) in enumerate(LABELS):
    for o, orow in LABELS[i + 1:]:
        if _box(t).overlaps(_box(o)):
            raise SystemExit(f"[ERROR] fig_forest_tc.py: labels of {row} and {orow} touch")
fig.savefig("fig_forest_tc.pdf")
fig.savefig("fig_forest_tc.png", dpi=150)
print("saved fig_forest_tc.pdf/.png")
