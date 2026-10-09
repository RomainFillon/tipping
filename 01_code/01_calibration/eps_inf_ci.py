# The level A and the floor ε_∞ are not computed: in dollars per tonne they require η*, which no element identifies (the appendix on where the elements sit); the deadline bound eq:tc_bound drops the floor.
"""
REPLICATION HEADER
  PRODUCES   the deadline-bound block parsed into computed.json
  FEEDS      \tcAMOC \tcAmazon \tcWAIS
  INPUTS     inputs_literature.py; tcre_draw.py; no network
  SEED       numpy seed 42, N = 100,000 draws
  RUNTIME    under a minute
  IMPLEMENTS eq:tc_bound (the reported deadline)

eps_inf_ci.py
The deadline bound t_c <= mu_now/E0 of eq:tc_bound, at the median of the Monte Carlo of the calibration-inputs appendix.
An earlier version of this script also computed the level A and the floor eps_inf = (A/c_bar)^2,
with A in mixed units (a budget in GtCO2 paired with a noise in the element's own units);
both are withdrawn (see the line above). The name of the script is kept for the pipeline.
"""

import os
import sys
import numpy as np
from scipy.stats import norm as spnorm

# Lmax is sourced (Dietz et al. 2021) and declared once in inputs_literature.py, which is
# also what values.tex reads; see the same import in calibration_ci.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs_literature import LITERATURE_MACROS, DERIVATION_INPUTS  # noqa: E402
from tcre_draw import draw_tcre  # noqa: E402

L_MAX = float(LITERATURE_MACROS["Lmax"][0])   # 0.67, upper end of the common damage sweep


def _lit(key):
    """A sourced constant, by name. A missing key is fatal and named: no default here."""
    if key not in DERIVATION_INPUTS:
        raise SystemExit("[ERROR] eps_inf_ci.py: DERIVATION_INPUTS has no '%s'. Declare "
                         "it with its source rather than writing it out here." % key)
    return float(DERIVATION_INPUTS[key][0])


def _DTstar(element):
    """(central, low end, high end) of an element's threshold, from the one sourced home.
    Same reason as the Lmax note above: these nine numbers had three copies."""
    return tuple(_lit("DTstar_%s_%s" % (element, k)) for k in ("best", "p5", "p95"))


np.random.seed(42)
N = int(DERIVATION_INPUTS["N_MC"][0])

# rho, GDP, E0 (as $), the backstop costs and Phi_fold served only the level A.
DT_current  = _lit("DT_current")

# ── Helpers ──────────────────────────────────────────────────────────────────
def fit_lognormal(p5, best, p95):
    mu_ln = np.log(best)
    sigma_ln = (np.log(p95) - np.log(p5)) / (2 * spnorm.ppf(0.95))
    return mu_ln, sigma_ln

def sample_lognormal(p5, best, p95, n, lower=0.0, upper=np.inf):
    # Estimated priors that stay lognormal: DT* (and AMOC sigma). Not L, not swept sigma.
    mu, sig = fit_lognormal(p5, best, p95)
    raw = np.exp(np.random.normal(mu, sig, n))
    return np.clip(raw, lower, upper)

# Swept priors (kept identical to calibration_ci.py so SCC and eps_inf/t_c stay in sync):
#   L/GDP  : uniform on [lo, hi], floored at 0 (eps_inf ~ L^2).
#   sigma  : log-uniform on [lo, hi] (sigma enters as sigma^{2/3}); AMOC keeps EWS lognormal.
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

def draw_Mstar(DT_best, DT_p5, DT_p95, TCRE_draws):
    """(M*, mu_now) in GtCO2.

    M* is the budget from PRE-INDUSTRIAL to the threshold, eq:Mstar_coupling, because the
    proximity eps_now = (DT* - DT_0)/DT* is the fraction of THAT budget still unspent.
    mu_now = M* eps_now is the part still unspent. Returning mu_now under the name M*, as
    this function used to, applies eps_now twice wherever the two are multiplied.
    """
    DT_draws = sample_lognormal(DT_p5, DT_best, DT_p95, len(TCRE_draws), lower=0.1)
    remaining_DT = np.maximum(0.0, DT_draws - DT_current)
    return (DT_draws / TCRE_draws * 1000.0,
            remaining_DT / TCRE_draws * 1000.0)

TCRE_draws = draw_tcre(np.random, N)   # °C/1000 GtCO2 (tcre_draw.py)

elements = {
    "AMOC": {
        # AMOC central threshold (Armstrong McKay 2022): 4.0°C [1.4, 8.0]
        # (previously the SPG-proxy 1.8°C leading-indicator threshold; now central,
        #  matching calibration_ci.py — the EWS near-threshold reading is in App. A)
        "DT_best": _DTstar("AMOC")[0], "DT_p5": _DTstar("AMOC")[1], "DT_p95": _DTstar("AMOC")[2],
        # AMOC keeps the estimated EWS lognormal sigma (only sigma from data)
        "sigma_mu": _lit("sigma_mu_AMOC"), "sigma_cv": _lit("sigma_cv_AMOC"),
        # L/GDP: uniform robustness sweep, floored at 0
        "L_lo": 0.0, "L_hi": L_MAX,
    },
    "Amazon": {
        "DT_best": _DTstar("Amazon")[0], "DT_p5": _DTstar("Amazon")[1], "DT_p95": _DTstar("Amazon")[2],
        # sigma: the EFFECTIVE noise of the VOD dynamics (level 2), NOT commensurable with
        # sigma_AMOC, which is in kelvin. Boulton (2022) gives lambda_Amazon about 9.7/yr
        # (1/kappa about 1.24 months); Var(X) is not extracted from VOD. Units VOD yr^{-1/2}.
        "sigma_lo": DERIVATION_INPUTS["sigma_prior_lo"][0], "sigma_hi": DERIVATION_INPUTS["sigma_prior_hi"][0],
        "L_lo": 0.0, "L_hi": L_MAX,           # prior COMMON to the three, [0, Lmax], floored at 0
    },
    "WAIS": {
        "DT_best": _DTstar("WAIS")[0], "DT_p5": _DTstar("WAIS")[1], "DT_p95": _DTstar("WAIS")[2],
        # sigma: EPISTEMIC uncertainty (level 3), NOT calibrated. Routing it through the
        # ocean forcing, sigma_X = (dX/dF) sigma_F, is deferred: dX/dF is model-dependent.
        # The prior is STRICTLY common with the Amazon, [0.05, 0.45]; see calibration_ci.py.
        "sigma_lo": DERIVATION_INPUTS["sigma_prior_lo"][0], "sigma_hi": DERIVATION_INPUTS["sigma_prior_hi"][0],
        "L_lo": 0.0, "L_hi": L_MAX,           # prior COMMON to the three, [0, Lmax], floored at 0
    },
}

print("=" * 72)
print("DEADLINE BOUND t_c <= (M*_med/E0) eps_0 = mu_now_med/E0 — the Monte Carlo of sec:calibration")
print("=" * 72)

M_med_store = {}   # median M* (GtCO2, pre-industrial to threshold)
MU_med_store = {}  # median mu_now (GtCO2, still unspent) - what t_c runs on

for name, p in elements.items():
    print(f"\n{'─'*60}\nElement: {name}")

    M_star_draws, mu_now_draws = draw_Mstar(p["DT_best"], p["DT_p5"], p["DT_p95"], TCRE_draws)
    # sigma and L entered only the level A, which is no longer computed. They are
    # still drawn, in the same order, so that the random stream -- and with it the draws of
    # the next element -- is the one every reported number was computed on.
    sigma_draws  = draw_sigma(p, N)
    L_draws      = draw_L(p, N)

    # filter on the REMAINING budget, so the conditioning (element not yet tipped) is
    # exactly the one used before the M* pairing was corrected
    valid = mu_now_draws > 5.0
    M_v = M_star_draws[valid]
    mu_v = mu_now_draws[valid]
    M_med_store[name] = float(np.median(M_v))
    MU_med_store[name] = float(np.median(mu_v))
    print(f"  valid draws (mu_now > 5 GtCO2): {valid.mean():.2%}")

# ── Deadline bound ─────────────────────────────────────────────────────────────
eps0_current = {"AMOC": 0.70, "Amazon": 0.66, "WAIS": 0.20}  # current proximity (AMOC central)
print("\n" + "=" * 72)
print("Deadline BOUND t_c <= (M*_med/E0) eps_0 = mu_now_med/E0   [what the paper reports]")
print("On the bound (M*/E0) eps_0 = mu_now/E0 = (DT*-DT_0)x1000/(TCRE x E0): the remaining")
print("warming over TCRE over the emission rate, with no unmeasured quantity in it.")
print("eps_0: AMOC 0.70 (central AM2022 threshold), Amazon 0.66, WAIS 0.20; E0=40 GtCO2/yr")
print("=" * 72)
for _n in elements:
    print(f"  {_n}: t_c_bound={MU_med_store[_n] / 40.0:.2f}yr "
          f"(M*_med={M_med_store[_n]:.0f}, mu_now_med={MU_med_store[_n]:.0f}, "
          f"eps0={eps0_current[_n]:.2f})")
