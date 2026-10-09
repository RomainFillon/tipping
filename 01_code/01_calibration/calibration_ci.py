# The level A and the floor ε_∞ are not computed: in dollars per tonne they require η*, which no element identifies (the appendix on where the elements sit); the deadline bound eq:tc_bound drops the floor.
r"""
REPLICATION HEADER
  PRODUCES   the KEY NUMBERS block parsed into computed.json
  FEEDS      \MstarAMOCmed \MstarAmazonmed \MstarWAISmed, \OmegaAMOCmed \OmegaAmazonmed \OmegaWAISmed; tab:inputs
  INPUTS     inputs_literature.py (thresholds, TCRE, current warming); no file, no network
  SEED       numpy seed 42, N = 100,000 draws
  RUNTIME    3 to 6 min (three cold runs; the spread is machine load, not the work)
  IMPLEMENTS eq:Mstar_coupling and the proximity eps_now = (DT*-DT_0)/DT*

calibration_ci.py
The Monte Carlo of the thresholds, the budget M* and the realization factor of the calibration-inputs appendix.
An earlier version also computed a level of the tipping SCC from A, in mixed units; withdrawn
(see the line above).

Physical coupling:
  M*(GtCO2, pre-industrial to threshold) = ΔT* / TCRE × 1000      [eq:Mstar_coupling]
  mu_now (GtCO2, still unspent)          = max(0, ΔT* - ΔT_current) / TCRE × 1000
  eps_now = mu_now / M* = (ΔT* - ΔT_current)/ΔT*   is a fraction of M*, NOT of mu_now
  where:
    ΔT*      = temperature threshold for tipping (Armstrong McKay et al. 2022)
    ΔT_current = 1.2°C current warming above pre-industrial (IPCC AR6 WG1 2021)
    TCRE     = transient climate response to cumulative emissions (IPCC AR6)
              = 0.45°C per 1000 GtCO2, likely (17-83%) range [0.27, 0.63]°C/1000 GtCO2
                (SPM D.1.1); drawn by tcre_draw.draw_tcre, see there

σ: from EWS (Caesar 2018 for AMOC); literature prior for Amazon/WAIS.
L: from damage literature (Armstrong McKay 2022, Lenton 2008).

References:
  Armstrong McKay et al. (2022) Nature Climate Change — Table 1 temperature thresholds
  IPCC AR6 WG1 (2021) — TCRE, current warming
  Caesar et al. (2018) Nature — AMOC SST fingerprint (σ calibration)
  Rahmstorf (1995, 1996) Tellus — box model fold structure of AMOC
"""

import os
import sys
import numpy as np
from scipy.integrate import solve_ivp

# Phi and the realization factor Omega(rho*T) live in phi_fold.py so that this script and reduced_form_scc.py
# share ONE definition of the amplitude (see phi_fold.py docstring).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phi_fold import realization_factor  # noqa: E402

# Lmax is a SOURCED constant (Dietz et al. 2021) and lives in inputs_literature.py, which
# is also what values.tex reads. It used to be written out as 0.67 in three element dicts
# here and three more in eps_inf_ci.py, none of them aware of the others — the exact
# configuration that let a superseded M* survive inside a hard-coded eps_dyn.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs_literature import LITERATURE_MACROS, DERIVATION_INPUTS  # noqa: E402
from tcre_draw import draw_tcre  # noqa: E402

L_MAX = float(LITERATURE_MACROS["Lmax"][0])   # upper end of the common damage sweep


def _lit(key):
    """A sourced constant, by name. A missing key is fatal and named: no default here."""
    if key not in DERIVATION_INPUTS:
        raise SystemExit("[ERROR] calibration_ci.py: DERIVATION_INPUTS has no '%s'. "
                         "Declare it with its source rather than writing it out here."
                         % key)
    return float(DERIVATION_INPUTS[key][0])


def _DTstar(element):
    """(central, low end, high end) of an element's threshold, from the one sourced home.

    The same nine numbers used to be written out here, in eps_inf_ci.py and in
    fig_forest_tc.py, none of the three aware of the others — the same configuration the
    Lmax note above describes. They are read now, so a revision of Armstrong McKay moves
    the three scripts together or none of them.
    """
    return tuple(_lit("DTstar_%s_%s" % (element, k)) for k in ("best", "p5", "p95"))

np.random.seed(42)
N = int(DERIVATION_INPUTS["N_MC"][0])   # Monte Carlo draws

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants and calibration
# ─────────────────────────────────────────────────────────────────────────────
rho         = 0.03                 # discount rate
DT_current  = _lit("DT_current")   # °C current warming above pre-industrial (Forster et al. 2024)

# ─────────────────────────────────────────────────────────────────────────────
# LogNormal fitting: fit (P5, best, P95) to a lognormal distribution
# ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import norm as spnorm

def fit_lognormal(p5, best, p95):
    """Return (mu_ln, sigma_ln) for a lognormal with given percentiles."""
    # P50 ≈ best (use as median), fit sigma from P5/P95
    mu_ln = np.log(best)
    sigma_ln = (np.log(p95) - np.log(p5)) / (2 * spnorm.ppf(0.95))
    return mu_ln, sigma_ln

def sample_lognormal(p5, best, p95, n, lower=0.0, upper=np.inf):
    """Draw n samples from lognormal with truncation.
    Kept for the *estimated* priors that stay lognormal: DT* and T (and AMOC sigma).
    Do NOT route L or the swept sigmas through this — use the sweeps below."""
    mu, sig = fit_lognormal(p5, best, p95)
    raw = np.exp(np.random.normal(mu, sig, n))
    return np.clip(raw, lower, upper)

# ─────────────────────────────────────────────────────────────────────────────
# Swept priors for L and sigma (robustness bounds, not fitted lognormals).
#   L/GDP  : UNIFORM on [lo, hi], floored at 0 (eps_inf ~ L^2, so L<0 would read as
#            large damage; the floor makes a net-benefit finding enter as zero).
#   sigma  : LOG-uniform on [lo, hi] (sigma enters the SCC as sigma^{2/3}, so the
#            economically neutral sweep is uniform in log sigma, not in sigma).
# ─────────────────────────────────────────────────────────────────────────────
def sample_uniform(lo, hi, n, floor=0.0):
    """Uniform sweep on [lo, hi], clipped from below at `floor`."""
    return np.clip(np.random.uniform(lo, hi, n), floor, None)

def sample_loguniform(lo, hi, n):
    """Log-uniform sweep on [lo, hi] (uniform in log sigma)."""
    return np.exp(np.random.uniform(np.log(lo), np.log(hi), n))

def draw_L(p, n):
    """L/GDP prior: uniform robustness sweep, floored at 0."""
    return sample_uniform(p["L_lo"], p["L_hi"], n, floor=0.0)

def draw_sigma(p, n):
    """sigma prior: EWS lognormal for AMOC (the only sigma estimated on data);
    log-uniform robustness sweep (sigma_lo/sigma_hi) for the other elements."""
    if "sigma_lo" in p:
        return sample_loguniform(p["sigma_lo"], p["sigma_hi"], n)
    return np.clip(np.random.lognormal(np.log(p["sigma_mu"]), p["sigma_cv"], n), 0.02, 2.0)

# ─────────────────────────────────────────────────────────────────────────────
# TCRE distribution (IPCC AR6, SPM D.1.1): normal, mean TCRE, sd from the likely range
# read as the 17th-83rd centiles; draws <= 0 discarded by rejection (tcre_draw.py).
# ─────────────────────────────────────────────────────────────────────────────
TCRE_draws, (_tcre_disc, _tcre_tot) = draw_tcre(np.random, N, return_discarded=True)   # °C/1000 GtCO2
# The share of the draws the rejection discards, read by collect_computed.py (\TCREdiscardPct).
print(f"TCRE DRAW: {N} kept, {_tcre_disc} discarded (<= 0) of {_tcre_tot} drawn, "
      f"discarded share = {100 * _tcre_disc / _tcre_tot:.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# M* derivation from physical coupling:
#   M*[GtCO2] = ΔT* / TCRE × 1000 ; mu_now = max(0, ΔT*-ΔT_cur)/TCRE × 1000
# ─────────────────────────────────────────────────────────────────────────────
def draw_Mstar(DT_best, DT_p5, DT_p95, TCRE_draws):
    """
    Derive the M* and mu_now distributions from ΔT* uncertainty + TCRE uncertainty.
    DT values in °C above pre-industrial (from Armstrong McKay 2022).

    M* is the budget from PRE-INDUSTRIAL to the threshold, eq:Mstar_coupling of the paper,
    M* = DT*/TCRE x 1000, because the paper's proximity eps_now = (DT* - DT_0)/DT* is the
    fraction of THAT budget still unspent. The part still unspent is mu_now = M* eps_now,
    which is the remaining budget this function used to return under the name M*. Returning
    the remaining budget while eps_now is a fraction of the total applies eps_now twice.
    """
    DT_draws = sample_lognormal(DT_p5, DT_best, DT_p95, len(TCRE_draws), lower=0.1)
    remaining_DT = np.maximum(0.0, DT_draws - DT_current)
    M_star = DT_draws / TCRE_draws * 1000.0       # GtCO2, pre-industrial to threshold
    mu_now = remaining_DT / TCRE_draws * 1000.0   # GtCO2, still unspent today
    return M_star, mu_now

# ─────────────────────────────────────────────────────────────────────────────
# Tipping element parameter distributions
# ─────────────────────────────────────────────────────────────────────────────

elements = {
    "AMOC": {
        # Temperature threshold: Armstrong McKay (2022) Table 1, AMOC collapse
        # central 4°C, range 1.4-8°C above pre-industrial — read from inputs_literature.py
        "DT_best": _DTstar("AMOC")[0], "DT_p5": _DTstar("AMOC")[1], "DT_p95": _DTstar("AMOC")[2],
        # Realization timescale (AM 2022 Table 1): central 50 yr, range 15-300 yr
        # (aligns the code sampler with \TAMOC=50 in values.tex; previously 100 [50,300])
        "T_best": float(LITERATURE_MACROS["TAMOC"][0]), "T_p5": DERIVATION_INPUTS["T_AMOC_p5"][0], "T_p95": DERIVATION_INPUTS["T_AMOC_p95"][0],
        # σ_clim: from EWS (Caesar 2018 HadISST), with ±30% uncertainty.
        # AMOC keeps the *estimated* lognormal prior — the only sigma from data.
        "sigma_mu": _lit("sigma_mu_AMOC"), "sigma_cv": _lit("sigma_cv_AMOC"),
        # L/GDP: UNIFORM robustness sweep, floored at 0 (net-benefit → 0 damage).
        "L_lo": 0.0, "L_hi": L_MAX,
        # Current table values (for comparison)
        "M_star_table": 1000, "sigma_table": 0.20, "L_table": 0.20,
    },
    "Amazon": {
        # Armstrong McKay (2022) Table 1: 3.5°C [2.0, 6.0] — from inputs_literature.py
        "DT_best": _DTstar("Amazon")[0], "DT_p5": _DTstar("Amazon")[1], "DT_p95": _DTstar("Amazon")[2],
        # Timescale (AM 2022): central 100 yr, range 50-200 yr
        "T_best": float(LITERATURE_MACROS["TAmazon"][0]), "T_p5": DERIVATION_INPUTS["T_Amazon_p5"][0], "T_p95": DERIVATION_INPUTS["T_Amazon_p95"][0],
        # sigma: the EFFECTIVE noise of the VOD dynamics (level 2), NOT commensurable with
        # sigma_AMOC, which is in kelvin. Boulton (2022) constrains lambda_Amazon at about
        # 9.7/yr (1/kappa about 1.24 months); Var(X) is not extracted from VOD here, since
        # VODCA is not processed in this package. The log-uniform prior is kept, in units of
        # VOD yr^{-1/2}.
        "sigma_lo": DERIVATION_INPUTS["sigma_prior_lo"][0], "sigma_hi": DERIVATION_INPUTS["sigma_prior_hi"][0],
        # L/GDP: a prior COMMON to the three elements, UNIFORM on [0, Lmax], floored at 0.
        # Differences in the price then come from geometry, proximity and timescale, and
        # never from a damage figure chosen per element.
        "L_lo": 0.0, "L_hi": L_MAX,
        "M_star_table": 800, "sigma_table": 0.15, "L_table": 0.10,
    },
    "WAIS": {
        # Armstrong McKay (2022) Table 1: 1.5°C [1.0, 3.0] — from inputs_literature.py
        "DT_best": _DTstar("WAIS")[0], "DT_p5": _DTstar("WAIS")[1], "DT_p95": _DTstar("WAIS")[2],
        # Timescale (AM 2022): millennial — central 2000 yr, range 500-13000 yr
        "T_best": float(LITERATURE_MACROS["TWAIS"][0]), "T_p5": DERIVATION_INPUTS["T_WAIS_p5"][0], "T_p95": DERIVATION_INPUTS["T_WAIS_p95"][0],
        # sigma: EPISTEMIC uncertainty (level 3), NOT calibrated on data. Routing it through
        # the ocean forcing, sigma_X = (dX/dF) sigma_F, is deferred: dX/dF is model-dependent
        # and identifying it is a separate exercise.
        # Nothing in the record distinguishes sigma_WAIS from sigma_Amazon, so the prior is
        # STRICTLY common to the two, [0.05, 0.45]. The centre 0.15 is the choice that
        # perturbs the total least, the Amazon dominating it and being unchanged; the
        # element ranking and the deadlines t_c are invariant to it.
        "sigma_lo": DERIVATION_INPUTS["sigma_prior_lo"][0], "sigma_hi": DERIVATION_INPUTS["sigma_prior_hi"][0],
        # L/GDP: the same COMMON prior, UNIFORM on [0, Lmax], floored at 0.
        "L_lo": 0.0, "L_hi": L_MAX,
        "M_star_table": 1500, "sigma_table": 0.10, "L_table": 0.42,
    },
}

eps_nat = {"AMOC": 0.70, "Amazon": 0.66, "WAIS": 0.20}  # current (natural) proximity per element

# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo CI computation
# ─────────────────────────────────────────────────────────────────────────────

print("="*72)
print("CALIBRATION — thresholds, budget M*, realization factor (Monte Carlo)")
print("Physical coupling: M*(GtCO2) = max(0, (ΔT*-1.2°C)/TCRE) × 1000")
print("References: Armstrong McKay 2022 (ΔT*), IPCC AR6 (TCRE), Caesar 2018 (σ)")
print("="*72)

# Spot-check: Φ_fold at typical AMOC ρ̃ values (numerical inner ODE)

for name, p in elements.items():
    print(f"\n{'─'*60}")
    print(f"Element: {name}")

    # --- Draw parameter samples ---
    M_star_draws, mu_now_draws = draw_Mstar(p["DT_best"], p["DT_p5"], p["DT_p95"], TCRE_draws)
    sigma_draws  = draw_sigma(p, N)
    L_draws      = draw_L(p, N)
    T_draws = sample_lognormal(p["T_p5"], p["T_best"], p["T_p95"], N, lower=1.0)
    f_draws = realization_factor(rho, T_draws)

    # Filter: only valid M* > 0 (not already crossed threshold)
    valid = mu_now_draws > 5.0   # require at least 5 GtCO2 REMAINING budget (unchanged sample)
    frac_valid = valid.mean()
    print(f"  Fraction of draws with M* > 5 GtCO2: {frac_valid:.2%}")
    print(f"  M* (GtCO2): P10={np.percentile(M_star_draws[valid], 10):.0f}  "
          f"P50={np.percentile(M_star_draws[valid], 50):.0f}  "
          f"P90={np.percentile(M_star_draws[valid], 90):.0f}")
    print(f"  σ (K/√yr): P10={np.percentile(sigma_draws[valid], 10):.3f}  "
          f"P50={np.percentile(sigma_draws[valid], 50):.3f}  "
          f"P90={np.percentile(sigma_draws[valid], 90):.3f}")
    print(f"  L/GDP:     P10={np.percentile(L_draws[valid], 10):.2f}  "
          f"P50={np.percentile(L_draws[valid], 50):.2f}  "
          f"P90={np.percentile(L_draws[valid], 90):.2f}")
    print(f"  T (yr):    P10={np.percentile(T_draws[valid], 10):.0f}  "
          f"P50={np.percentile(T_draws[valid], 50):.0f}  "
          f"P90={np.percentile(T_draws[valid], 90):.0f}")
    print(f"  f(ρT):     P10={np.percentile(f_draws[valid], 10):.3f}  "
          f"P50={np.percentile(f_draws[valid], 50):.3f}  "
          f"P90={np.percentile(f_draws[valid], 90):.3f}")

# The aggregate SCC numbers, the variance figure, the LaTeX table and the AMOC
# fan-chart figure are withdrawn with the level A. The AMOC draws that the figure made are
# kept, in the same order, so that the KEY NUMBERS draws below are unchanged.
p = elements["AMOC"]
M_star_draws, mu_now_draws = draw_Mstar(p["DT_best"], p["DT_p5"], p["DT_p95"], TCRE_draws)
sigma_draws  = draw_sigma(p, N)
L_draws      = draw_L(p, N)
T_draws      = sample_lognormal(p["T_p5"], p["T_best"], p["T_p95"], N, lower=1.0)
valid        = mu_now_draws > 5.0

# ─────────────────────────────────────────────────────────────────────────────
# Summary of key numbers for paper
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("KEY NUMBERS FOR PAPER")
print("="*72)
for name in elements:
    p_e = elements[name]
    M_star_draws, mu_now_draws = draw_Mstar(p_e["DT_best"], p_e["DT_p5"], p_e["DT_p95"], TCRE_draws)
    sigma_draws  = draw_sigma(p_e, N)
    L_draws      = draw_L(p_e, N)
    T_draws      = sample_lognormal(p_e["T_p5"], p_e["T_best"], p_e["T_p95"], N, lower=1.0)
    valid        = mu_now_draws > 5.0
    Mstar_cen = np.percentile(M_star_draws[valid], 50)
    T_cen     = np.percentile(T_draws[valid], 50)
    f_cen     = realization_factor(rho, T_cen)
    print(f"{name}: M*_P50={Mstar_cen:.0f} GtCO2 | T_P50={T_cen:.0f} yr | f(ρT)={f_cen:.3f}")
