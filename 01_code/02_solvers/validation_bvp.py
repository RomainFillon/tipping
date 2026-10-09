"""
REPLICATION HEADER
  PRODUCES   nothing on disk; the tables below and a MACROS block of "VALIDATION name = value"
             lines, parsed into computed.json
  FEEDS      the \val* macros (the numerical-methods appendix, app:solver), through collect_computed.py
  INPUTS     sections 1-4 none; section 5 reads theta and M* through eps_dyn.py (the ews and
             calibration captures). Imports the solvers of paper2_all_bifurcations.py, the price
             of price.py and the amplitude of 01_calibration/phi_fold.py (imported, not copied)
  SEED       none (deterministic)
  RUNTIME    about 60 s (printed at the end of the run)
  IMPLEMENTS the numerical validation of the numerical-methods appendix: collapse of the amplitude (H2), the
             slope of the fold price, the error of the leading law, pitchfork, transcritical
             and Hopf

validation_bvp.py -- the numerical validation of the solvers; the price at fixed state.

CONVENTIONS. M* = 1 and L = 1, so mu = eps. phi is the BVP solution of eq:fk_body,
paper2_all_bifurcations.solve_phi_at_xstar, at solve_bvp tol = 1e-8: every case converges
there (a case that does not raises), and the values agree with tol = 1e-10 to six digits
wherever 1e-10 converges (checked once).

THE PRICE. Equation eq:gen_scc prices at the CURRENT state, so the price is the derivative
of phi(x; eps) at FIXED x = x*(eps), computed by price.price_fixed, the one implementation:
  P(eps) = -[phi(x; eps e^d) - phi(x; eps e^-d)] / [eps (e^d - e^-d)],   x = x*(eps),  d = 0.02.
The derivative along the stable state, -d phi(x*(eps); eps)/d eps, which an earlier version of
this script used, is printed as a COMPARISON COLUMN only ("total"); no macro is computed from it.
  local slope(eps) = [log P(eps e^D) - log P(eps e^-D)] / (2 D),   D = D_SLOPE.

THE AMPLITUDE. The amplitude-to-price appendix gives 1 - phi(x*) = 2 Phi_fold eta* (1 - Phi_fold eta* + ...) with
eta* = sqrt(mu)/sigma^(2/3), so sigma^(2/3) (1 - phi(x*))/sqrt(mu) tends to 2 Phi_fold(rho~),
TWICE the amplitude. Section 1 extracts that limit and compares it with 2 Phi_fold.

THE LEADING LAW at fixed state is kappa |x_-'(mu)|, kappa = Phi_fold(rho~)/l (price.py): for the
fold, Phi/(2 l sqrt(eps)), half the law along the stable state.
"""
import os
import sys
import time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_calibration"))
import paper2_all_bifurcations as p2          # noqa: E402  (the solvers, imported)
from price import price_fixed, fold_lead_price, D_PRICE   # noqa: E402  (the one price, eq:gen_scc)
from phi_fold import phi_true_scalar           # noqa: E402  (Phi_fold, app:amp)
sys.path.insert(0, os.path.dirname(HERE))
from inputs_literature import LITERATURE_MACROS  # noqa: E402  (rhoBase, for section 5)

T_START = time.time()

# ── settings (choices of this script, emitted as macros where the text quotes them) ─────
TOL = 1e-8                       # solve_bvp tolerance (see the docstring)
MAX_NODES = 200000               # enough for every case; a failure raises
D_SLOPE = 0.05                   # log-step of the local slope
SIGMAS = (0.5, 1.0, 2.0)         # robustness_all grid of paper2_all_bifurcations.py
RHOS = (0.02, 0.05, 0.10)
SLOPE_EPS = (1e-2, 1e-3, 1e-4)   # where the local slopes are read
RT_COLLAPSE = (0.05, 0.2, 1.0)   # rho~ of the collapse test
SIGMAS_COLLAPSE = (0.5, 1.0, 2.0, 4.0)
MU_COLLAPSE = (1e-4, 1e-5, 1e-6)  # the limit mu -> 0 is extrapolated linearly in sqrt(mu)
EPS_WIDE, EPS_NARROW, EPS_MIN = 0.05, 0.01, 1e-4
EPS_ERR = np.geomspace(EPS_MIN, EPS_WIDE, 30, endpoint=False)   # [1e-4, 0.05)
EPS_TC = (1e-4, 3e-4, 1e-3)      # transcritical: behaviour as eps -> 0

GEOMETRY = {  # drift, stable state, boundary, right end of the domain (as in paper2), exponent of l
    "fold":          (p2.drift_fold, p2.xstar_fold, p2.xleft_fold, 6.0, 5.0, 2 / 3),
    "pitchfork":     (p2.drift_pitchfork, p2.xstar_pitchfork, lambda mu: 0.0, 7.0, 5.0, 1 / 2),
    "transcritical": (p2.drift_transcritical, p2.xstar_transcritical, lambda mu: 0.0, 8.0, 4.0, 2 / 3),
}
# THE DOMAIN. paper2 cuts the domain at an ABSOLUTE x_right (5 for the fold, 4 for the
# transcritical) and imposes phi = 0 there. That is the condition of a state too far to come
# back, and it is only harmless if x_right is many noise scales l = sigma^(2/(k+1)) away: the
# cut reaches the stable state through diffusion against the drift, damped like
# exp(-(2/(k+1)) (x/l)^(k+1)). At sigma = 4 the fold's x_right = 5 is 2 l, and the collapse of
# section 1 is off by 31%; at sigma = 2 the transcritical's x_right = 4 is 2.5 l, and its limit is
# off by 1%. Here x_right is at least X_RIGHT_L noise scales, never less than paper2's default.
X_RIGHT_L = 8.0


def phi_at(geo, mu, sigma, rho, x, x_right_l=X_RIGHT_L):
    """phi(x; mu) at a GIVEN state x."""
    drift, xstar, xleft, mult, absr, a = GEOMETRY[geo]
    absr = max(absr, x_right_l * sigma ** a)
    xl = xleft(mu)
    if not x > xl:
        raise SystemExit(f"[ERROR] validation_bvp.py: state {x} is not inside the domain "
                         f"(boundary {xl}) at mu = {mu}")
    return p2.solve_phi_at_xstar(mu, sigma, rho, drift, xl, lambda m: x, mult, absr,
                                 tol=TOL, max_nodes=MAX_NODES)


def phi(geo, mu, sigma, rho):
    """phi at the stable state of the same budget, phi(x*(mu); mu)."""
    return phi_at(geo, mu, sigma, rho, float(GEOMETRY[geo][1](mu)))


def price(geo, eps, sigma, rho):
    """The price at fixed state, eq:gen_scc, through price.price_fixed."""
    return price_fixed(lambda m, x: phi_at(geo, m, sigma, rho, x), eps, float(GEOMETRY[geo][1](eps)))


def price_total(geo, eps, sigma, rho, d=D_PRICE):
    """COMPARISON ONLY: the derivative along the stable state, used by an earlier version."""
    hi, lo = phi(geo, eps * np.exp(d), sigma, rho), phi(geo, eps * np.exp(-d), sigma, rho)
    return -(hi - lo) / (eps * (np.exp(d) - np.exp(-d)))


def local_slope(geo, eps, sigma, rho, pricefn=price, D=D_SLOPE):
    return (np.log(pricefn(geo, eps * np.exp(D), sigma, rho))
            - np.log(pricefn(geo, eps * np.exp(-D), sigma, rho))) / (2 * D)


def fmt_sig(x, n=2):
    """n significant figures, plain decimal notation (no exponent), for a macro."""
    if x == 0:
        return "0"
    dec = max(0, n - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{dec}f}"


MACROS = {}

# ── 1. Collapse in sigma^(2/3) ────────────────────────────────────────────────────────
print("=" * 78)
print("1. COLLAPSE: q(mu) = sigma^(2/3) (1 - phi(x*))/sqrt(mu), fold, rho = rho~ sigma^(2/3)")
print("   limit mu -> 0 by a linear fit of q against sqrt(mu) on mu =", MU_COLLAPSE)
print("=" * 78)
print(f"{'rho~':>5} {'sigma':>5} " + " ".join(f"{'q(%g)' % m:>10}" for m in MU_COLLAPSE)
      + f" {'limit':>10} {'2 Phi_fold':>10} {'limit/2Phi-1':>13}")
gap_max = 0.0
for k, rt in enumerate(RT_COLLAPSE):
    two_phi = 2.0 * phi_true_scalar(rt)
    limits = []
    for sig in SIGMAS_COLLAPSE:
        rho = rt * sig ** (2 / 3)
        q = [sig ** (2 / 3) * (1 - phi("fold", m, sig, rho)) / np.sqrt(m) for m in MU_COLLAPSE]
        lim = np.polyfit(np.sqrt(MU_COLLAPSE), q, 1)[1]
        limits.append(lim)
        gap_max = max(gap_max, abs(lim / two_phi - 1))
        print(f"{rt:5.2f} {sig:5.2f} " + " ".join(f"{v:10.6f}" for v in q)
              + f" {lim:10.6f} {two_phi:10.6f} {lim / two_phi - 1:+13.2e}")
    limits = np.array(limits)
    dev = float(np.max(np.abs(limits / limits.mean() - 1)))
    print(f"   rho~ = {rt}: largest relative deviation of the limit from its mean over sigma "
          f"= {100 * dev:.2e} %")
    MACROS["valCollapseRhoTilde" + "ABC"[k]] = f"{rt:g}"
    MACROS["valCollapseDevPct" + "ABC"[k]] = fmt_sig(100 * dev)
print(f"   largest |limit / (2 Phi_fold) - 1| on the grid = {100 * gap_max:.2e} %")
MACROS["valCollapseGapPct"] = fmt_sig(100 * gap_max)

# ── 2. Slope of the fold price ────────────────────────────────────────────────────────
print()
print("=" * 78)
print("2. FOLD: local slope of log P against log eps, P at fixed state (eq:gen_scc)")
print("   last column: the same slope for the derivative along x*(eps), for comparison only")
print("=" * 78)
print(f"{'sigma':>5} {'rho':>5} {'rho~':>6} " + " ".join(f"{'eps=%g' % e:>10}" for e in SLOPE_EPS)
      + f" {'total 1e-4':>11}")
fold_slopes = {}
for sig in SIGMAS:
    for rho in RHOS:
        sl = [local_slope("fold", e, sig, rho) for e in SLOPE_EPS]
        fold_slopes[(sig, rho)] = sl
        tot = local_slope("fold", SLOPE_EPS[-1], sig, rho, pricefn=price_total)
        print(f"{sig:5.1f} {rho:5.2f} {rho / sig ** (2 / 3):6.3f} " + " ".join(f"{v:10.4f}" for v in sl)
              + f" {tot:11.4f}")
far = max((s[-1] for s in fold_slopes.values()), key=lambda v: abs(v + 0.5))
print(f"   slope farthest from -1/2 at eps = {SLOPE_EPS[-1]:g}: {far:.4f}")
MACROS["valFoldSlopeFar"] = f"{far:.3f}"

# ── 3. Error of the leading law ───────────────────────────────────────────────────────
print()
print("=" * 78)
print("3. FOLD: relative error |P / P_lead - 1|, P_lead = kappa |x_-'| = Phi_fold(rho~)/(2 sigma^(2/3) sqrt(eps))")
print(f"   on {len(EPS_ERR)} points of [{EPS_MIN:g}, {EPS_WIDE:g}); slope of log(error) on eps < {EPS_NARROW:g}")
print("=" * 78)
print(f"{'sigma':>5} {'rho':>5} {'max eps<%g' % EPS_WIDE:>13} {'max eps<%g' % EPS_NARROW:>13} "
      f"{'err(1e-4)':>10} {'slope':>8} {'95% interval':>18}")
err_wide = err_narrow = 0.0
eta_at_wide = 0.0   # eta* = sqrt(eps)/sigma^(2/3) at the largest error for eps < EPS_WIDE
X, Y, G = [], [], []
signed = {}
for g, sig in enumerate(SIGMAS):
    for h, rho in enumerate(RHOS):
        rt = rho / sig ** (2 / 3)
        lead = fold_lead_price(phi_true_scalar(rt), sig ** (2 / 3), EPS_ERR)
        P = np.array([price("fold", e, sig, rho) for e in EPS_ERR])
        signed[(sig, rho)] = P / lead - 1
        err = np.abs(P / lead - 1)
        nar = EPS_ERR < EPS_NARROW
        res = stats.linregress(np.log(EPS_ERR[nar]), np.log(err[nar]))
        tq = stats.t.ppf(0.975, nar.sum() - 2)
        if err.max() > err_wide:
            eta_at_wide = float(np.sqrt(EPS_ERR[np.argmax(err)]) / sig ** (2 / 3))
        err_wide = max(err_wide, err.max())
        err_narrow = max(err_narrow, err[nar].max())
        X += list(np.log(EPS_ERR[nar])); Y += list(np.log(err[nar])); G += [3 * g + h] * int(nar.sum())
        print(f"{sig:5.1f} {rho:5.2f} {100 * err.max():12.2f}% {100 * err[nar].max():12.2f}% "
              f"{100 * err[0]:9.3f}% {res.slope:8.4f}   [{res.slope - tq * res.stderr:.4f}, "
              f"{res.slope + tq * res.stderr:.4f}]")
# One slope for the nine cases: least squares with an intercept per case (fixed effects).
X, Y, G = np.array(X), np.array(Y), np.array(G)
D = np.zeros((len(X), G.max() + 1)); D[np.arange(len(X)), G] = 1.0
A = np.column_stack([X, D])
coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
resid = Y - A @ coef
dof = len(Y) - A.shape[1]
se = np.sqrt(resid @ resid / dof * np.linalg.inv(A.T @ A)[0, 0])
tq = stats.t.ppf(0.975, dof)
print(f"   pooled slope (one intercept per case) = {coef[0]:.4f}, 95% interval "
      f"[{coef[0] - tq * se:.4f}, {coef[0] + tq * se:.4f}]")
print("   (the curves are deterministic: the interval measures their curvature in the window,"
      " not a sampling error)")
print(f"   largest error: {100 * err_wide:.2f} % for eps < {EPS_WIDE:g}, "
      f"{100 * err_narrow:.2f} % for eps < {EPS_NARROW:g}")
show = [0, int(np.argmin(np.abs(EPS_ERR - 1e-3))), int(np.argmin(np.abs(EPS_ERR - 1e-2)))]
print("   signed error P/P_lead - 1:")
print(f"   {'sigma':>5} {'rho':>5} " + " ".join(f"{'eps=%.2g' % EPS_ERR[i]:>11}" for i in show))
n_change = 0
for (sig, rho), s_err in signed.items():
    nar = EPS_ERR < EPS_NARROW
    n_change += int(np.any(np.sign(s_err[nar]) != np.sign(s_err[0])))
    print(f"   {sig:5.1f} {rho:5.2f} " + " ".join(f"{100 * s_err[i]:+10.3f}%" for i in show))
print(f"   the error changes sign on eps < {EPS_NARROW:g} in {n_change} of {len(signed)} cases")
MACROS["valLeadEpsWide"] = f"{EPS_WIDE:g}"
MACROS["valLeadEpsNarrow"] = f"{EPS_NARROW:g}"
MACROS["valLeadErrMaxPctWide"] = fmt_sig(100 * err_wide)
MACROS["valLeadErrMaxPctNarrow"] = fmt_sig(100 * err_narrow)
# The numerical-methods appendix says the largest error for eps < EPS_WIDE is reached at the
# largest eta* of the grid, and quotes that eta*; checked here rather than assumed.
eta_grid_max = float(np.sqrt(EPS_ERR.max()) / min(SIGMAS) ** (2 / 3))
print(f"   largest eta* on the grid: {eta_grid_max:.4f}; eta* at the largest error for "
      f"eps < {EPS_WIDE:g}: {eta_at_wide:.4f}")
if not abs(eta_at_wide - eta_grid_max) < 1e-12:
    raise SystemExit("[ERROR] validation_bvp.py: the largest error for eps < %g is at eta* = %.4f, "
                     "not at the largest eta* of the grid, %.4f; app:solver says it is"
                     % (EPS_WIDE, eta_at_wide, eta_grid_max))
MACROS["valLeadEtaMax"] = fmt_sig(eta_grid_max)
MACROS["valLeadErrSlope"] = f"{coef[0]:.2f}"
MACROS["valLeadErrSlopeLo"] = f"{coef[0] - tq * se:.2f}"
MACROS["valLeadErrSlopeHi"] = f"{coef[0] + tq * se:.2f}"

# ── 4. Pitchfork, transcritical, Hopf ─────────────────────────────────────────────────
print()
print("=" * 78)
print("4a. PITCHFORK (mu x - x^3, boundary fixed at 0): local slope of log P against log eps")
print("    last column: the derivative along x*(eps), for comparison only")
print("=" * 78)
print(f"{'sigma':>5} {'rho':>5} " + " ".join(f"{'eps=%g' % e:>10}" for e in SLOPE_EPS) + f" {'total 1e-4':>11}")
pf_slopes = {}
for sig in SIGMAS:
    for rho in RHOS:
        sl = [local_slope("pitchfork", e, sig, rho) for e in SLOPE_EPS]
        pf_slopes[(sig, rho)] = sl
        tot = local_slope("pitchfork", SLOPE_EPS[-1], sig, rho, pricefn=price_total)
        print(f"{sig:5.1f} {rho:5.2f} " + " ".join(f"{v:10.4f}" for v in sl) + f" {tot:11.4f}")
far_pf = max((s[-1] for s in pf_slopes.values()), key=lambda v: abs(v + 0.5))
print(f"   slope farthest from -1/2 at eps = {SLOPE_EPS[-1]:g}: {far_pf:.4f}")
print(f"   the slope tends to +1/2: the price vanishes as sqrt(eps), the boundary 0 does not move")
MACROS["valPitchSlopeFar"] = f"{far_pf:.3f}"   # Retired (REMOVED_MACROS)
# The slope of the price at fixed state at the smallest eps computed, the one
# farthest from zero on the grid. Positive: the price falls to zero with the budget.
fix_pf = max((s[-1] for s in pf_slopes.values()), key=abs)
print(f"   slope farthest from 0 at eps = {SLOPE_EPS[-1]:g}: {fix_pf:.4f}")
if not min(s[-1] for s in pf_slopes.values()) > 0:
    raise SystemExit("[ERROR] pitchfork: a slope of the price at fixed state is not positive")
MACROS["valPitchSlopeFix"] = f"{fix_pf:.2f}"

print()
print("=" * 78)
print("4b. TRANSCRITICAL (mu x - x^2, boundary fixed at 0): the price as eps -> 0")
print("    P at fixed state, P/eps and the local slope; the constant -phi_0'(0) of the mu = 0 BVP,")
print("    which is the limit of the derivative along x*(eps) (comparison), against Phi_fold/sigma^(2/3)")
print("=" * 78)
print(f"{'sigma':>5} {'rho':>5} " + " ".join(f"{'P(%g)' % e:>11}" for e in EPS_TC)
      + f" {'P/eps (1e-4)':>13} {'slope 3e-4':>11} {'-phi0p(0)':>10} {'Phi/s^2/3':>10} {'c/(Phi/s^2/3)-1':>16}")
tc_fold = 0.0
tc_fix = []   # Local slope at the smallest eps computed, EPS_TC[0]
for sig in SIGMAS:
    for rho in RHOS:
        P = [price("transcritical", e, sig, rho) for e in EPS_TC]
        sl = local_slope("transcritical", EPS_TC[1], sig, rho)
        tc_fix.append(local_slope("transcritical", EPS_TC[0], sig, rho))
        _, c_tc = p2.transcritical_scc_constant(sig, rho, tol=TOL)
        c_fold = phi_true_scalar(rho / sig ** (2 / 3)) / sig ** (2 / 3)
        tc_fold = max(tc_fold, abs(c_tc / c_fold - 1))
        print(f"{sig:5.1f} {rho:5.2f} " + " ".join(f"{v:11.4e}" for v in P)
              + f" {P[0] / EPS_TC[0]:13.5f} {sl:11.4f} {c_tc:10.6f} {c_fold:10.6f} {c_tc / c_fold - 1:+16.2e}")
print(f"   largest |constant / (Phi_fold(rho~)/sigma^(2/3)) - 1| = {100 * tc_fold:.2e} %")
print("   At fixed state the boundary x = 0 does not move and the price falls as eps (slope 1).")
print("   The constant -phi_0'(0) = Phi_fold(rho~)/sigma^(2/3) is the limit of the derivative along")
print("   x*(eps) only; it is no longer the limit of the price.")
MACROS["valTcFoldGapPct"] = fmt_sig(100 * tc_fold)   # Retired (REMOVED_MACROS)
fix_tc = max(tc_fix, key=abs)
print(f"   local slope at eps = {EPS_TC[0]:g}, farthest from 0: {fix_tc:.4f} "
      f"(all in [{min(tc_fix):.4f}, {max(tc_fix):.4f}])")
if not min(tc_fix) > 0:
    raise SystemExit("[ERROR] transcritical: a slope of the price at fixed state is not positive")
MACROS["valTcSlopeFix"] = f"{fix_tc:.2f}"

print()
print("=" * 78)
print("4c. HOPF (not reported in the paper): radial dynamics r(mu - r^2) + sigma^2/(2r), crossing circle r_c = 1 fixed,")
print("    stable state on the cycle r* = sqrt(mu), budget mu - mu_c = eps (mu_c = r_c^2 = 1)")
print("=" * 78)
R_C = 1.0


def phi_hopf(mu, sigma, rho, r):
    drift = lambda rr, m: rr * (m - rr ** 2) + sigma ** 2 / (2.0 * rr)
    return p2.solve_phi_at_xstar(mu, sigma, rho, drift, R_C, lambda m: r, 1.0,
                                 R_C + X_RIGHT_L * max(sigma, 0.5), tol=TOL, max_nodes=MAX_NODES)


def price_hopf(eps, sigma, rho):
    return price_fixed(lambda m, r: phi_hopf(R_C ** 2 + m, sigma, rho, r), eps,
                       float(np.sqrt(R_C ** 2 + eps)))


print(f"{'sigma':>5} {'rho':>5} " + " ".join(f"{'P(%g)' % e:>11}" for e in SLOPE_EPS)
      + " " + " ".join(f"{'slope %g' % e:>11}" for e in SLOPE_EPS))
for sig in (0.5, 1.0):
    for rho in (0.02, 0.10):
        P = [price_hopf(e, sig, rho) for e in SLOPE_EPS]
        sl = [(np.log(price_hopf(e * np.exp(D_SLOPE), sig, rho))
               - np.log(price_hopf(e * np.exp(-D_SLOPE), sig, rho))) / (2 * D_SLOPE) for e in SLOPE_EPS]
        print(f"{sig:5.1f} {rho:5.2f} " + " ".join(f"{v:11.4e}" for v in P)
              + " " + " ".join(f"{v:11.4f}" for v in sl))
print("   The crossing circle does not move with the budget: at fixed state the price is bounded")
print("   and vanishes as the cycle reaches the circle.")

# ── 5. Sensitivity of F, the AMOC ───────────────────────────────────────
# F = (eps_now/eps_dyn)^(1/2) (the appendix on where the elements sit, eps_dyn.py) is the ratio of the LEADING-LAW prices
# at eps_dyn and eps_now. Here the same ratio is computed on the BVP, at fixed state. For a fold,
# theta = 2 sqrt(mu) at the stable state and l = sigma^(2/3), so eta* = sqrt(mu)/l and rho~ = rho/l
# = 2 rho eta*/theta: once eta* is fixed at today's proximity, the measured theta fixes rho~.
# eta*_now is NOT identified -- it needs the physical margin (same appendix) -- so it is set to two
# values. Model units: M* = 1, mu = eps. The gap reported is F_BVP / F - 1, with
# F_BVP = P(eps_dyn) / P(eps_now).
# Inputs: E_0, M*, theta, eps_now through eps_dyn.amoc_inputs(), which reads the ews and
# calibration captures; this section therefore needs the AMOC series. Without the two captures
# it says so and emits nothing (collect_computed.py then stops on the missing ews capture).
print()
print("=" * 78)
print("5. AMOC: F = P(eps_dyn)/P(eps_now) on the BVP against the leading law (eps_now/eps_dyn)^(1/2)")
print("=" * 78)
ETA_SENS = (0.1, 1.0)
RAW = os.path.join(os.path.dirname(os.path.dirname(HERE)), "02_output", "values", "raw_runs")
if all(os.path.exists(os.path.join(RAW, k + ".out")) for k in ("ews_calibration", "calibration_ci")):
    import eps_dyn as ed                                   # noqa: E402  (01_calibration, on sys.path)
    e0, mstar, theta, eps_now = ed.amoc_inputs()
    rho_y = float(LITERATURE_MACROS["rhoBase"][0])
    e_dyn = ed.eps_dyn(e0, mstar, theta, eps_now)
    f_lead = ed.amp(e_dyn, eps_now)
    print(f"   E_0 = {e0:g}, M* = {mstar:g}, theta = {theta:g}/yr, rho = {rho_y:g}, eps_now = {eps_now:g}, "
          f"eps_dyn = {e_dyn:.4f}, F = {f_lead:.3f}")
    print(f"{'eta*_now':>9} {'rho~':>7} {'P/P_lead (now)':>15} {'(dyn)':>8} {'F_BVP':>8} {'F_BVP/F - 1':>12} {'total':>9}")
    for k, eta in enumerate(ETA_SENS):
        l = np.sqrt(eps_now) / eta
        sig = l ** 1.5
        rt = 2 * rho_y * eta / theta
        lead = lambda e: fold_lead_price(phi_true_scalar(rt), l, e)
        p_now, p_dyn = price("fold", eps_now, sig, rt * l), price("fold", e_dyn, sig, rt * l)
        gap = (p_dyn / p_now) / f_lead - 1
        gap_tot = (price_total("fold", e_dyn, sig, rt * l) / price_total("fold", eps_now, sig, rt * l)) / f_lead - 1
        print(f"{eta:9g} {rt:7.4f} {p_now / lead(eps_now):15.4f} {p_dyn / lead(e_dyn):8.4f} "
              f"{p_dyn / p_now:8.3f} {100 * gap:+11.1f}% {100 * gap_tot:+8.1f}%")
        sfx = "Lo" if k == 0 else "Hi"
        MACROS["valSensFEta" + sfx] = f"{eta:g}"
        MACROS["valSensFGapPct" + sfx] = ("-" if gap < 0 else "") + fmt_sig(100 * abs(gap))
        # that appendix says the lift is SMALLER than F at these two eta*; the text quotes the shortfall
        # as a positive percentage, so the sign is checked here rather than assumed.
        if not gap < 0:
            raise SystemExit("[ERROR] validation_bvp.py: F_BVP/F - 1 = %+.3f at eta*_now = %g; "
                             "ssec:window says the lift is smaller than F there" % (gap, eta))
        MACROS["valSensFShortPct" + sfx] = fmt_sig(100 * abs(gap))
    # the appendix on where the elements sit: "There the crossing value falls fastest as the margin opens". phi(x*; eps_now)
    # on a grid of eta*_now values, printed without a macro; the largest drop between
    # two successive eta* must fall in an interval that contains eta* = 1 (the edge of the layer).
    ETA_PHI = (0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0)
    # Abstract: "grows large only once crossing is likely". The price at fixed state
    # today and its ratio to the leading law on the same grid, printed without a macro.
    phis, p_fix, p_lead = [], [], []
    for eta in ETA_PHI:
        l = np.sqrt(eps_now) / eta
        rt = 2 * rho_y * eta / theta
        phis.append(phi("fold", eps_now, l ** 1.5, rt * l))
        p_fix.append(price("fold", eps_now, l ** 1.5, rt * l))
        p_lead.append(fold_lead_price(phi_true_scalar(rt), l, eps_now))
    print("   crossing value and price at fixed state at today's proximity (model units):")
    print(f"     {'eta*_now':>8} {'phi(x*; eps_now)':>17} {'P_fix(eps_now)':>15} {'P_lead':>11} {'P_fix/P_lead':>13}")
    for eta, ph, pf, pl in zip(ETA_PHI, phis, p_fix, p_lead):
        print(f"     {eta:8g} {ph:17.4e} {pf:15.4e} {pl:11.4e} {pf / pl:13.4e}")
    drops = [phis[k] - phis[k + 1] for k in range(len(phis) - 1)]
    k_max = int(np.argmax(drops))
    lo, hi = ETA_PHI[k_max], ETA_PHI[k_max + 1]
    print(f"   largest drop of phi between successive eta*: {drops[k_max]:.4f}, on [{lo:g}, {hi:g}]")
    if not lo <= 1.0 <= hi:
        raise SystemExit("[ERROR] validation_bvp.py: phi falls fastest on [%g, %g], which does not "
                         "contain eta* = 1; ssec:window says it falls fastest at the edge of the layer" % (lo, hi))
    # Further out (that appendix: "the lift exceeds F many times over"): the same ratio at
    # eta*_now = ETA_FAR, printed without a macro; collect_computed.py asserts it is > 10.
    ETA_FAR = 1.5
    l = np.sqrt(eps_now) / ETA_FAR
    rt = 2 * rho_y * ETA_FAR / theta
    p_now, p_dyn = price("fold", eps_now, l ** 1.5, rt * l), price("fold", e_dyn, l ** 1.5, rt * l)
    print(f"   further out: F_BVP/F at eta*_now = {ETA_FAR:g} = {(p_dyn / p_now) / f_lead:.4g}")
else:
    print("   SKIPPED: the ews_calibration and calibration_ci captures are not there (the AMOC")
    print("   series is needed for theta); no valSens* macro is emitted.")

# ── 6. the leading law at small rho~ ────────────────────────────────────────────
# At a fixed state the closed form of the amplitude-to-price appendix has a second term, from the dependence of
# B(x) = mu x - x^3/3 on the budget: -((x - x_-)/sigma^2) phi. At the stable state x - x_- = d,
# so its ratio to the leading term kappa |x_-'| is 4 eta*^2 / Phi_fold(rho~) (with l^3 = sigma^2).
# Printed: that ratio computed from phi at the stable state, the closed form 4 eta*^2/Phi, and the
# exact price at fixed state against the leading law, P/P_lead. sigma = 1, so l = 1, mu = eta*^2.
print()
print("=" * 78)
print("6. FOLD: the leading law at small rho~; the second term of the amplitude-to-price expansion against 4 eta*^2/Phi_fold")
print("=" * 78)
SMALL_RHO = 0.02
ETA_SMALL = (0.1, 0.3)
CHECK_RHOS = (0.02, 0.05)
ETA_CHECK = (0.05, 0.1)
print(f"{'rho~':>6} {'eta*':>6} {'second/lead':>12} {'4eta^2/Phi':>11} {'ratio-1':>9} {'P/P_lead':>9}")
worst = 0.0
ratios = {}
for rt_ in sorted(set(CHECK_RHOS) | {SMALL_RHO}):
    Phi_ = phi_true_scalar(rt_)
    for eta in sorted(set(ETA_CHECK) | (set(ETA_SMALL) if rt_ == SMALL_RHO else set())):
        mu_ = eta ** 2
        x_ = float(GEOMETRY["fold"][1](mu_))
        ph_ = phi_at("fold", mu_, 1.0, rt_, x_)
        second = 2.0 * np.sqrt(mu_) * ph_                 # (x - x_-)/sigma^2 phi, sigma = 1
        lead_ = fold_lead_price(Phi_, 1.0, mu_)
        closed = 4.0 * eta ** 2 / Phi_
        p_ = price("fold", mu_, 1.0, rt_)
        dev = second / lead_ / closed - 1.0
        if rt_ in CHECK_RHOS and eta in ETA_CHECK:
            worst = max(worst, abs(dev))
        ratios[(rt_, eta)] = p_ / lead_
        print(f"{rt_:6.2f} {eta:6.2f} {second / lead_:12.4f} {closed:11.4f} {dev:+9.4f} {p_ / lead_:9.4f}")
print(f"   largest |(second/lead)/(4 eta*^2/Phi) - 1| at eta* in {ETA_CHECK}: {100 * worst:.2f}%")
if worst > 0.10:
    raise SystemExit("[ERROR] validation_bvp.py: the second term of the amplitude-to-price expansion departs from 4 eta*^2/Phi_fold "
                     "by %.1f%% (> 10%%) at small eta*" % (100 * worst))
for k, eta in zip(("A", "B"), ETA_SMALL):
    r = ratios[(SMALL_RHO, eta)]
    MACROS["valLawRatioSmallRho" + k] = f"{r:.2g}"
    MACROS["valSmallRhoEta" + k] = f"{eta:g}"
MACROS["valSmallRho"] = f"{SMALL_RHO:g}"

# The same comparison at the overturning circulation's rho~,
# rho~ = 2 rho eta*/theta at eta* = ETA_SMALL[0]. theta comes from the ews capture through
# section 5, so these three need the AMOC series, like the valSens* macros.
if "theta" in globals():
    eta_a = ETA_SMALL[0]
    rt_amoc = 2.0 * rho_y * eta_a / theta
    mu_a = eta_a ** 2
    Phi_a = phi_true_scalar(rt_amoc)
    ratio_amoc = price("fold", mu_a, 1.0, rt_amoc) / fold_lead_price(Phi_a, 1.0, mu_a)
    bterm_amoc = 4.0 * eta_a ** 2 / Phi_a
    print(f"   AMOC: rho~ = 2 rho eta*/theta = {rt_amoc:.5f} at eta* = {eta_a:g}: "
          f"P/P_lead = {ratio_amoc:.4f}, 4 eta*^2/Phi_fold = {bterm_amoc:.4f}")
    MACROS["valAmocRhoTilde"] = f"{rt_amoc:.2g}"
    MACROS["valLawRatioAmoc"] = f"{ratio_amoc:.2g}"
    MACROS["valBtermRatioAmoc"] = f"{bterm_amoc:.2g}"
else:
    print("   AMOC: SKIPPED (no ews capture, so no theta); the three valAmoc* macros are not emitted.")

elapsed = time.time() - T_START
print(f"\nruntime: {elapsed:.0f} s")
print("MACROS")
for k, v in MACROS.items():
    print(f"VALIDATION {k} = {v}")
print("END MACROS")
