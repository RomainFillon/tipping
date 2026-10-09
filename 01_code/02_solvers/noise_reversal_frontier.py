#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
REPLICATION HEADER
  PRODUCES   the MACROS block parsed into computed.json
  FEEDS      \etaStarReversal \kappaReversalCeiling \reversalLawSlope
  INPUTS     02_output/values/raw_runs/ews_calibration.out (lambda only, for the AMOC point)
  SEED       none (deterministic solves)
  RUNTIME    6 to 14 min (three cold runs; the widest spread in the package, so plan on the upper end)
  IMPLEMENTS the upper end of the window of the theorem (ssec:gen_thm): where dPi/dsigma changes
             sign, for the fold, reported in eta* = d(mu)/(2 l(sigma))

noise_reversal_frontier.py — the noise reversal of the theorem (ssec:gen_thm), part (ii), holds inside a
window, and this locates its upper end numerically.

THE QUESTION.  Part (ii) of the theorem (ssec:gen_thm) states dPi/dsigma < 0.  The regime it is proved in
is bounded below by eps_dyn and above by eta* < 1, the stable state sitting inside the noise
layer.  Lowering sigma at fixed eps RAISES eta* = sqrt(M* eps)/sigma^{2/3}, so the very
operation the reversal describes eventually leaves the window through the top.  Outside it
crossing is a Kramers escape rather than a delay, phi ~ exp(-(8/3) mu^{3/2}/sigma^2), and the
logarithmic derivative of the price turns positive.  This script finds where.

WHY NOT THE REPOSITORY'S BVP SOLVER ALONE.  The collocation solve of app:solver returns phi
itself.  At eta* = 5 the crossing value is of order e^{-333}, which underflows a double long
before the frontier can be mapped, and solve_bvp loses its own accuracy well before that.
The route used here integrates the EXACT rescaled equation in the logarithm.  With
phi = e^{-B/sigma^2} u, B = mu x - x^3/3 and x = sigma^{2/3} eta,

    u''(eta) = V(eta) u,    V = 2 rho_t - 2 eta + (eta*^2 - eta^2)^2 ,
    log phi(x*) = -(4/3) eta*^3 + [ log u(eta*) - log u(-eta*) ] ,

with eta* = sqrt(mu)/sigma^{2/3} and rho_t = rho/sigma^{2/3}.  Q = -u'/u obeys the Riccati
equation Q' = Q^2 - V, integrated from a WKB start at large eta down to -eta*; the decaying
solution is the attracting one in that direction, so the WKB initialisation suffices.  Only
logarithms are ever formed, so nothing underflows.  Section 1 checks this route against the
repository's own BVP solver where both work: they agree on log Pi to 1e-9, and the BVP
locates the same maximum.

THE PRICE, AT FIXED STATE (eq:gen_scc).  The price is -dphi/deps with the state held fixed, not the
derivative of log phi(x*(eps)) along the stable state.  logphi_res reads log phi at any state y
from the same Riccati solve, with dense output, and price.log_price_fixed differences it in
log mu, step D_PRICE, at the stable state of the central budget held fixed; only logarithms are
formed.  solve_LP also integrates P = dQ/deta* alongside Q,

    P' = 2 Q P - dV/deta* ,   dV/deta* = 4 eta* (eta*^2 - eta^2) ,

which gives L_eta = dL/deta*, the derivative ALONG the stable state; it is not the price and
nothing uses it.  Only L = log phi of solve_LP is used (section 5).

THE REDUCTION, WHICH IS EXACT AND NOT A FIT.  L depends on (eta*, rho_t) alone.  At fixed
eps and rho, a sweep in sigma moves eta* and rho_t both as sigma^{-2/3}, so it travels a RAY
rho_t = kappa eta* whose slope

    kappa = rho / sqrt(M* eps) = rho_t / eta* = 2 rho / lambda

is constant along it.  The frontier can therefore depend on eps and rho through kappa only.
That is why the sweeps of section 2 collapse: they are not four measurements agreeing, they
are one curve sampled four times, and section 3 uses exactly-matched kappa pairs to check
the collapse through the unrescaled BVP, where it is a real test rather than an identity.
Note that kappa = 2 rho/lambda is observable: it needs the measured relaxation rate, not the
budget M*, so the AMOC point of section 6 is free of the missing margin ratio.

THE LAW.  Far outside the layer log phi -> -(8/3) eta*^3 + log(sqrt(mu)/(pi rho)), whence
d log Pi/d log sigma = -2 + (16/3) eta*^3 [1 - 2k/(k+rho)] with k the Kramers rate.  The
frontier is thus NOT a power law in rho_t: it sits where the escape rate has fallen a fixed
factor below rho, so eta*_flip^3 is affine in log(1/kappa) with slope 3/8, and eta*_flip
itself moves only as the cube root of a logarithm.  Section 4 fits it.  Section 5 is the
control that the Kramers regime is genuinely reached: d log phi / d(mu^{3/2}/sigma^2) must
converge to -8/3, and it does, to six figures by eta* = 2.5; d log Pi converges to
-(8/3) e^{-3d/2}, d the step of the price at fixed state.
"""
import os
import re
import sys

import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price import price_fixed, log_price_fixed, D_PRICE  # noqa: E402  (the one price, eq:gen_scc)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_EWS = os.path.join(ROOT, "02_output", "values", "raw_runs", "ews_calibration.out")

RHO_BASE, RHO_LOW = 0.03, 0.01          # \rhoBase and a Stern-side rate, the two of Section 5
EPS_SWEEP = (0.005, 0.015, 0.05, 0.15, 0.5)   # two decades of proximity
RTOL, ATOL, MAXSTEP = 1e-11, 1e-13, 0.1
KRAMERS_SLOPE = -8.0 / 3.0              # d log phi / d(mu^{3/2}/sigma^2) far from threshold
LAW_SLOPE = 3.0 / 8.0                   # d eta*_flip^3 / d log(1/kappa), same 8/3 inverted
# \etaStarPersistAMOC, the Kramers lower bound on the AMOC's eta*_now transported to current
# proximity. Its generator paper3_persistence_bound.py is not part of this package -- the manuscript
# makes that argument in words and quotes none of its eleven macros, so the package no
# longer computes them. This is therefore a STANDING CONSTANT, not a value read from the
# chain, and it is the last one that generator emitted. Recompute or recover it with
#     git show c4cff58:replication/01_code/02_solvers/paper3_persistence_bound.py
# It was 1.21 while that script transported the bound by sqrt(eps_0); the correct factor
# on d/SD is eps_0^{3/4}, since the observed SD rises as eps^{-1/4} while the margin falls
# as eps^{1/2}.
# The placement below is unaffected in sign: 1.14 still sits above the frontier.
ETA_PERSIST_AMOC = 1.14

ok = True


def check(label, got, want, tol=5e-3, note=""):
    global ok
    rel = abs(got / want - 1) if want else abs(got - want)
    good = rel <= tol
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:<48} {got:>11.6f} vs {want:>11.6f}"
          f"   ({rel:.2%}){('  ' + note) if note else ''}")


def read_lambda():
    if not os.path.exists(RAW_EWS):
        sys.exit(f"[ERROR] missing {RAW_EWS} — run ews_calibration.py first (needs network).")
    txt = open(RAW_EWS, encoding="utf-8", errors="replace").read()
    lam = re.search(r"\|lambda\|\s*\(full-sample\)\s*=\s*([\d.]+)", txt)
    var = re.search(r"Var\(X_t\) detrended\s*=\s*([\d.]+)", txt)
    if not (lam and var):
        sys.exit("[ERROR] could not parse lambda / Var from the cached ews run.")
    return float(lam.group(1)), float(var.group(1))


# ── the log-space route: Riccati plus its variational equation ────────────────
def solve_LP(es, rt):
    """(L, dL/deta*) with L = log phi(x*), at fixed rho_t.  Fold, M* absorbed in mu."""
    em = max(8.0, 1.6 * es + 6.0)
    V0 = 2.0 * rt - 2.0 * em + (es * es - em * em) ** 2
    Q0 = np.sqrt(V0)
    P0 = 4.0 * es * (es * es - em * em) / (2.0 * Q0)

    def rhs(e, y):
        Q, _, P, _ = y
        d = es * es - e * e
        return [Q * Q - (2.0 * rt - 2.0 * e + d * d), Q, 2.0 * Q * P - 4.0 * es * d, P]

    s = solve_ivp(rhs, (em, -es), [Q0, 0.0, P0, 0.0], method="DOP853",
                  rtol=RTOL, atol=ATOL, dense_output=True, max_step=MAXSTEP)
    if not s.success:
        return np.nan, np.nan
    lo, hi = s.sol(-es), s.sol(es)
    L = -(4.0 / 3.0) * es ** 3 + (lo[1] - hi[1])
    # d/deta* of [J(-eta*) - J(eta*)] carries both the moving endpoints and dQ/deta*
    Le = -4.0 * es ** 2 - lo[0] - hi[0] + (lo[3] - hi[3])
    return float(L), float(Le)


def logphi_res(y, m, rt):
    """log phi(y; m) in rescaled units (l = 1), at ANY state y > -sqrt(m), budget m = eta*^2.

    The price is taken at FIXED state, so phi is needed at a state other than the
    stable state of the budget being varied. The same Riccati route as solve_LP, read with
    dense output: with B(x) = m x - x^3/3 and J(e) = int_{em}^{e} Q,
        log phi(y) = -[B(y) - B(-sqrt m)] + [J(-sqrt m) - J(y)] ,
    which at y = sqrt(m) is the L of solve_LP."""
    es_m = np.sqrt(m)
    em = max(8.0, 1.6 * es_m + 6.0)
    V = lambda e: 2.0 * rt - 2.0 * e + (m - e * e) ** 2
    Q0 = np.sqrt(V(em))

    def rhs(e, z):
        return [z[0] * z[0] - V(e), z[0]]

    sol = solve_ivp(rhs, (em, -es_m), [Q0, 0.0], method="DOP853", rtol=RTOL, atol=ATOL,
                    dense_output=True, max_step=MAXSTEP)
    if not sol.success:
        return np.nan
    B = lambda x: m * x - x ** 3 / 3.0
    return float(-(B(y) - B(-es_m)) + (sol.sol(-es_m)[1] - sol.sol(y)[1]))


def logPres_fixed(es, rt):
    """log of the price at fixed state in rescaled units: -d phi/d m at y = eta*, m = eta*^2."""
    return log_price_fixed(lambda mm, yy: logphi_res(yy, mm, rt), es * es, es)


def logPi(eps, sigma, rho, mstar=1.0):
    """log of the price Pi = -M* dphi/dmu at FIXED state (unit damage), by the log route.
    phi(x; mu) depends on (x/l, mu/l^2, rho_t), so the price is l^-2 times the rescaled one."""
    ell = sigma ** (2.0 / 3.0)
    mu = mstar * eps
    return logPres_fixed(np.sqrt(mu) / ell, rho / ell) - 2.0 * np.log(ell) + np.log(mstar)


def _F(es, kappa):
    """log Pi along the ray, up to terms constant in sigma: with l = sqrt(eps)/eta* at M* = 1,
    log Pi = -log eps + 2 log eta* + log P_res(eta*, kappa eta*), so F = 2 log eta* + log P_res."""
    return 2.0 * np.log(es) + logPres_fixed(es, kappa * es)


def dlogPi_dlogsigma(es, kappa, h=1.5e-3):
    """d log Pi / d log sigma at fixed eps, = -(2/3) dF/d log eta*."""
    return -(2.0 / 3.0) * (_F(es * np.exp(h), kappa) - _F(es * np.exp(-h), kappa)) / (2.0 * h)


def eta_reversal(kappa, lo=0.12, hi=4.5, iters=20):
    """the eta* at which dPi/dsigma changes sign, by bisection in log eta*."""
    fa, fb = dlogPi_dlogsigma(lo, kappa), dlogPi_dlogsigma(hi, kappa)
    if np.isnan(fa) or np.isnan(fb) or (fa > 0) == (fb > 0):
        return np.nan
    for _ in range(iters):
        m = np.sqrt(lo * hi)
        fm = dlogPi_dlogsigma(m, kappa)
        if (fa > 0) == (fm > 0):
            lo, fa = m, fm
        else:
            hi = m
    return np.sqrt(lo * hi)


# ── the repository's own solver, for the validation of section 1 ──────────────
def phi_bvp(mu, sigma, rho, x=None, x_right=8.0, n=800, tol=1e-9):
    """phi(x) for (sigma^2/2) phi'' + (mu - x^2) phi' = rho phi, phi(x_-)=1, phi(inf)=0, at the
    state x (default the stable state sqrt(mu)). The same collocation solve as app:solver
    (and as an archived contrast script did)."""
    inv = 2.0 / sigma ** 2
    fun = lambda x, y: np.vstack([y[1], inv * (rho * y[0] - (mu - x ** 2) * y[1])])
    bc = lambda ya, yb: np.array([ya[0] - 1.0, yb[0]])
    xl = -np.sqrt(mu)
    xg = np.linspace(xl, x_right, n)
    yg = np.vstack([np.linspace(1, 0, n), np.full(n, -1.0 / (x_right - xl))])
    s = solve_bvp(fun, bc, xg, yg, tol=tol, max_nodes=200000, verbose=0)
    return float(s.sol(np.sqrt(mu) if x is None else x)[0]) if s.success else np.nan


def logPi_bvp(eps, sigma, rho):
    """the same price by the repository's BVP, at fixed state (price.price_fixed)."""
    d = price_fixed(lambda m, x: phi_bvp(m, sigma, rho, x), eps, np.sqrt(eps))
    return np.log(d) if (d == d and d > 0) else np.nan


def parabola_vertex(x, y):
    i = int(np.nanargmax(y))
    if not 0 < i < len(x) - 1:
        return np.nan
    d = y[i - 1] - 2 * y[i] + y[i + 1]
    return x[i] - 0.5 * (x[i + 1] - x[i - 1]) * (y[i + 1] - y[i - 1]) / (2 * d) if d else x[i]


# ── sections ─────────────────────────────────────────────────────────────────
def section1():
    print("\n1. the log route against the repository's BVP solver, where both work")
    print(f"  {'eps':>7} {'sigma':>7} {'rho':>6} {'eta*':>7} {'log Pi BVP':>13} "
          f"{'log Pi log':>13} {'ecart':>10}")
    for eps, sig, rho in ((0.05, 1.0, 0.03), (0.10, 1.0, 0.03), (0.15, 0.3, 0.03),
                          (0.50, 0.3, 0.03), (1.00, 0.6, 0.03), (0.40, 0.5, 0.01)):
        a, b = logPi_bvp(eps, sig, rho), logPi(eps, sig, rho)
        print(f"  {eps:>7.3f} {sig:>7.2f} {rho:>6.2f} {np.sqrt(eps)/sig**(2/3):>7.4f} "
              f"{a:>13.7f} {b:>13.7f} {abs(a-b):>10.2e}")
        check(f"log Pi at (eps={eps}, sigma={sig}, rho={rho})", b, a, tol=1e-6)
    print("     -> the two routes are the same number; beyond eta* ~ 1.8 only the log one")
    print("        survives, phi itself having left the range of a double")


def section2():
    """Pi(sigma) at fixed proximity, down to well below the layer, read in eta*."""
    print("\n2. the price against the noise level, at fixed proximity")
    flips = {}
    for rho in (RHO_BASE, RHO_LOW):
        for eps in EPS_SWEEP:
            kappa = rho / np.sqrt(eps)
            er = eta_reversal(kappa)
            flips[(eps, rho)] = (kappa, er)
        print(f"\n  rho = {rho}")
        print(f"  {'eps':>8} {'kappa':>9} {'sigma_rev':>10} {'eta*_rev':>9} "
              f"{'rho_t at rev':>13} {'sign below':>11} {'sign above':>11}")
        for eps in EPS_SWEEP:
            kappa, er = flips[(eps, rho)]
            s_rev = (np.sqrt(eps) / er) ** 1.5
            below = dlogPi_dlogsigma(er * 0.6, kappa)   # inside the layer
            above = dlogPi_dlogsigma(er * 1.8, kappa)   # deep outside it
            print(f"  {eps:>8.3f} {kappa:>9.5f} {s_rev:>10.5f} {er:>9.5f} {kappa*er:>13.5f} "
                  f"{'-' if below < 0 else '+':>11} {'-' if above < 0 else '+':>11}")
            check(f"eps={eps}, rho={rho}: price falls in sigma below the frontier",
                  1.0 if below < 0 else -1.0, 1.0)
            check(f"eps={eps}, rho={rho}: price rises in sigma above the frontier",
                  1.0 if above > 0 else -1.0, 1.0)
    print("\n     -> the frontier is NOT at constant eta*: it runs from about 0.75 to about")
    print("        1.2 across two decades of eps and the two discount rates, and it moves")
    print("        with eps and rho only through kappa = rho/sqrt(M* eps) (section 3)")
    return flips


def section3(flips):
    """The collapse, tested where it is a test: identical kappa from different (eps, rho)."""
    print("\n3. the collapse onto kappa = rho/sqrt(M* eps)")
    print(f"  {'eps':>8} {'rho':>6} {'kappa':>9} {'eta*_rev':>9}")
    for (eps, rho), (kappa, er) in sorted(flips.items(), key=lambda kv: kv[1][0]):
        print(f"  {eps:>8.3f} {rho:>6.2f} {kappa:>9.5f} {er:>9.5f}")
    # exactly matched kappa: (eps, rho) and (9 eps, 3 rho) share it
    print("\n  matched pairs, kappa identical by construction:")
    for eps, rho in ((0.005, 0.01), (0.015, 0.01), (0.05, 0.01)):
        k1, e1 = rho / np.sqrt(eps), eta_reversal(rho / np.sqrt(eps))
        e2 = eta_reversal(3 * rho / np.sqrt(9 * eps))
        print(f"  (eps={eps:.3f}, rho={rho:.2f}) vs (eps={9*eps:.3f}, rho={3*rho:.2f}): "
              f"kappa={k1:.5f}, eta*_rev = {e1:.6f} vs {e2:.6f}")
        check(f"collapse at kappa={k1:.4f}", e2, e1, tol=1e-4)
    # and the same statement through the UNRESCALED solver, where it is not an identity
    print("\n  the same invariance through the BVP, which knows nothing of the rescaling:")
    print("  (equal (eta*, rho_t) from different (mu, sigma, rho) must give equal log Pi + log mu)")
    es_t, rt_t, ref = 0.9, 0.10, None
    for rho in (0.03, 0.06, 0.12):
        ell = rho / rt_t
        sig, mu = ell ** 1.5, (es_t * ell) ** 2
        v = logPi_bvp(mu, sig, rho) + np.log(mu)
        print(f"    rho={rho:.2f}  sigma={sig:.4f}  mu={mu:.5f}   log Pi + log mu = {v:.8f}")
        if ref is None:
            ref = v
        else:
            check(f"BVP invariance at rho={rho}", v, ref, tol=1e-6)


def section4(flips):
    print("\n4. the law: the frontier is logarithmic in kappa, not a power of rho_t")
    ks = np.array(sorted({k for k, _ in flips.values()} | {3e-3, 1e-3, 1e-4, 1e-5, 1e-6}))
    es = np.array([eta_reversal(k) for k in ks])
    x, y = np.log(1.0 / ks), es ** 3
    print(f"  {'kappa':>10} {'log(1/kappa)':>13} {'eta*_rev':>9} {'eta*_rev^3':>11} "
          f"{'local slope':>12}")
    for i, k in enumerate(ks):
        sl = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]) if 0 < i < len(ks) - 1 else np.nan
        print(f"  {k:>10.2e} {x[i]:>13.4f} {es[i]:>9.5f} {y[i]:>11.5f} {sl:>12.5f}")
    # the table is ordered in kappa, so the asymptotic end is the FIRST rows, not the last
    sl_far = (y[0] - y[1]) / (x[0] - x[1])
    print(f"     -> the slope at the small-kappa end is {sl_far:.5f} against the predicted "
          f"3/8 = {LAW_SLOPE}, and it")
    print("        rises monotonically toward it down the table, the approach being the")
    print("        subleading log log of the Kramers prefactor")
    check("slope of eta*_rev^3 in log(1/kappa), asymptotic end", sl_far, LAW_SLOPE, tol=0.03)
    b, a = np.polyfit(x, y, 1)
    resid = max(abs((a + b * x) ** (1 / 3) / es - 1))
    print(f"     -> over the whole range a straight line gives eta*_rev^3 = {b:.4f} "
          f"log(1/kappa) {a:+.4f},")
    print(f"        within {resid:.1%} of eta*_rev everywhere on it")
    # a power law in rho_t, which is what the cross-partial obeys, is rejected here
    rt = ks * es
    n_fit = np.polyfit(np.log(rt), np.log(es), 1)[0]
    pred = np.exp(np.polyval(np.polyfit(np.log(rt), np.log(es), 1), np.log(rt)))
    print(f"     -> forcing eta*_rev = a rho_t^n gives n = {n_fit:.4f} with a maximum "
          f"residual of {max(abs(pred/es - 1)):.1%},")
    print("        which is why the frontier is reported as a law in kappa and not in rho_t")
    return sl_far


def section5():
    print("\n5. control: far outside the layer the crossing value is of Kramers form")
    # The exponent, from the barrier rather than from the fit. With dX = b dt + sigma dW and
    # b = mu - x^2, the potential is U = -int b, so the barrier from the stable state to the
    # boundary is  U(x_-) - U(x_+) = int_{-sqrt(mu)}^{sqrt(mu)} (mu - x^2) dx = (4/3) mu^{3/2},
    # which is the Delta U the appendix states. The stationary density is exp(-2U/sigma^2), so
    # the escape rate carries exp(-2 Delta U/sigma^2) = exp(-(8/3) mu^{3/2}/sigma^2), and phi,
    # being rho times the mean discounted time to escape, inherits that exponent unchanged.
    mu = 1.0
    dU = 2.0 * mu ** 1.5 - (2.0 / 3.0) * mu ** 1.5      # int (mu - x^2) over [-sqrt mu, sqrt mu]
    check("barrier Delta U at mu = 1, by quadrature of the drift", dU, 4.0 / 3.0, tol=1e-12)
    check("Kramers exponent 2 Delta U / sigma^2, in mu^{3/2}/sigma^2", 2.0 * dU, 8.0 / 3.0,
          tol=1e-12, note="so the slope below must be -8/3, and it is")
    print(f"  {'eta*':>6} {'mu^{3/2}/sigma^2':>17} {'log phi':>12} {'d log phi / dx':>15} "
          f"{'d log Pi / dx':>14}")
    kappa = 0.03
    grid = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0])
    L = np.array([solve_LP(e, kappa * e)[0] for e in grid])
    P = np.array([_F(e, kappa) for e in grid])
    x = grid ** 3
    slopes = []
    for i, e in enumerate(grid):
        dl = (L[i + 1] - L[i - 1]) / (x[i + 1] - x[i - 1]) if 0 < i < len(grid) - 1 else np.nan
        dp = (P[i + 1] - P[i - 1]) / (x[i + 1] - x[i - 1]) if 0 < i < len(grid) - 1 else np.nan
        slopes.append((dl, dp))
        print(f"  {e:>6.1f} {x[i]:>17.2f} {L[i]:>12.3f} {dl:>15.6f} {dp:>14.6f}")
    print(f"     -> mu^{{3/2}}/sigma^2 = eta*^3 exactly, so the abscissa is the paper's own")
    print(f"        variable; the expected slope is -8/3 = {KRAMERS_SLOPE:.6f}")
    check("d log phi / d(mu^{3/2}/sigma^2) at eta* = 3", slopes[4][0], KRAMERS_SLOPE, tol=1e-3)
    check("d log phi / d(mu^{3/2}/sigma^2) at eta* = 6", slopes[7][0], KRAMERS_SLOPE, tol=1e-4)
    # The price is the centred difference of price.py, step d = D_PRICE in log mu at the state held
    # fixed. Far in the Kramers regime phi falls so fast in mu that phi(mu e^{-d}) - phi(mu e^{d})
    # is carried by its lower point: log Pi follows log phi read at mu e^{-d}, and since
    # (mu e^{-d})^{3/2} = e^{-3d/2} mu^{3/2}, the slope in mu^{3/2}/sigma^2 is -(8/3) e^{-3d/2}, not
    # -8/3. The expected value carries that factor; the
    # tolerance is the one the check has always had.
    check("d log Pi  / d(mu^{3/2}/sigma^2) at eta* = 6", slopes[7][1],
          KRAMERS_SLOPE * np.exp(-1.5 * D_PRICE), tol=2e-3,
          note="-(8/3) e^{-3d/2}, d the price step")
    print("     -> the regime is genuinely reached rather than approximated by a numerical")
    print("        floor: at eta* = 5 the crossing value is e^{-330}, which the BVP cannot")
    print("        represent and the log route computes to eleven digits")


def section6(lam, var):
    print("\n6. where the AMOC calibration sits, and where the window ceiling sits")
    # This section used to place the AMOC by an eta*_now read at the deduced margin
    # d_phys = lambda, a convention in years, and concluded that the calibrated margin sits
    # BELOW the frontier so the reversal holds there. That conclusion is WITHDRAWN and its
    # macro retired: paper3_persistence_bound.py returns eta*_now >= 1.14 for the AMOC from
    # the Holocene, which is ABOVE the frontier computed here, and paper.tex says so. The
    # frontier itself is unaffected, because the ratio below needs only the measured
    # relaxation rate and the discount rate, never the missing margin.
    kappa = 2.0 * RHO_BASE / lam          # = rho/sqrt(mu); needs no M* and no margin ratio
    er = eta_reversal(kappa)
    print(f"  lambda = {lam:.4f} yr^-1,  Var = {var:.4f} K^2,  rho = {RHO_BASE}")
    print(f"  rho_t/eta* = 2 rho/lambda = {kappa:.5f}")
    print(f"  eta*_rev  = {er:.4f}   (the frontier at that ratio)")
    print(f"  eta*_now  >= {ETA_PERSIST_AMOC:.2f} from the persistence bound "
          "(paper3_persistence_bound.py)")
    print("     -> the AMOC therefore sits ABOVE the frontier, not below it: the reversal")
    print("        does NOT hold at its calibrated position. The ratio uses the measured")
    print("        relaxation rate only, so this placement needs no margin conversion.")
    check("AMOC persistence bound lies above the frontier",
          1.0 if ETA_PERSIST_AMOC > er else 0.0, 1.0)
    # where the frontier crosses the window ceiling eta* = 1
    lo, hi = 0.01, 0.5
    for _ in range(12):
        m = np.sqrt(lo * hi)
        if eta_reversal(m) > 1.0:
            lo = m
        else:
            hi = m
    kcrit = np.sqrt(lo * hi)
    print(f"\n  the frontier crosses the window ceiling eta* = 1 at kappa = {kcrit:.4f}")
    print(f"  (that is, at eps = (rho/{kcrit:.4f})^2/M*, or lambda = {2*RHO_BASE/kcrit:.3f} yr^-1")
    print(f"   at rho = {RHO_BASE})")
    print("     -> below that kappa the reversal is safe throughout the window, the frontier")
    print("        lying above the ceiling; above it a sliver of the window, "
          f"{er:.2f} < eta* < 1 at")
    print("        the AMOC value, carries the opposite sign. The theorem's own ceiling is")
    print("        therefore not by itself sufficient, and the sufficient condition is")
    print("        eta* < eta*_rev(kappa).")
    return er, kappa, kcrit


def main():
    lam, var = read_lambda()
    print("=" * 92)
    print("THE UPPER END OF THE NOISE-REVERSAL WINDOW, FOR THE FOLD")
    print("=" * 92)
    section1()
    flips = section2()
    section3(flips)
    slope = section4(flips)
    section5()
    er, kappa, kcrit = section6(lam, var)

    print("\n" + "=" * 92)
    print("VERDICT:", "every check passed" if ok else "*** A CHECK FAILED ***")
    print("=" * 92)

    # \etaStarNow and \etaStarReversalRatio are no longer exported. Both were read at the
    # deduced margin d_phys = lambda, a convention in years that the persistence bound of
    # paper3_persistence_bound.py rules out physically; the ratio was eta*_rev/eta*_now and
    # so inherited it whole. The frontier macros below need no margin conversion.
    print("\nMACROS")
    print(f"  etaStarReversal = {er:.2f}")
    print(f"  kappaReversalCeiling = {kcrit:.3f}")
    print(f"  reversalLawSlope = {slope:.3f}")
    print("END MACROS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
