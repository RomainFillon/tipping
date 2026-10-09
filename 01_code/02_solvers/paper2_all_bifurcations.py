"""
REPLICATION HEADER
  PRODUCES   fig_p2_universality_table.pdf
  FEEDS      NOTHING in the paper; kept as evidence, not reported in the paper (see section 4 of README.md)
  INPUTS     none
  SEED       none (deterministic)
  RUNTIME    about 30 s
  IMPLEMENTS the four-geometry exponent table, not reported in the paper

Paper 2 — Universal table of critical exponents for codimension-1 bifurcations

Four bifurcations analyzed:
  1. Fold:         dX = (mu - X^2) dt + sigma dW
  2. Pitchfork:    dX = (mu*X - X^3) dt + sigma dW
  3. Transcritical:dX = (mu*X - X^2) dt + sigma dW
  4. (Hopf:        2D, deferred to Paper 3)

For each: compute SCC ~ L * C_bif * epsilon^{-alpha} numerically,
estimate alpha, and derive the inner ODE analytically.

MAIN RESULT (to be proved):
  - Fold:          alpha = 1/2,  inner eq = Airy,        sigma-scale sigma^{2/3}
  - Pitchfork:     alpha = 1/2,  inner eq = QHO/Weber,   sigma-scale sigma^{1/2}
  - Transcritical: alpha = 0,    inner eq = exponential  (no divergence at leading order)
"""

import numpy as np
from scipy.integrate import solve_bvp
from scipy.special import airy
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price import price_fixed  # noqa: E402  (the one price, eq:gen_scc)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

MSTAR = 1.0


# ── Generic BVP solver ────────────────────────────────────────────────────────

def solve_phi_generic(mu, sigma, rho, drift_fn, x_left, x_right_mult=6.0,
                      x_right_abs=5.0, n_grid=200):
    """
    Solve the Feynman-Kac BVP:
      (sigma^2/2) phi'' + b(x; mu) phi' = rho phi
      phi(x_left) = 1,  phi(x_right) = 0

    drift_fn(x, mu) = b(x; mu).
    x_left = tipping boundary (fixed).
    x_right = max(x_right_mult * x_star, x_right_abs).
    x_star is passed via mu to the caller.
    """
    inv_hs2 = 2.0 / sigma**2

    def fun(x, y):
        phi, dphi = y
        d2phi = inv_hs2 * (rho * phi - drift_fn(x, mu) * dphi)
        return np.vstack([dphi, d2phi])

    def bc(ya, yb):
        return np.array([ya[0] - 1.0, yb[0]])

    return fun, bc


def solve_phi_at_xstar(mu, sigma, rho, drift_fn, x_left, x_star,
                       x_right_mult=6.0, x_right_abs=5.0, n_grid=200,
                       tol=1e-8, max_nodes=5000):
    """Solve BVP and return phi(x_star).

    A solve that does not converge raises, naming the case. It used to return
    np.nan, which the slope fits below then masked out without a word."""
    mu      = max(mu, 1e-12)
    x_star  = max(x_star(mu), 1e-10)
    x_right = max(x_right_mult * x_star, x_right_abs)
    x_grid  = np.linspace(x_left, x_right, n_grid)

    fun, bc = solve_phi_generic(mu, sigma, rho, drift_fn, x_left,
                                x_right_mult, x_right_abs, n_grid)

    y_guess = np.vstack([np.linspace(1, 0, n_grid),
                         np.full(n_grid, -1.0 / (x_right - x_left))])
    sol = solve_bvp(fun, bc, x_grid, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    if not sol.success:
        raise RuntimeError("solve_phi_at_xstar: solve_bvp did not converge (mu=%.6g, sigma=%g, "
                           "rho=%g, drift=%s, tol=%g, max_nodes=%d): %s"
                           % (mu, sigma, rho, drift_fn.__name__, tol, max_nodes, sol.message))
    return float(sol.sol(x_star)[0])


def _fixed_state_prices(mus, sigma, rho, drift_fn, x_left_fn, x_star_fn,
                        x_right_mult, x_right_abs, with_phi):
    """The price at FIXED state, eq:gen_scc, through price.price_fixed. The state is the
    stable state of the central budget, x*(mu), held fixed in both evaluations. Earlier versions
    this function differentiated phi(x*(mu); mu) along the stable state (np.gradient over the
    grid), which adds the motion of the state."""
    prices, phis = [], []
    for mu in mus:
        x = float(x_star_fn(mu))
        phi_at = lambda m, xx: solve_phi_at_xstar(m, sigma, rho, drift_fn, x_left_fn(m),
                                                  lambda _m: xx, x_right_mult, x_right_abs)
        prices.append(price_fixed(phi_at, mu, x))
        phis.append(phi_at(mu, x) if with_phi else np.nan)
    return np.array(prices), np.array(phis)


def compute_scc(eps_arr, sigma, rho, drift_fn, x_left, x_star_fn, mstar=MSTAR,
                x_right_mult=6.0, x_right_abs=5.0, with_phi=True):
    """Price per unit of mu at fixed state (eq:gen_scc), and phi at x*(mu)."""
    return _fixed_state_prices(mstar * np.asarray(eps_arr), sigma, rho, drift_fn,
                               lambda m: x_left, x_star_fn, x_right_mult, x_right_abs, with_phi)


def log_slope(eps, vals, eps_min=0.003, eps_max=0.05):
    mask = (eps >= eps_min) & (eps < eps_max) & np.isfinite(vals) & (vals > 0)
    if mask.sum() < 3:
        return np.nan
    return np.polyfit(np.log(eps[mask]), np.log(np.abs(vals[mask])), 1)[0]


# ── 1. Fold ───────────────────────────────────────────────────────────────────

def drift_fold(x, mu):
    return mu - x**2

def xstar_fold(mu):
    return np.sqrt(mu)

def xleft_fold(mu):
    return -np.sqrt(mu)


def scc_fold_numerical(eps_arr, sigma, rho, with_phi=True):
    """For fold, x_left moves with mu. Price at fixed state, per unit of mu."""
    return _fixed_state_prices(MSTAR * np.asarray(eps_arr), sigma, rho, drift_fold,
                               xleft_fold, xstar_fold, 6.0, 5.0, with_phi)


# ── 2. Pitchfork ──────────────────────────────────────────────────────────────

def drift_pitchfork(x, mu):
    return mu * x - x**3

def xstar_pitchfork(mu):
    return np.sqrt(max(mu, 1e-12))


def scc_pitchfork_numerical(eps_arr, sigma, rho, with_phi=True):
    return compute_scc(eps_arr, sigma, rho,
                       drift_pitchfork, 0.0, xstar_pitchfork,
                       x_right_mult=7.0, x_right_abs=5.0, with_phi=with_phi)


# ── 3. Transcritical ─────────────────────────────────────────────────────────
#
#  dX = (mu*X - X^2) dt + sigma dW
#  Equilibria for mu > 0: X=0 (unstable), X=mu (stable)
#  X*(mu) = mu, tipping at x=0
#
#  Key: X*(mu) = mu → 0 LINEARLY in mu (not sqrt(mu) like fold/pitchfork)
#  => phi(mu) ≈ 1 + mu*phi_0'(0) + ...
#  => dφ/dmu ≈ phi_0'(0) = CONSTANT ALONG x*(mu) -- the TOTAL derivative.
#  At FIXED state (eq:gen_scc) the boundary x = 0 does not move, the leading term
#  vanishes and the price falls as mu.

def drift_transcritical(x, mu):
    return mu * x - x**2

def xstar_transcritical(mu):
    return max(mu, 1e-10)


def scc_transcritical_numerical(eps_arr, sigma, rho, with_phi=True):
    return compute_scc(eps_arr, sigma, rho,
                       drift_transcritical, 0.0, xstar_transcritical,
                       x_right_mult=8.0, x_right_abs=4.0, with_phi=with_phi)


# ── Analytical pitchfork coefficient via mu=0 BVP ────────────────────────────

def pitchfork_inner_coeff(sigma, rho, eta_max=8.0, n=600, tol=1e-8, max_nodes=8000):
    """
    Solve the mu=0 BVP for pitchfork on [0, x_max]:
      (sigma^2/2) phi_0'' - x^3 phi_0' = rho phi_0
      phi_0(0) = 1,  phi_0(x_max) = 0

    SCC_pf ~ L * |phi_0'(0)| / (2 * sigma^{1/2} * sqrt(M*)) * epsilon^{-1/2}

    Returns phi_0'(0) (should be negative).
    """
    x_max  = eta_max * sigma**0.5
    x_grid = np.linspace(0.0, x_max, n)
    inv_hs2 = 2.0 / sigma**2

    def fun(x, y):
        phi, dphi = y
        d2phi = inv_hs2 * (rho * phi - (-x**3) * dphi)
        return np.vstack([dphi, d2phi])

    def bc(ya, yb):
        return np.array([ya[0] - 1.0, yb[0]])

    y_guess = np.vstack([np.linspace(1, 0, n), np.full(n, -1.0 / x_max)])
    # tol was 1e-10, at which solve_bvp exceeded max_nodes in all nine
    # (sigma, rho) cases here and in five of nine for the transcritical, and the function
    # returned (nan, nan) without a word. At 1e-8 every case converges; a failure raises.
    sol = solve_bvp(fun, bc, x_grid, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    if not sol.success:
        raise RuntimeError("solve_bvp did not converge (sigma=%g, rho=%g, tol=%g, "
                           "max_nodes=%d): %s" % (sigma, rho, tol, max_nodes, sol.message))

    phi0_prime = float(sol.sol(0.0)[1])
    # Coefficient: SCC ~ C_pf * epsilon^{-1/2}
    # C_pf = -phi0_prime / (2 * sigma^{1/2} * M*^{1/2} * M*)
    C_pf = -phi0_prime / (2.0 * sigma**0.5 * MSTAR**1.5)
    return phi0_prime, C_pf


def pitchfork_inner_coeff_formula(sigma, rho):
    """Numerical inner-layer coefficient for pitchfork."""
    _, C = pitchfork_inner_coeff(sigma, rho)
    return C


# ── Analytical transcritical coefficient via mu=0 BVP ────────────────────────

def transcritical_scc_constant(sigma, rho, x_max_mult=8.0, n=600, tol=1e-8, max_nodes=8000):
    """
    mu=0 BVP for transcritical:
      (sigma^2/2) phi_0'' - x^2 phi_0' = rho phi_0   [drift = 0*x - x^2 at mu=0]
      phi_0(0) = 1,  phi_0(x_max) = 0

    SCC_tc ~ -L * phi_0'(0) / M*  (constant, no divergence)

    Returns (phi_0'(0), C_tc) where SCC ~ L * C_tc.
    """
    x_max  = x_max_mult * max(1.0, sigma)
    x_grid = np.linspace(0.0, x_max, n)
    inv_hs2 = 2.0 / sigma**2

    def fun(x, y):
        phi, dphi = y
        d2phi = inv_hs2 * (rho * phi - (-x**2) * dphi)
        return np.vstack([dphi, d2phi])

    def bc(ya, yb):
        return np.array([ya[0] - 1.0, yb[0]])

    y_guess = np.vstack([np.linspace(1, 0, n), np.full(n, -1.0 / x_max)])
    # tol was 1e-10, at which solve_bvp exceeded max_nodes in all nine
    # (sigma, rho) cases here and in five of nine for the transcritical, and the function
    # returned (nan, nan) without a word. At 1e-8 every case converges; a failure raises.
    sol = solve_bvp(fun, bc, x_grid, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    if not sol.success:
        raise RuntimeError("solve_bvp did not converge (sigma=%g, rho=%g, tol=%g, "
                           "max_nodes=%d): %s" % (sigma, rho, tol, max_nodes, sol.message))

    phi0_prime = float(sol.sol(0.0)[1])
    C_tc = -phi0_prime / MSTAR
    return phi0_prime, C_tc


# ── Fold analytical coefficient (from Paper 1 Airy formula) ──────────────────

def fold_coeff_airy(sigma, rho):
    # ABANDONED: the Airy amplitude, replaced in the paper by Phi_fold of
    # the amplitude of app:amp (phi_fold.py). Called only from this script's __main__, which
    # run_all.sh does not run; no output of the paper depends on it.
    rt  = rho / sigma**(2/3)
    z0  = 2**(1/3) * rt
    Ai0, Aip0, _, _ = airy(z0)
    return 2**(1/3) / sigma**(2/3) / MSTAR**0.5 * (-Aip0 / Ai0)


# ── Robustness table ──────────────────────────────────────────────────────────

def robustness_all(sigmas=[0.5, 1.0, 2.0], rhos=[0.02, 0.05, 0.10], n_eps=28):
    eps = np.logspace(-2.5, 0.0, n_eps)

    print("\n{'='*70}")
    print("UNIVERSALITY TABLE — log-log slopes of SCC vs epsilon")
    print(f"{'sigma':>6}  {'rho':>5}  {'fold':>7}  {'pitch':>7}  {'tc':>7}")
    print("-" * 50)

    results = {}
    for sig in sigmas:
        for rho in rhos:
            scc_f, _  = scc_fold_numerical(eps, sig, rho)
            scc_p, _  = scc_pitchfork_numerical(eps, sig, rho)
            scc_t, _  = scc_transcritical_numerical(eps, sig, rho)

            sl_f = log_slope(eps, scc_f)
            sl_p = log_slope(eps, scc_p)
            sl_t = log_slope(eps, scc_t)

            results[(sig, rho)] = (sl_f, sl_p, sl_t)
            print(f"{sig:6.1f}  {rho:5.2f}  {sl_f:+7.3f}  {sl_p:+7.3f}  {sl_t:+7.3f}")

    print("-" * 50)
    print(f"{'theory':>6}  {'':>5}  {'':>7}  {'':>7}  {'':>7}")
    print(f"  alpha:          -0.500  -0.500   0.000")
    return results, eps


# ── Analytical coefficients table ────────────────────────────────────────────

def analytical_coefficients(sigmas=[0.5, 1.0, 2.0], rhos=[0.02, 0.05, 0.10]):
    print("\n" + "="*70)
    print("ANALYTICAL COEFFICIENTS  (SCC ~ L * C * epsilon^{-alpha})")
    print(f"{'sigma':>6}  {'rho':>5}  {'C_fold(Airy)':>14}  {'C_pf(Weber)':>13}  {'C_tc(const)':>13}")
    print("-" * 65)
    for sig in sigmas:
        for rho in rhos:
            c_fold = fold_coeff_airy(sig, rho)
            _, c_pf = pitchfork_inner_coeff(sig, rho)
            _, c_tc = transcritical_scc_constant(sig, rho)
            print(f"{sig:6.1f}  {rho:5.2f}  {c_fold:14.5f}  {c_pf:13.5f}  {c_tc:13.5f}")


# ── Main figure: all three bifurcations on one plot ──────────────────────────

def figure_universality_table(results, eps):
    sigmas = sorted(set(k[0] for k in results))
    rhos   = sorted(set(k[1] for k in results))
    sig, rho = 1.0, 0.05

    print(f"\nComputing SCC curves for figure (sigma={sig}, rho={rho})...")
    scc_f, phi_f = scc_fold_numerical(eps, sig, rho)
    scc_p, phi_p = scc_pitchfork_numerical(eps, sig, rho)
    scc_t, phi_t = scc_transcritical_numerical(eps, sig, rho)

    c_fold = fold_coeff_airy(sig, rho)
    _, c_pf = pitchfork_inner_coeff(sig, rho)
    _, c_tc = transcritical_scc_constant(sig, rho)

    fig = plt.figure(figsize=(15, 10))
    gs  = plt.GridSpec(2, 3, hspace=0.42, wspace=0.32)

    # ── panels A-C: SCC vs epsilon for each bifurcation ──────────────────────

    bif_data = [
        ('Fold',          scc_f, c_fold, eps**(-0.5),         r'$\varepsilon^{-1/2}$', 'k'),
        ('Pitchfork',     scc_p, c_pf,   eps**(-0.5),         r'$\varepsilon^{-1/2}$', 'b'),
        ('Transcritical', scc_t, c_tc,   np.ones_like(eps),   r'const.',               'g'),
    ]

    for col, (name, scc_num, c_anal, ref_shape, ref_label, color) in enumerate(bif_data):
        ax = fig.add_subplot(gs[0, col])
        valid = np.isfinite(scc_num) & (scc_num > 0)
        ax.loglog(eps[valid], scc_num[valid], color=color, lw=2, label='BVP (num.)')
        # analytical curve
        anal_curve = c_anal * ref_shape
        ax.loglog(eps, np.abs(anal_curve), 'r--', lw=1.8, label=f'Analytical')
        sl = log_slope(eps, scc_num)
        ax.set_title(f'{name}  (slope={sl:.3f})', fontsize=11)
        ax.set_xlabel(r'$\varepsilon$', fontsize=11)
        ax.set_ylabel('SCC', fontsize=11)
        ax.legend(fontsize=8)

    # ── panel D: all three normalized on one plot ─────────────────────────────

    ax = fig.add_subplot(gs[1, 0])
    for (name, scc_num, _, _, _, color) in bif_data:
        valid = np.isfinite(scc_num) & (scc_num > 0)
        if valid.sum() < 3:
            continue
        norm = scc_num[valid][-1]
        ax.loglog(eps[valid], scc_num[valid]/norm, color=color, lw=2, label=name)
    ax.loglog(eps, (eps/eps[-1])**(-0.5), 'r--', lw=1.5, label=r'$\varepsilon^{-1/2}$')
    ax.loglog(eps, np.ones_like(eps), 'k:', lw=1.0, label='const.')
    ax.set_xlabel(r'$\varepsilon$', fontsize=11)
    ax.set_ylabel('SCC (normalized)', fontsize=11)
    ax.set_title('Three bifurcation classes', fontsize=11)
    ax.legend(fontsize=8)

    # ── panel E: slope heatmap across (sigma, rho) ───────────────────────────

    ax = fig.add_subplot(gs[1, 1])
    sl_fold_arr = np.array([[results[(s,r)][0] for r in rhos] for s in sigmas])
    sl_pf_arr   = np.array([[results[(s,r)][1] for r in rhos] for s in sigmas])
    sl_tc_arr   = np.array([[results[(s,r)][2] for r in rhos] for s in sigmas])

    x = np.arange(len(rhos))
    width = 0.25
    for i, (sl_arr, label, col) in enumerate([
        (sl_fold_arr, 'Fold', 'k'),
        (sl_pf_arr,   'Pitchfork', 'b'),
        (sl_tc_arr,   'Transcritical', 'g')
    ]):
        means = sl_arr.mean(axis=0)
        ax.bar(x + (i-1)*width, means, width, label=label, color=col, alpha=0.7)
    ax.axhline(-0.5, color='r', ls='--', lw=1.5, label='theory: -1/2')
    ax.axhline(0.0,  color='gray', ls=':', lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ρ={r}' for r in rhos])
    ax.set_ylabel('mean log-log slope', fontsize=11)
    ax.set_title('Slopes across σ ∈ {0.5,1,2}', fontsize=11)
    ax.legend(fontsize=8)

    # ── panel F: phi(X*(epsilon)) for all three ───────────────────────────────

    ax = fig.add_subplot(gs[1, 2])
    for (name, _, _, _, _, color), phi in zip(bif_data, [phi_f, phi_p, phi_t]):
        valid = np.isfinite(phi) & (phi > 0)
        ax.loglog(eps[valid], phi[valid], color=color, lw=2, label=name)
    ax.set_xlabel(r'$\varepsilon$', fontsize=11)
    ax.set_ylabel(r'$\varphi(X^*(\varepsilon))$', fontsize=11)
    ax.set_title('Survival probability', fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle(
        r'Paper 2 — Universal exponent $\alpha=1/2$ for fold & pitchfork;'
        r' $\alpha=0$ for transcritical'
        f'\n(σ={sig}, ρ={rho})',
        fontsize=12
    )
    return fig


# ── Summary table for the paper ──────────────────────────────────────────────

def print_paper2_table():
    print("\n" + "="*70)
    print("TABLE 1 — Universal exponents for codimension-1 bifurcations")
    print("="*70)
    print(f"{'Bifurcation':<16}  {'alpha':>6}  {'beta':>6}  {'x_star':>10}  {'Inner eq.':>20}  {'phi_scale'}")
    print("-"*70)
    rows = [
        ("Fold",           "1/2", "3/2", "sqrt(mu)",   "Airy  u'' = z u",         "sigma^{2/3}"),
        ("Pitchfork",      "1/2", "2",   "sqrt(mu)",   "Weber u''=(C-3eta^2)u",   "sigma^{1/2}"),
        ("Transcritical",  "0",   "—",   "mu",         "Exp.  u''=(2rho/s^2)u",   "—"),
        ("Hopf (2D)",      "?",   "?",   "0 (focus)",  "2D radial BVP",           "?"),
    ]
    for name, alpha, beta, xstar, inner, phi_s in rows:
        print(f"{name:<16}  {alpha:>6}  {beta:>6}  {xstar:>10}  {inner:>20}  {phi_s}")
    print("="*70)
    print()
    print("KEY INSIGHT: alpha = 1/2 is universal for fold AND pitchfork.")
    print("  => The exponent is NOT the distinguishing feature between classes.")
    print("  => The universal signature is (beta, inner ODE, sigma-scaling of C).")
    print("Transcritical: alpha=0 because X*(mu)=mu (linear, not sqrt) => phi")
    print("  approaches 1 faster => SCC stays bounded.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    # Replication package: portability fix, same as paper2_pitchfork.py. The original
    # hard-coded WSL path did not resolve off one machine.
    outdir = os.environ.get("REPL_FIGDIR", os.getcwd())

    print("=== Paper 2: All codimension-1 bifurcations ===")
    print_paper2_table()

    print("\nAnalytical coefficients:")
    analytical_coefficients(sigmas=[0.5, 1.0, 2.0], rhos=[0.02, 0.05, 0.10])

    print("\nRobustness table...")
    results, eps = robustness_all(sigmas=[0.5, 1.0, 2.0], rhos=[0.02, 0.05, 0.10])

    print("\nGenerating universality figure...")
    fig = figure_universality_table(results, eps)
    out = os.path.join(outdir, 'fig_p2_universality_table.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f"  saved: {out}")

    # ── Detailed comparison: why transcritical alpha=0 ────────────────────────
    print("\n--- Transcritical: verifying alpha=0 ---")
    sigma, rho = 1.0, 0.05
    _, c_tc = transcritical_scc_constant(sigma, rho)
    print(f"  SCC_tc (theoretical constant) = L * {c_tc:.5f}")

    eps_test = np.logspace(-3, 0, 20)
    scc_t, _ = scc_transcritical_numerical(eps_test, sigma, rho)
    valid = np.isfinite(scc_t) & (scc_t > 0)
    if valid.sum() > 3:
        sl = log_slope(eps_test, scc_t, eps_min=0.001, eps_max=0.05)
        print(f"  Numerical slope: {sl:.4f}  [theory: 0.000]")
        c_num = np.median(scc_t[valid & (eps_test < 0.05)])
        print(f"  Numerical const: {c_num:.5f}  vs theory: {c_tc:.5f}")

    print("\nDone.")
