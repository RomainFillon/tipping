r"""
REPLICATION HEADER
  PRODUCES   the MACROS block ("MOVING name = value") parsed into computed.json, and the tables
  FEEDS      \epsRhoAMOC \movingRatioAtEpsRhoPct \movingRatioAtEpsRhoPctEtaOne \movingDepartLo
             \movingDepartHi \movingSlopeDeep \movingPeakRatio \movingErrNowPct
  INPUTS     theta, M*, E_0, eps_now of the AMOC through 01_calibration/eps_dyn.py (the ews and
             calibration captures); rho from inputs_literature.py; the price of price.py
  SEED       none (deterministic)
  RUNTIME    measured and printed at the end (about 2 to 4 min with the convergence check)
  IMPLEMENTS the price when the budget is being spent

moving_budget.py -- the price of restraint when the budget is spent at a constant rate.

THE PROBLEM. With emissions at a constant rate the budget mu falls at the rate E_0 (in units of
M*, model time). The crossing value phi(x, mu) = E[exp(-rho tau)] then solves

    E_0 d phi/d mu = (sigma^2/2) phi_xx + f(x; mu) phi_x - rho phi ,      f = mu - x^2 (fold),
    phi(x_-(mu), mu) = 1 ,   phi bounded as x -> +inf ,   phi(x, 0) = 1 ,

where x_-(mu) = -sqrt(mu) is the boundary and phi(x, 0) = 1 says that crossing is immediate once
the budget is exhausted. It is parabolic forward in mu, so it is marched up from mu = 0.

THE COORDINATE. z = x - x_-(mu) = x + sqrt(mu) >= 0 fixes the boundary at z = 0. Since
d phi/d mu|_x = d phi/d mu|_z + phi_z /(2 sqrt mu),

    E_0 d phi/d mu|_z = (sigma^2/2) phi_zz + [mu - (z - sqrt mu)^2 - E_0/(2 sqrt mu)] phi_z - rho phi.

THE SCHEME. Backward Euler in mu on a logarithmic mu-grid (implicit: E_0 is small, so the step
in mu/E_0 is large); finite differences in z on a sinh-stretched grid, second order where the
cell Peclet number is below 2 and upwind above it; phi = 1 at z = 0 and phi_z = 0 at the right
end (an entrance boundary for a drift -x^2: the state comes back in finite time, so phi stays
bounded and its slope vanishes far out).

THE PRICE, at fixed state (eq:gen_scc): P(eps) = -(1/M*) d phi/d eps at the current state
x = x*(eps) = sqrt(eps), i.e. -d phi/d mu|_x in model units (M* = 1), by price.price_fixed, the
one implementation. P_dyn applies it to the marched solution, read at fixed x on the mu-grid and
interpolated by a cubic in log mu; P_frozen applies it to the steady problem (E_0 -> 0, the
fixed-budget price of the paper) on the same z-grid, so that the two share their discretisation.

CONVERGENCE. Every number is computed twice, the second time with both grids doubled; the run
stops if the price moves by 1% or more, and the macros come from the doubled grids.

THE UNITS. M* = 1 and mu = eps. A fold relaxes at theta = 2 sqrt(mu) at its stable state, so one
model time unit is (2 sqrt(eps_now)/theta) years, and every rate r in 1/yr becomes
r x 2 sqrt(eps_now)/theta: rho_m and E_0,m = (E_0/M*) x 2 sqrt(eps_now)/theta. eta*_now =
sqrt(eps_now)/l is not identified (it needs the physical margin); for each value swept,
l = sqrt(eps_now)/eta*_now and sigma = l^(3/2). The two scales of the paper are then
eps_dyn = (E_0,m/2)^(2/3) (the appendix on where the elements sit: the budget drains as fast as the element recovers) and
eps_rho = E_0,m/rho_m = E_0/(rho M*) (the draining-budget appendix: as fast as the planner discounts).
"""
import os
import sys
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_calibration"))
sys.path.insert(0, os.path.dirname(HERE))
import eps_dyn as ed                                   # noqa: E402
sys.path.insert(0, HERE)
from price import price_fixed                          # noqa: E402  (the one price, eq:gen_scc)
from inputs_literature import LITERATURE_MACROS        # noqa: E402


def zgrid(Z, n, dz0):
    """sinh-stretched grid on [0, Z] whose first step is about dz0."""
    a = 1.0
    for _ in range(200):                                # solve Z a /(n sinh a) = dz0 for a
        a = np.arcsinh(Z * a / (n * dz0))
    s = np.linspace(0.0, 1.0, n + 1)
    return Z * np.sinh(a * s) / np.sinh(a)


def operator(z, mu, sigma, rho, E0):
    """Tridiagonal coefficients (lower, diag, upper) of L phi at the interior nodes 1..n-1,
    plus the row of the right end (Neumann). L = (s^2/2) d2 + v d1 - rho."""
    D = 0.5 * sigma ** 2
    hm, hp = z[1:-1] - z[:-2], z[2:] - z[1:-1]
    sq = np.sqrt(mu)
    v = mu - (z[1:-1] - sq) ** 2 - (E0 / (2.0 * sq) if E0 > 0 else 0.0)
    # diffusion
    lo = 2.0 * D / (hm * (hm + hp)); up = 2.0 * D / (hp * (hm + hp)); di = -(lo + up)
    # convection: centred (second order) where the cell Peclet number is small, upwind above
    pe = np.abs(v) * np.maximum(hm, hp) / D
    cen = pe < 2.0
    lc = -hp / (hm * (hm + hp)); dc = (hp - hm) / (hm * hp); uc = hm / (hp * (hm + hp))
    lo = lo + np.where(cen, v * lc, np.where(v < 0, -v / hm, 0.0))
    up = up + np.where(cen, v * uc, np.where(v > 0, v / hp, 0.0))
    di = di + np.where(cen, v * dc, np.where(v > 0, -v / hp, v / hm))
    di = di - rho
    return lo, di, up


def steady(z, mu, sigma, rho):
    """phi of the frozen problem (E_0 -> 0) at budget mu, on the grid z."""
    lo, di, up = operator(z, mu, sigma, rho, 0.0)
    n = len(z) - 1
    ab = np.zeros((3, n))                       # unknowns phi_1 .. phi_n
    ab[1, :n - 1] = di; ab[0, 1:n] = up; ab[2, :n - 2] = lo[1:]
    ab[1, n - 1] = 1.0; ab[2, n - 2] = -1.0     # Neumann: phi_n - phi_{n-1} = 0
    rhs = np.zeros(n); rhs[0] = -lo[0] * 1.0
    return np.concatenate([[1.0], solve_banded((1, 1), ab, rhs)])


def march(z, mus, sigma, rho, E0):
    """phi on z at every mu of the increasing grid mus, from phi = 1 at mus[0]."""
    n = len(z) - 1
    out = np.empty((len(mus), n + 1))
    phi = np.ones(n + 1)
    out[0] = phi
    for k in range(1, len(mus)):
        h = mus[k] - mus[k - 1]
        lo, di, up = operator(z, mus[k], sigma, rho, E0)
        ab = np.zeros((3, n))
        ab[1, :n - 1] = E0 / h - di; ab[0, 1:n] = -up; ab[2, :n - 2] = -lo[1:]
        ab[1, n - 1] = 1.0; ab[2, n - 2] = -1.0
        rhs = np.empty(n); rhs[:n - 1] = E0 / h * phi[1:n]; rhs[n - 1] = 0.0
        rhs[0] += lo[0] * 1.0
        phi = np.concatenate([[1.0], solve_banded((1, 1), ab, rhs)])
        out[k] = phi
    return out


def at_x(z, phi, mu, x):
    """phi(x; mu) from the z-grid solution (z = x + sqrt mu)."""
    return float(CubicSpline(z, phi)(x + np.sqrt(mu)))


class Dynamic:
    """phi(x; mu) of the marched (moving-budget) solution at any state x and budget mu inside the
    grid: phi at fixed x on the four nearest mu-nodes, interpolated by a cubic in log mu."""

    def __init__(self, z, mus, sol):
        self.z, self.mus, self.sol, self.lm = z, mus, sol, np.log(np.maximum(mus, 1e-300))

    def __call__(self, mu, x):
        k = int(np.searchsorted(self.mus, mu))
        idx = [j for j in range(k - 2, k + 2)]
        if idx[0] < 1 or idx[-1] >= len(self.mus):
            raise SystemExit(f"[ERROR] moving_budget.py: mu = {mu} outside the marched grid")
        xs = self.lm[idx]
        ys = [at_x(self.z, self.sol[j], self.mus[j], x) for j in idx]
        t = np.log(mu)
        out = 0.0
        for i in range(4):                       # Lagrange cubic in log mu
            w = 1.0
            for j in range(4):
                if j != i:
                    w *= (t - xs[j]) / (xs[i] - xs[j])
            out += w * ys[i]
        return out


def setting(eta_now):
    """Model-unit parameters of the AMOC at eta*_now (see the docstring)."""
    e0_phys, mstar, theta, eps_now = ed.amoc_inputs()
    rho_y = float(LITERATURE_MACROS["rhoBase"][0])
    tconv = 2.0 * np.sqrt(eps_now) / theta               # years per model time unit
    rho, E0 = rho_y * tconv, e0_phys / mstar * tconv
    l = np.sqrt(eps_now) / eta_now
    return dict(sigma=l ** 1.5, rho=rho, E0=E0, l=l, eps_now=eps_now,
                eps_dyn=(E0 / 2.0) ** (2.0 / 3.0), eps_rho=E0 / rho,
                eps_rho_phys=e0_phys / (rho_y * mstar))


def solve(eta_now, n_z, n_mu):
    """March the moving-budget problem; return the setting, the grid and the two phi evaluators."""
    st = setting(eta_now)
    Z = 2.0 * np.sqrt(st["eps_now"]) + 12.0 * st["l"] + 4.0
    z = zgrid(Z, n_z, min(2e-3, 0.02 * st["l"]))
    mus = np.concatenate([[0.0], np.geomspace(1e-9, st["eps_now"] * 1.12, n_mu)])
    sol = march(z, mus, st["sigma"], st["rho"], st["E0"])
    dyn = Dynamic(z, mus, sol)
    frz = lambda m, x: at_x(z, steady(z, m, st["sigma"], st["rho"]), m, x)
    return st, dyn, frz


def prices_on(eps_list, dyn, frz):
    """(P_dyn, P_frozen) at fixed state x = sqrt(eps), both by price.price_fixed."""
    out = []
    for e in eps_list:
        x = np.sqrt(e)
        out.append((price_fixed(dyn, e, x), price_fixed(frz, e, x)))
    return np.array(out)


def crossings(eps, ratio, level):
    """eps (from eps_now downward) at which the ratio first crosses the level, log-interpolated."""
    for i in range(1, len(eps)):
        a, b = ratio[i - 1] - level, ratio[i] - level
        if a == 0 or a * b < 0:
            t = a / (a - b)
            return float(np.exp(np.log(eps[i - 1]) + t * (np.log(eps[i]) - np.log(eps[i - 1]))))
    return np.nan


def fmt_sig(x, n=2):
    if x == 0:
        return "0"
    dec = max(0, n - 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{dec}f}"


ETAS = (0.1, 0.25, 0.5, 1.0)
GRID = dict(n_z=1500, n_mu=4000)          # the run; the convergence check doubles both
D_SLOPE = 0.05

if __name__ == "__main__":
    T0 = time.time()
    print("=" * 78)
    print("MOVING BUDGET (AMOC, constant emissions): P_dyn / P_frozen at fixed state")
    print("=" * 78)
    res = {}
    for eta in ETAS:
        st, dyn, frz = solve(eta, GRID["n_z"] * 2, GRID["n_mu"] * 2)
        st0, dyn0, frz0 = solve(eta, GRID["n_z"], GRID["n_mu"])
        e_now, e_dyn, e_rho = st["eps_now"], st["eps_dyn"], st["eps_rho"]
        grid = np.unique(np.concatenate([np.geomspace(e_now, e_dyn / 10.0, 40), [e_rho, e_dyn]]))[::-1]
        P = prices_on(grid, dyn, frz)
        P0 = prices_on(grid, dyn0, frz0)
        conv = float(np.max(np.abs(P[:, 0] / P0[:, 0] - 1)))
        convf = float(np.max(np.abs(P[:, 1] / P0[:, 1] - 1)))
        print(f"\n  eta*_now = {eta:g}: sigma = {st['sigma']:.4g}, rho_m = {st['rho']:.4g}, "
              f"E0_m = {st['E0']:.4g}, eps_dyn = {e_dyn:.4f}, eps_rho = {e_rho:.4f}")
        print(f"  convergence (grids doubled): max change of P_dyn {100*conv:.3f}%, of P_frozen {100*convf:.3f}%")
        if not (conv < 0.01 and convf < 0.01):
            raise SystemExit(f"[ERROR] moving_budget.py: doubling the grids moves the price by "
                             f"{100*max(conv, convf):.2f}% (> 1%) at eta*_now = {eta}")
        r = P[:, 0] / P[:, 1]
        print(f"  {'eps':>9} {'eps/eps_dyn':>11} {'P_dyn':>12} {'P_frozen':>12} {'ratio':>9}")
        for e, (pd, pf), rr in zip(grid, P, r):
            print(f"  {e:9.5f} {e/e_dyn:11.3f} {pd:12.5g} {pf:12.5g} {rr:9.4f}")
        sl = (np.log(price_fixed(dyn, e_dyn / 10 * np.exp(D_SLOPE), np.sqrt(e_dyn / 10)))
              - np.log(price_fixed(dyn, e_dyn / 10 * np.exp(-D_SLOPE), np.sqrt(e_dyn / 10)))) / (2 * D_SLOPE)
        res[eta] = dict(st=st, grid=grid, r=r, cross=crossings(grid, r, 1.1),
                        at_rho=float(r[np.argmin(np.abs(grid - e_rho))]), at_now=float(r[0]),
                        peak=float(np.max(r[grid < e_dyn])), slope_deep=float(sl))
        print(f"  ratio crosses 1.1 at eps = {res[eta]['cross']:.4g} = {res[eta]['cross']/e_dyn:.3f} eps_dyn; "
              f"ratio at eps_rho {res[eta]['at_rho']:.4f}, at eps_now {res[eta]['at_now']:.4f}; "
              f"largest ratio below eps_dyn {res[eta]['peak']:.4f}; local slope of log P_dyn at "
              f"eps_dyn/10: {sl:.3f}", flush=True)

    st1 = res[ETAS[0]]["st"]
    dev_rho = max(abs(res[e]["at_rho"] - 1) for e in ETAS if e <= 0.5)
    dev_rho_one = abs(res[1.0]["at_rho"] - 1)
    departs = [res[e]["cross"] / res[e]["st"]["eps_dyn"] for e in ETAS]
    peak = max(res[e]["peak"] for e in ETAS)
    err_now = max(abs(res[e]["at_now"] - 1) for e in ETAS)
    print(f"\n  across eta*_now: |ratio - 1| at eps_rho <= {100*dev_rho:.2f}% (eta* <= 0.5), "
          f"{100*dev_rho_one:.1f}% at eta* = 1; the ratio crosses 1.1 between "
          f"{min(departs):.2f} and {max(departs):.2f} eps_dyn; largest ratio below eps_dyn {peak:.3f}; "
          f"|ratio - 1| at eps_now <= {100*err_now:.1f}%")
    print(f"\nruntime: {time.time() - T0:.0f} s")
    print("MACROS")
    print(f"MOVING epsRhoAMOC = {fmt_sig(st1['eps_rho_phys'])}")
    print(f"MOVING movingRatioAtEpsRhoPct = {fmt_sig(100 * dev_rho)}")
    print(f"MOVING movingRatioAtEpsRhoPctEtaOne = {fmt_sig(100 * dev_rho_one)}")
    print(f"MOVING movingDepartLo = {min(departs):.1f}")
    print(f"MOVING movingDepartHi = {max(departs):.1f}")
    print(f"MOVING movingSlopeDeep = {res[1.0]['slope_deep']:.2f}")
    print(f"MOVING movingPeakRatio = {peak:.1f}")
    print(f"MOVING movingErrNowPct = {fmt_sig(100 * err_now)}")
    print("END MACROS")
