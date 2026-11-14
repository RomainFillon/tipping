r"""
REPLICATION HEADER
  PRODUCES   the MACROS block ("MOVING name = value") parsed into computed.json, and the tables
  FEEDS      \epsRhoAMOC \movingRatioAtEpsRhoPct \movingRatioAtEpsRhoPctEtaOne \movingDepartLo
             \movingDepartHi \movingSlopeDeep \movingPeakRatio \movingErrNowPct
  INPUTS     theta, M*, E_0, eps_now of the AMOC through 01_calibration/eps_dyn.py (the ews and
             calibration captures); rho from inputs_literature.py; the price of price.py
  SEED       none (deterministic)
  RUNTIME    measured and printed at the end (about 2 to 4 min with the convergence check)
  IMPLEMENTS the price when the budget is being spent; and the one marcher of the draining problem,
             with the class of bifurcation as a parameter, that the step-54 scripts import

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

THE CLASS. The fold is the default. Every function of the solver (operator, steady, step, march,
march_cols, at_x, Dynamic, setting, zgrid_of, solve) takes the class of Table 1 (tab:geometries) as a
parameter, a Bifurcation: drift f(x; mu), boundary x_-(mu) and its speed x_-'(mu), stable state
x*(mu), margin d(mu) = x* - x_-, order k (l = sigma^(2/(k+1))) and recovery rate |f'(x*)|. In the
coordinate z = x - x_-(mu) the drift is f(z + x_-(mu); mu) + E_0 x_-'(mu); for the pitchfork and
the transcritical the boundary is fixed at 0 and the second term vanishes. The fold's entries are
written so that every floating-point operation is the one of the fold-only code this module
replaced (step 54): its outputs, and the macros, are unchanged to the last bit.
march accepts initial data (one column or several, advanced together); march_cols lets each
column enter at its own level. self_test() checks both, and the generic drift against the fold
equation above; the run starts with it.

THE SCHEME. Backward Euler in mu on a logarithmic mu-grid (implicit: E_0 is small, so the step
in mu/E_0 is large); finite differences in z on a sinh-stretched grid, second order where the
cell Peclet number is below 2 and upwind above it; phi = 1 at z = 0 and phi_z = 0 at the right
end (an entrance boundary for a drift -x^2: the state comes back in finite time, so phi stays
bounded and its slope vanishes far out).
The scheme in mu is a parameter, order = 1 or 2, of march, march_cols and solve. Order 1 is the
backward Euler above, the default, and the scheme of every macro this script emits. Order 2 is the
variable-step BDF2 (step_bdf2), each column taking its first level by backward Euler; it is for
the levels of cost of the step-54 scripts, whose error at order 1 is about 1.5% (step 53).
order_test() is the gate on it: the level of a cost converges at order 2 +- 0.2 in eps on three
grids, and backward Euler falls outside that window. The run starts with it.

THE PRICE, at fixed state (eq:gen_scc): P(eps) = -(1/M*) d phi/d eps at the current state
x = x*(eps) = sqrt(eps), i.e. -d phi/d mu|_x in model units (M* = 1), by price.price_fixed, the
one implementation. P_dyn applies it to the marched solution, read at fixed x on the mu-grid and
interpolated by a cubic in log mu; P_frozen applies it to the steady problem (E_0 -> 0, the
fixed-budget price of the paper) on the same z-grid, so that the two share their discretisation.

CONVERGENCE. Every number is computed twice, the second time with both grids doubled; the run
stops if the price moves by 1% or more, and the macros come from the doubled grids.

THE UNITS. M* = 1 and mu = eps. For any class, one model time unit is |f'(x*(eps_now))|/theta
years, so that the class recovers today at the measured rate theta, l = d(eps_now)/(2 eta*_now)
and sigma = l^((k+1)/2); eps_dyn is where theta |f'(x*(eps))|/|f'(x*(eps_now))| = E_0/(M* eps),
in closed form for the fold (below) and found as a root otherwise. A fold relaxes at theta = 2 sqrt(mu) at its stable state, so one
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
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_calibration"))
sys.path.insert(0, os.path.dirname(HERE))
import eps_dyn as ed                                   # noqa: E402
sys.path.insert(0, HERE)
from price import price_fixed                          # noqa: E402  (the one price, eq:gen_scc)
from inputs_literature import LITERATURE_MACROS        # noqa: E402


# ── the class of bifurcation (Table 1, tab:geometries) ────────────────────────────────────
@dataclass(frozen=True)
class Bifurcation:
    """One class of Table 1, as the marcher needs it.

    drift     f(x, mu), the deterministic dynamics at budget mu
    xl        x_-(mu), the boundary (crossing at x = x_-)
    dxl       x_-'(mu), its speed in the budget (0 for a fixed boundary)
    xs        x*(mu), the stable state
    margin    d(mu) = x*(mu) - x_-(mu), so that eta* = d/(2 l) (eq:etastar_def)
    k         the order of the class: l = sigma^(2/(k+1)), sigma = l^((k+1)/2)
    rec       |f'(x*(mu))|, the recovery rate at the stable state (model units)
    bflow     optional E_0 x_-'(mu), written in the form whose rounding the fold's original code used
    eps_dyn   optional closed form of eps_dyn in model units, eps_dyn(E_0,m); otherwise a root is found
    """
    name: str
    drift: Callable
    xl: Callable
    dxl: Callable
    xs: Callable
    margin: Callable
    k: int
    rec: Callable
    bflow: Optional[Callable] = None
    eps_dyn: Optional[Callable] = None

    def boundary_flow(self, mu, E0):
        """E_0 x_-'(mu), the drift that the moving boundary adds in the coordinate z = x - x_-(mu)."""
        return self.bflow(mu, E0) if self.bflow is not None else E0 * self.dxl(mu)


FOLD = Bifurcation(
    name="fold", drift=lambda x, mu: mu - x ** 2, xl=lambda mu: -np.sqrt(mu),
    dxl=lambda mu: -0.5 / np.sqrt(mu), xs=lambda mu: np.sqrt(mu), margin=lambda mu: 2.0 * np.sqrt(mu),
    k=2, rec=lambda mu: 2.0 * np.sqrt(mu),
    bflow=lambda mu, E0: -(E0 / (2.0 * np.sqrt(mu))),          # the fold's original rounding
    eps_dyn=lambda E0: (E0 / 2.0) ** (2.0 / 3.0))               # 2 sqrt(eps) = E_0,m / eps
PITCHFORK = Bifurcation(
    name="pitchfork", drift=lambda x, mu: mu * x - x ** 3, xl=lambda mu: 0.0, dxl=lambda mu: 0.0,
    xs=lambda mu: np.sqrt(mu), margin=lambda mu: np.sqrt(mu), k=3, rec=lambda mu: 2.0 * mu)
TRANSCRITICAL = Bifurcation(
    name="transcritical", drift=lambda x, mu: mu * x - x ** 2, xl=lambda mu: 0.0, dxl=lambda mu: 0.0,
    xs=lambda mu: mu, margin=lambda mu: mu, k=2, rec=lambda mu: mu)
CLASSES = {c.name: c for c in (FOLD, PITCHFORK, TRANSCRITICAL)}


def _cls(cls):
    """A Bifurcation, given itself or its name."""
    return cls if isinstance(cls, Bifurcation) else CLASSES[cls]


def zgrid(Z, n, dz0):
    """sinh-stretched grid on [0, Z] whose first step is about dz0."""
    a = 1.0
    for _ in range(200):                                # solve Z a /(n sinh a) = dz0 for a
        a = np.arcsinh(Z * a / (n * dz0))
    s = np.linspace(0.0, 1.0, n + 1)
    return Z * np.sinh(a * s) / np.sinh(a)


def operator(z, mu, sigma, rho, E0, cls="fold"):
    """Tridiagonal coefficients (lower, diag, upper) of L phi at the interior nodes 1..n-1,
    plus the row of the right end (Neumann). L = (s^2/2) d2 + v d1 - rho, with the drift in z
    v = f(z + x_-(mu); mu) + E_0 x_-'(mu) of the class."""
    C = _cls(cls)
    D = 0.5 * sigma ** 2
    hm, hp = z[1:-1] - z[:-2], z[2:] - z[1:-1]
    v = C.drift(z[1:-1] + C.xl(mu), mu) + (C.boundary_flow(mu, E0) if E0 > 0 else 0.0)
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


def steady(z, mu, sigma, rho, cls="fold"):
    """phi of the frozen problem (E_0 -> 0) at budget mu, on the grid z."""
    lo, di, up = operator(z, mu, sigma, rho, 0.0, cls)
    n = len(z) - 1
    ab = np.zeros((3, n))                       # unknowns phi_1 .. phi_n
    ab[1, :n - 1] = di; ab[0, 1:n] = up; ab[2, :n - 2] = lo[1:]
    ab[1, n - 1] = 1.0; ab[2, n - 2] = -1.0     # Neumann: phi_n - phi_{n-1} = 0
    rhs = np.zeros(n); rhs[0] = -lo[0] * 1.0
    return np.concatenate([[1.0], solve_banded((1, 1), ab, rhs)])


def step(z, mu_prev, mu, phi, sigma, rho, E0, cls="fold"):
    """One backward-Euler level, from mu_prev to mu, of every column of phi at once.
    phi has shape (n+1,) or (n+1, m); the result has the same shape, with phi = 1 at z = 0."""
    n = len(z) - 1
    h = mu - mu_prev
    lo, di, up = operator(z, mu, sigma, rho, E0, cls)
    ab = np.zeros((3, n))
    ab[1, :n - 1] = E0 / h - di; ab[0, 1:n] = -up; ab[2, :n - 2] = -lo[1:]
    ab[1, n - 1] = 1.0; ab[2, n - 2] = -1.0
    rhs = np.empty((n,) + phi.shape[1:]); rhs[:n - 1] = E0 / h * phi[1:n]; rhs[n - 1] = 0.0
    rhs[0] += lo[0] * 1.0
    return np.concatenate([np.ones((1,) + phi.shape[1:]), solve_banded((1, 1), ab, rhs)])


def step_bdf2(z, mu_old, mu_prev, mu, phi_old, phi, sigma, rho, E0, cls="fold"):
    """One level of the variable-step BDF2, from (mu_old, mu_prev) to mu, of every column at once:
    with h = mu - mu_prev and w = h/(mu_prev - mu_old),
        E_0 [ (1+2w)/(1+w) phi^{k+1} - (1+w) phi^k + w^2/(1+w) phi^{k-1} ] / h = L phi^{k+1}.
    Second order in the step; zero-stable for w < 1 + sqrt(2) (the geometric grids of this module
    have w about 1.005)."""
    n = len(z) - 1
    h = mu - mu_prev
    w = h / (mu_prev - mu_old)
    if not w < 1.0 + np.sqrt(2.0):
        raise SystemExit(f"[ERROR] moving_budget.py: BDF2 step ratio {w:.3f} >= 1 + sqrt(2) (unstable)")
    a0, a1, a2 = (1.0 + 2.0 * w) / (1.0 + w), 1.0 + w, w * w / (1.0 + w)
    lo, di, up = operator(z, mu, sigma, rho, E0, cls)
    ab = np.zeros((3, n))
    ab[1, :n - 1] = E0 * a0 / h - di; ab[0, 1:n] = -up; ab[2, :n - 2] = -lo[1:]
    ab[1, n - 1] = 1.0; ab[2, n - 2] = -1.0
    rhs = np.empty((n,) + phi.shape[1:])
    rhs[:n - 1] = E0 / h * (a1 * phi[1:n] - a2 * phi_old[1:n]); rhs[n - 1] = 0.0
    rhs[0] += lo[0] * 1.0
    return np.concatenate([np.ones((1,) + phi.shape[1:]), solve_banded((1, 1), ab, rhs)])


BDF2_W_MAX = 2.0   # above this step ratio a level is taken by backward Euler (a restart of BDF2)


def _check_order(order):
    if order not in (1, 2):
        raise ValueError("moving_budget: order must be 1 (backward Euler) or 2 (BDF2)")


def bdf2_ok(mus, k):
    """True if level k can be taken by BDF2: two previous levels, and a step ratio
    w = (mu_k - mu_k-1)/(mu_k-1 - mu_k-2) below BDF2_W_MAX. A grid that inserts a node between two
    geometric ones (eps_now in the step-54 scripts) can make w large; that one level is then taken
    by backward Euler, whose local error O(h^2) leaves the global order at 2."""
    return k >= 2 and (mus[k] - mus[k - 1]) < BDF2_W_MAX * (mus[k - 1] - mus[k - 2])


def march(z, mus, sigma, rho, E0, cls="fold", init=None, keep="all", order=1):
    """phi on z at every mu of the increasing grid mus, from the data init at mus[0].

    init   None (phi = 1, the condition phi(x, 0) = 1 of the problem), one column of shape (n+1,),
           or several columns of shape (n+1, m), all advanced together
    keep   "all": every level, shape (len(mus), n+1) or (len(mus), n+1, m); "last": the last level only
    order  1: backward Euler (the scheme of every macro of the paper); 2: BDF2, its first level by
           backward Euler
    """
    _check_order(order)
    phi = np.ones(len(z)) if init is None else np.array(init, dtype=float)
    if phi.shape[0] != len(z):
        raise ValueError("moving_budget.march: init must have len(z) rows")
    if keep == "all":
        out = np.empty((len(mus),) + phi.shape)
        out[0] = phi
    old = None
    for k in range(1, len(mus)):
        if order == 2 and bdf2_ok(mus, k):
            phi, old = step_bdf2(z, mus[k - 2], mus[k - 1], mus[k], old, phi, sigma, rho, E0, cls), phi
        else:
            phi, old = step(z, mus[k - 1], mus[k], phi, sigma, rho, E0, cls), phi
        if keep == "all":
            out[k] = phi
    return out if keep == "all" else phi


def march_cols(z, mus, sigma, rho, E0, start, init, cls="fold", order=1):
    """Column j enters at level start[j] with data init[j] and is advanced to the last level of
    mus; the columns at the last level, shape (n+1, len(start)), in the order of the inputs.
    At order 2 each column takes its first level by backward Euler and BDF2 after (see bdf2_ok)."""
    _check_order(order)
    start = np.asarray(start)
    cols = np.empty((len(z), 0))
    olds = np.empty((len(z), 0))
    fresh = np.zeros(0, dtype=bool)          # True for a column that has no previous level yet
    order_in = []
    for k in range(len(mus)):
        if k > 0 and cols.shape[1] > 0:
            if order == 1:
                cols = step(z, mus[k - 1], mus[k], cols, sigma, rho, E0, cls)
            else:
                nxt = np.empty_like(cols)
                euler = fresh | (not bdf2_ok(mus, k))
                fresh = euler                    # the columns that take this level by backward Euler
                if fresh.any():
                    nxt[:, fresh] = step(z, mus[k - 1], mus[k], cols[:, fresh], sigma, rho, E0, cls)
                if (~fresh).any():
                    nxt[:, ~fresh] = step_bdf2(z, mus[k - 2], mus[k - 1], mus[k], olds[:, ~fresh],
                                               cols[:, ~fresh], sigma, rho, E0, cls)
                cols, olds = nxt, cols
                fresh[:] = False
        new = np.where(start == k)[0]
        if len(new):
            add = np.column_stack([init[j] for j in new])
            cols = np.hstack([cols, add])
            olds = np.hstack([olds, np.full_like(add, np.nan)])
            fresh = np.concatenate([fresh, np.ones(len(new), dtype=bool)])
            order_in += list(new)
    return cols[:, np.argsort(order_in)]


def at_x(z, phi, mu, x, cls="fold"):
    """phi(x; mu) from the z-grid solution (z = x - x_-(mu); x + sqrt mu for the fold)."""
    return float(CubicSpline(z, phi)(x - _cls(cls).xl(mu)))


class Dynamic:
    """phi(x; mu) of the marched (moving-budget) solution at any state x and budget mu inside the
    grid: phi at fixed x on the four nearest mu-nodes, interpolated by a cubic in log mu."""

    def __init__(self, z, mus, sol, cls="fold"):
        self.z, self.mus, self.sol, self.lm = z, mus, sol, np.log(np.maximum(mus, 1e-300))
        self.cls = cls

    def __call__(self, mu, x):
        k = int(np.searchsorted(self.mus, mu))
        idx = [j for j in range(k - 2, k + 2)]
        if idx[0] < 1 or idx[-1] >= len(self.mus):
            raise SystemExit(f"[ERROR] moving_budget.py: mu = {mu} outside the marched grid")
        xs = self.lm[idx]
        ys = [at_x(self.z, self.sol[j], self.mus[j], x, self.cls) for j in idx]
        t = np.log(mu)
        out = 0.0
        for i in range(4):                       # Lagrange cubic in log mu
            w = 1.0
            for j in range(4):
                if j != i:
                    w *= (t - xs[j]) / (xs[i] - xs[j])
            out += w * ys[i]
        return out


def setting(eta_now, cls="fold"):
    """Model-unit parameters of the AMOC at eta*_now for the class (see the docstring). One model
    time unit is rec(eps_now)/theta years, so that every class recovers today at the measured rate
    theta; l = d(eps_now)/(2 eta*_now) and sigma = l^((k+1)/2)."""
    C = _cls(cls)
    e0_phys, mstar, theta, eps_now = ed.amoc_inputs()
    rho_y = float(LITERATURE_MACROS["rhoBase"][0])
    tconv = C.rec(eps_now) / theta                      # years per model time unit
    rho, E0 = rho_y * tconv, e0_phys / mstar * tconv
    l = C.margin(eps_now) / (2.0 * eta_now)
    if C.eps_dyn is not None:
        e_dyn = C.eps_dyn(E0)
    else:   # theta rec(eps)/rec(eps_now) = E_0/(M* eps), in log eps
        gap = lambda le: (np.log(theta * C.rec(np.exp(le)) / C.rec(eps_now))
                          - np.log(e0_phys / (mstar * np.exp(le))))
        e_dyn = eps_now if gap(np.log(eps_now)) <= 0 else float(np.exp(brentq(gap, np.log(1e-12),
                                                                              np.log(eps_now))))
    t_c = ed.tc_med_budget()       # the date on this budget, from the one source the planner reads too
    if t_c != eps_now * (mstar / e0_phys):
        raise SystemExit("[ERROR] moving_budget.py: eps_dyn.tc_med_budget() = %r, eps_now M*/E_0 here = %r"
                         % (t_c, eps_now * (mstar / e0_phys)))
    return dict(cls=C.name, sigma=l ** ((C.k + 1) / 2.0), rho=rho, E0=E0, l=l, eps_now=eps_now,
                eps_dyn=e_dyn, eps_rho=E0 / rho, eps_rho_phys=e0_phys / (rho_y * mstar),
                tconv=tconv, years_per_eps=mstar / e0_phys, t_c=t_c)


def zgrid_of(st, n_z):
    """The z-grid of solve: Z = d(eps_now) + 12 l + 4 on a sinh grid whose first step is
    min(2e-3, 0.02 l)."""
    Z = _cls(st["cls"]).margin(st["eps_now"]) + 12.0 * st["l"] + 4.0
    return zgrid(Z, n_z, min(2e-3, 0.02 * st["l"]))


def mu_grid(st, n_mu):
    """The budget grid of solve: 0, then n_mu levels geometric from 1e-9 to 1.12 eps_now."""
    return np.concatenate([[0.0], np.geomspace(1e-9, st["eps_now"] * 1.12, n_mu)])


def solve(eta_now, n_z, n_mu, cls="fold", with_z=False, order=1):
    """March the moving-budget problem; return the setting and the two phi evaluators (and the
    z-grid if with_z). order = 1 (backward Euler, the default and the scheme of the macros) or 2."""
    st = setting(eta_now, cls)
    z = zgrid_of(st, n_z)
    mus = mu_grid(st, n_mu)
    sol = march(z, mus, st["sigma"], st["rho"], st["E0"], cls, order=order)
    dyn = Dynamic(z, mus, sol, cls)
    frz = lambda m, x: at_x(z, steady(z, m, st["sigma"], st["rho"], cls), m, x, cls)
    return (st, dyn, frz, z) if with_z else (st, dyn, frz)


def self_test():
    """The march_cols self-tests of step 53 (pas2_fixed_ceiling.self_test), and the generic drift
    against the fold equation of the docstring. Stops the run on any failure."""
    st = setting(1.0)
    z = zgrid(2.0 * np.sqrt(st["eps_now"]) + 12.0 * st["l"] + 4.0, 400, min(2e-3, 0.02 * st["l"]))
    mus = np.concatenate([[0.0], np.geomspace(1e-9, st["eps_now"] * 1.12, 600)])
    sg, rh, E0 = st["sigma"], st["rho"], st["E0"]
    # (0) the generic drift of the fold is the drift of the docstring, mu - (z - sqrt mu)^2 - E_0/(2 sqrt mu)
    zi = z[1:-1]
    e0 = max(float(np.max(np.abs(FOLD.drift(zi + FOLD.xl(m), m) + FOLD.boundary_flow(m, E0)
                                 - (m - (zi - np.sqrt(m)) ** 2 - E0 / (2.0 * np.sqrt(m))))))
             for m in (1e-6, 1e-3, 0.1, 0.7))
    print(f"self-test 0: generic drift of the fold against the docstring's, max |diff| = {e0:.2e}")
    if e0 > 0.0:
        raise SystemExit("[ERROR] moving_budget.py: the generic drift is not the fold's")
    # (1) one column entering at 0 with phi = 1 reproduces march
    ref = march(z, mus, sg, rh, E0)[-1]
    got = march_cols(z, mus, sg, rh, E0, [0], [np.ones(len(z))])[:, 0]
    err = float(np.max(np.abs(got - ref)))
    print(f"self-test 1: march_cols against march, max |diff| = {err:.2e}")
    if err > 1e-12:
        raise SystemExit("[ERROR] moving_budget.py: march_cols does not reproduce march")
    # (2) several columns entering at different levels, in an order unlike the order of entry: each
    # must equal the same column marched alone from its own level
    st_k = [400, 0, 200, 600, 0]
    inits = [np.ones(len(z)), np.ones(len(z)), steady(z, mus[200], sg, rh), steady(z, mus[600], sg, rh),
             0.5 + 0.5 * np.ones(len(z))]
    allc = march_cols(z, mus, sg, rh, E0, st_k, inits)
    err2 = max(float(np.max(np.abs(allc[:, j] - march_cols(z, mus[st_k[j]:], sg, rh, E0, [0], [inits[j]])[:, 0])))
               for j in range(len(st_k)))
    print(f"self-test 2: five columns, mixed entry levels, against each marched alone, max |diff| = {err2:.2e}")
    if err2 > 1e-12:
        raise SystemExit("[ERROR] moving_budget.py: march_cols mixes its columns")
    # (3) march with several columns of initial data equals each column marched alone
    cols = np.column_stack([np.ones(len(z)), steady(z, mus[0] + 1e-3, sg, rh)])
    many = march(z, mus, sg, rh, E0, init=cols, keep="last")
    err3 = max(float(np.max(np.abs(many[:, j] - march(z, mus, sg, rh, E0, init=cols[:, j], keep="last"))))
               for j in range(2))
    print(f"self-test 3: march with two columns of initial data against each alone, max |diff| = {err3:.2e}")
    if err3 > 1e-12:
        raise SystemExit("[ERROR] moving_budget.py: march mixes its columns")
    # (4) the same two checks at order 2: march_cols with one column = march; mixed entry levels =
    # each column marched alone (each takes its first level by backward Euler)
    ref2 = march(z, mus, sg, rh, E0, order=2)[-1]
    got2 = march_cols(z, mus, sg, rh, E0, [0], [np.ones(len(z))], order=2)[:, 0]
    allc2 = march_cols(z, mus, sg, rh, E0, st_k, inits, order=2)
    err4 = max([float(np.max(np.abs(got2 - ref2)))]
               + [float(np.max(np.abs(allc2[:, j] - march(z, mus[st_k[j]:], sg, rh, E0, init=inits[j],
                                                           keep="last", order=2)))) for j in range(len(st_k))])
    print(f"self-test 4: order 2, march_cols against march (one column, and five with mixed entry levels), "
          f"max |diff| = {err4:.2e}")
    if err4 > 1e-12:
        raise SystemExit("[ERROR] moving_budget.py: march_cols and march disagree at order 2")


ORDER_TEST = dict(g=0.1, eps_bar=0.045, n_mu=(100, 200, 400), slope=2.0, tol=0.2)


def order_test(n_z=None):
    """GATE (step 54): the order of convergence in eps of the level of the cost c, on three grids.

    The case: fold, eta*_now = F = (eps_now/eps_dyn)^(1/2), g = 0.1, the fixed ceiling eps_bar = 0.045
    (near the optimal fixed ceiling of step 53 at this (eta*_now, g)). c = g U1 + (1 - g) U2 at
    x0 = x*(eps_now): U1 and U2 are marched from eps_bar to eps_now with data 1 and the frozen phi at
    eps_bar, on n_mu geometric intervals; the z-grid is the run's (GRID n_z) and is the same on the
    three grids, so the differences measure the error in eps alone. The order is
    p = log2(|c_N - c_2N| / |c_2N - c_4N|), the slope of the successive differences in log-log.
    BDF2 must give p = 2 +- 0.2; backward Euler is computed too and must fall OUTSIDE that window
    (the test tells the two schemes apart). Returns {order: (levels of c, p)}."""
    T = ORDER_TEST
    n_z = GRID["n_z"] if n_z is None else n_z
    e0, ms, th, en = ed.amoc_inputs()
    F = ed.amp(ed.eps_dyn(e0, ms, th, en), en)
    st = setting(F)
    z = zgrid_of(st, n_z)
    x0 = FOLD.xs(st["eps_now"])
    pb = steady(z, T["eps_bar"], st["sigma"], st["rho"])
    init = np.column_stack([np.ones(len(z)), pb])
    out = {}
    for order in (1, 2):
        cs = []
        for N in T["n_mu"]:
            mus = np.geomspace(T["eps_bar"], st["eps_now"], N + 1)
            cols = march(z, mus, st["sigma"], st["rho"], st["E0"], init=init, keep="last", order=order)
            U1, U2 = (at_x(z, cols[:, j], st["eps_now"], x0) for j in (0, 1))
            cs.append(T["g"] * U1 + (1.0 - T["g"]) * U2)
        d1, d2 = abs(cs[1] - cs[0]), abs(cs[2] - cs[1])
        out[order] = (cs, float(np.log2(d1 / d2)))
        print(f"order test, {'BDF2' if order == 2 else 'backward Euler'}: eta*_now = F = {F:.4f}, g = {T['g']:g}, "
              f"eps_bar = {T['eps_bar']:g}, n_z = {n_z}; c on {T['n_mu']} intervals = "
              + ", ".join(f"{c:.10f}" for c in cs) + f"; p = {out[order][1]:.3f}")
    if not abs(out[2][1] - T["slope"]) <= T["tol"]:
        raise SystemExit(f"[ERROR] moving_budget.py: BDF2 converges at order {out[2][1]:.3f}, "
                         f"not {T['slope']:g} +- {T['tol']:g}")
    if abs(out[1][1] - T["slope"]) <= T["tol"]:
        raise SystemExit(f"[ERROR] moving_budget.py: backward Euler gives order {out[1][1]:.3f}, inside "
                         f"the window of order 2: the test does not tell the schemes apart")
    return out


def prices_on(eps_list, dyn, frz, cls="fold"):
    """(P_dyn, P_frozen) at fixed state x = x*(eps) (sqrt(eps) for the fold), both by price.price_fixed."""
    out = []
    for e in eps_list:
        x = _cls(cls).xs(e)
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
    self_test()
    order_test()
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
