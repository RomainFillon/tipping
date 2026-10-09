# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   nothing on disk; a module
  FEEDS      the amplitude used by calibration_ci.py, eps_inf_ci.py and reduced_form_scc.py
  INPUTS     none
  SEED       none (deterministic ODE)
  RUNTIME    under 1 s
  IMPLEMENTS the inner equation of app:derivation; Phi(rho_tilde) = -f'(0)/f(0)

phi_fold.py — the fold amplitude Phi(rho_tilde) and the realization factor Omega(rho*T) (the paper's Omega(rho H)).

SINGLE DEFINITION, shared by calibration_ci.py and reduced_form_scc.py, so that the
level of an earlier AMOC appendix (not reported in the paper) and the levels calibration_ci.py once tabulated (no longer
reported in the paper) use literally the same Phi and
never drift apart. Moved verbatim out of calibration_ci.py (no formula change).

Phi is the logarithmic slope at the merged point of the decaying solution of the inner
Schrodinger equation eq:amp_interior of app:amp (ssec:gen_amplitude),

    f''(eta) = (2*rho_tilde - 2*eta + eta**4) * f(eta),   f(eta) -> 0 as eta -> +inf
    Phi(rho_tilde) = -f'(0) / f(0)

Two exact checks the module verifies on demand (see self_check below):
  * at rho_tilde = 0 the decaying solution is f0 = exp(-eta^3/3), so Phi(0) = 0 exactly;
  * Lemma 1 gives Phi'(0) = 2 * Integral_0^inf exp(-2 eta^3/3) d(eta) = 2*Gamma(4/3)*(3/2)^(1/3)
    = 2.04440..., which the numerical derivative must reproduce.
"""
import numpy as np
from scipy.integrate import solve_ivp

# Lemma 1 closed form: Phi'(0) = 2 * Gamma(4/3) / (2/3)**(1/3)
from math import gamma as _gamma
PHI_PRIME_ZERO_EXACT = 2.0 * _gamma(4.0 / 3.0) / (2.0 / 3.0) ** (1.0 / 3.0)

_phi_cache = {}


def phi_true_scalar(rho_t, eta_max=12.0, rtol=1e-10, atol=1e-12):
    """Compute Phi_fold(rho_tilde) for a single scalar rho_tilde via backward ODE shooting."""
    key = round(rho_t, 8)
    if key in _phi_cache:
        return _phi_cache[key]
    eta0 = eta_max

    def rhs(eta, y):
        return [y[1], (2 * rho_t - 2 * eta + eta ** 4) * y[0]]

    sol = solve_ivp(rhs, t_span=(eta0, 0.0), y0=[1.0, -eta0 ** 2],
                    method='DOP853', dense_output=True,
                    rtol=rtol, atol=atol, max_step=0.05)
    u0 = float(sol.sol(0.0)[0])
    du0 = float(sol.sol(0.0)[1])
    val = -du0 / u0 if abs(u0) > 1e-15 else np.nan
    _phi_cache[key] = val
    return val


_PHI_GRID = None


def _build_phi_grid(rt_max):
    """Tabulate Phi_fold on a dense rho_tilde grid by direct numerical integration."""
    grid = np.concatenate(([0.0], np.linspace(1e-4, max(rt_max, 0.5), 600)))
    vals = np.array([phi_true_scalar(float(g)) for g in grid])
    return grid, vals


def Phi_fold_vec(rho_tilde):
    """
    Vectorized Phi_fold from the full inner ODE (no closed form).
    Phi is computed by numerical integration (phi_true_scalar) on a dense rho_tilde grid
    and linearly interpolated; accuracy is set by the grid spacing, not by any analytic
    approximation. Used for the Monte Carlo.
    """
    global _PHI_GRID
    rt = np.atleast_1d(np.asarray(rho_tilde, dtype=float))
    rt_max = float(np.nanmax(rt)) if rt.size else 0.5
    if _PHI_GRID is None or rt_max > _PHI_GRID[0][-1]:
        _PHI_GRID = _build_phi_grid(rt_max)
    grid, vals = _PHI_GRID
    return np.interp(np.clip(rt, grid[0], grid[-1]), grid, vals)


def realization_factor(rho, T):
    """
    Discount-adjusted realization factor Omega(rho*T) = (1 - exp(-rho*T))/(rho*T).
    Applies to a linear-ramp damage profile: flow rises linearly from 0 to d_bar over [0, T],
    then stays at d_bar thereafter. PV at tipping time is d_bar/rho x Omega(rho*T).
    f -> 1 as T -> 0 (instantaneous, step profile);
    f -> 0 as T -> inf (millennial timescales, fully discounted).
    """
    rT = np.asarray(rho * T, dtype=float)
    # Numerically stable for small rT
    return np.where(rT > 1e-6, (1.0 - np.exp(-rT)) / np.maximum(rT, 1e-12), 1.0 - rT / 2.0)


def self_check(h=1e-4):
    """Return (Phi(0), Phi'(0) numerical, Phi'(0) exact). Both checks are in the docstring."""
    p0 = phi_true_scalar(0.0)
    dp0 = (phi_true_scalar(h) - phi_true_scalar(0.0)) / h
    return p0, dp0, PHI_PRIME_ZERO_EXACT


if __name__ == "__main__":
    p0, dp0, exact = self_check()
    print(f"Phi(0)        = {p0:.3e}   (exact 0)")
    print(f"Phi'(0) num   = {dp0:.5f}  (exact {exact:.5f}, rel. err {abs(dp0/exact-1):.2e})")
