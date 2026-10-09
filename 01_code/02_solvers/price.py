"""
REPLICATION HEADER
  PRODUCES   nothing on disk; a module
  FEEDS      every price the pipeline computes: validation_bvp, paper2_all_bifurcations,
             noise_reversal_frontier, moving_budget; and the
             leading law used by calibration_ci, eps_inf_ci, fig_forest_tc, reduced_form_scc,
             eps_dyn, crossover
  INPUTS     none
  SEED       none
  RUNTIME    under 1 s
  IMPLEMENTS the price of restraint, eq:gen_scc: L |d phi/d M| at the CURRENT state

price.py -- the one implementation of the price of restraint.

THE DEFINITION. Equation eq:gen_scc prices a unit of the stock by L |d phi / d M| taken at the state the
system is in. Emitting a tonne moves the budget, not the state, so the derivative is taken at a
FIXED state x. With mu = M* - M the budget, |d phi/d M| = |d phi/d mu|. An earlier version of the pipeline
differentiated phi(x*(mu); mu) along the stable state instead, which adds the motion of the state;
for the fold this doubles the price, for the transcritical and the pitchfork (boundary fixed in
the physical coordinate) it replaces a price that vanishes with one that does not.

THE SCHEME. A centred difference in log mu at fixed x:

    d phi/d mu |_x  ~  [phi(x; mu e^d) - phi(x; mu e^-d)] / [mu (e^d - e^-d)],     d = D_PRICE = 0.02.

The caller supplies phi_at(mu, x), the crossing value at state x and budget mu from its own solver,
and the state x at which the price is read (normally the stable state x*(mu) of the CENTRAL budget,
held fixed in both evaluations). The price returned is -d phi/d mu, per unit of mu and unit loss.

THE LEADING LAW at fixed state. Inside the noise layer phi(x; mu) ~ 1 - kappa (x - x_-(mu)), with
kappa = Phi(rho~)/l (the amplitude appendices: 1 - phi(x*) = 2 Phi eta* = kappa (x* - x_-) for the fold).
At fixed x, |d phi/d mu| ~ kappa |x_-'(mu)|: the speed of the BOUNDARY, not of the margin. For the
fold x_- = -sqrt(mu), |x_-'| = 1/(2 sqrt mu), so the leading law is Phi/(2 l sqrt(mu)); for a
boundary that does not move (transcritical, pitchfork mu x - x^3) it vanishes.
"""
import numpy as np

D_PRICE = 0.02


def dphi_dmu_fixed(phi_at, mu, x, d=D_PRICE):
    """d phi/d mu at fixed state x, by the centred log-step of the module docstring."""
    hi = phi_at(mu * np.exp(d), x)
    lo = phi_at(mu * np.exp(-d), x)
    return (hi - lo) / (mu * (np.exp(d) - np.exp(-d)))


def price_fixed(phi_at, mu, x, d=D_PRICE):
    """The price of restraint at fixed state, -d phi/d mu (unit loss, per unit of the budget)."""
    return -dphi_dmu_fixed(phi_at, mu, x, d)


def log_price_fixed(logphi_at, mu, x, d=D_PRICE):
    """log of price_fixed from log phi, the same difference written so that nothing underflows
    when phi is exponentially small: phi_lo - phi_hi = phi_lo (1 - exp(L_hi - L_lo))."""
    l_hi = logphi_at(mu * np.exp(d), x)
    l_lo = logphi_at(mu * np.exp(-d), x)
    gap = l_hi - l_lo
    if not gap < 0:
        return np.nan
    return l_lo + np.log(-np.expm1(gap)) - np.log(mu * (np.exp(d) - np.exp(-d)))


def lead_price_fixed(Phi, ell, boundary_speed):
    """The leading law at fixed state, kappa |x_-'(mu)| with kappa = Phi/ell."""
    return Phi / ell * abs(boundary_speed)


def boundary_power_speed(c, p, mu):
    """|x_-'(mu)| for a boundary at distance c mu^p from the merged point: c p mu^(p-1)."""
    return c * p * np.asarray(mu, dtype=float) ** (p - 1.0)


def fold_boundary_speed(mu):
    """|x_-'(mu)| for the fold, x_- = -sqrt(mu): c = 1, p = 1/2."""
    return boundary_power_speed(1.0, 0.5, mu)


def fold_lead_price(Phi, ell, mu):
    """Leading law of the fold at fixed state: Phi / (2 ell sqrt(mu)), per unit of mu."""
    return lead_price_fixed(Phi, ell, fold_boundary_speed(mu))
