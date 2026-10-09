r"""
REPLICATION HEADER
  PRODUCES   nothing on its own; a module imported by ews_calibration.py
  FEEDS      \thetaAMOChat \VarAMOChat \sigmaAMOChat, through ews_calibration.py
  INPUTS     none
  SEED       none (closed forms and a bracketed root)
  RUNTIME    negligible
  IMPLEMENTS the map from the moments of a record of ANNUAL MEANS to the continuous-time
             Ornstein-Uhlenbeck parameters (lambda, Var, sigma); sigma is not reported in the paper

annual_mean_ou.py -- what an AR(1) fitted to annual means says about a continuous process.

WHY THIS MODULE EXISTS
----------------------
The SPG index is the annual mean of a continuous sea-surface temperature, not a sample of it
once a year. For a stationary OU process dX = -lambda X dt + sigma dW with Var(X) = v =
sigma^2/(2 lambda), the annual means A_k = int_{k}^{k+1} X dt are not an AR(1) with
coefficient exp(-lambda). Their moments are

    Var(A)      = v * 2 (lambda - 1 + e^{-lambda}) / lambda^2
    Corr_1(A)   = (1 - e^{-lambda})^2 / (2 (lambda - 1 + e^{-lambda}))

(the second one is the lag-one autocorrelation of the average of an exponential covariance
over adjacent unit intervals). Inverting -ln AC1, as an earlier version of ews_calibration.py did,
treats the record as point samples: at AC1 = 0.61 it returns 0.49/yr where the continuous
rate that produces that AC1 is 0.79/yr. Var(A) likewise understates v, so sigma^2 = 2
lambda v is understated twice.

05_record_study/puissance.py simulates records this way (annual means of a
continuous OU); this module makes the calibration agree with that simulation. Its
annual_ac1 has the same closed form as this one.
"""
import numpy as np
from scipy.optimize import brentq


def _g(lam):
    """lambda - 1 + e^{-lambda}, by series near zero where the closed form cancels."""
    lam = np.asarray(lam, float)
    small = lam < 1e-3
    series = lam ** 2 / 2 - lam ** 3 / 6 + lam ** 4 / 24
    return np.where(small, series, lam - 1 + np.exp(-np.where(small, 1.0, lam)))


def annual_ac1(lam):
    """Lag-one autocorrelation of the annual means of an OU with rate lam (yr^-1)."""
    return (1 - np.exp(-lam)) ** 2 / (2 * _g(lam))


def var_ratio(lam):
    """Var(X) / Var(annual mean): the factor that recovers the process variance."""
    return lam ** 2 / (2 * _g(lam))


def lam_from_ac1(ac1, lo=1e-6, hi=50.0):
    """Invert annual_ac1. annual_ac1 is decreasing from 1 (lam -> 0) to 0 (lam -> inf)."""
    if not 0.0 < ac1 < 1.0:
        raise ValueError(f"AC1 = {ac1} has no continuous-time rate (needs 0 < AC1 < 1)")
    return brentq(lambda l: float(annual_ac1(l)) - ac1, lo, hi, xtol=1e-12)


def ou_from_annual(ac1, var_annual):
    """(lambda, process variance, sigma) from the AC1 and variance of annual means."""
    lam = lam_from_ac1(ac1)
    v = var_annual * float(var_ratio(lam))
    return lam, v, float(np.sqrt(2 * lam * v))
