# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   nothing on disk; a module imported by fig_forest_tc.py
  FEEDS      the deadline bound on the two emission paths (\tc<element>SSPmid*, SSPhigh*)
  INPUTS     00_data/emissions/rcmip_co2_world_ssp245_ssp370.csv, the shipped extract of
             RCMIP v5.1.0 (Zenodo 4589756), Emissions|CO2, World, ssp245 and ssp370, with
             the extensions after 2100 of Meinshausen et al. (2020, GMD 13, 3571-3605)
  SEED       none
  RUNTIME    under 1 s
  IMPLEMENTS eq:tc_bound on a path whose emissions change over time: the number of years
             from T0 until emissions summed from T0 reach the remaining budget mu_now

The paper's start year. No other script fixes one: E_0 = 40 GtCO2/yr and
DT_0 = 1.2 degC are present-day values without a year attached. T0 = 2024 is the start year the
paper adopts in that case, and it is the year the scenario values below are
read at.

Annual emissions between the years the file reports (2020, then every ten years) are
interpolated linearly in time, and each year's emissions are spread evenly over that
year, so cumulative emissions are piecewise linear and the date is interpolated linearly
inside the year in which they reach mu_now. A draw whose mu_now is never reached by the
end of LAST_YEAR is returned as +inf, "not reached". On both paths emissions are zero
from 2250, so not reached by 2500 is not reached at all.

The constant path goes through the same function, with E = E_0 every year and an
unbounded horizon, and returns mu_now / E_0 exactly: that is the control
the paper needs against the existing \tc<element> macros.
"""
import os

import numpy as np
import pandas as pd

T0 = 2024
LAST_YEAR = 2500
_EXTRACT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "00_data", "emissions",
    "rcmip_co2_world_ssp245_ssp370.csv")
SCENARIOS = {"SSPmid": "ssp245", "SSPhigh": "ssp370"}


def annual_path(scenario):
    """Annual emissions in GtCO2/yr for the years T0 .. LAST_YEAR, linear in time
    between the reported years."""
    d = pd.read_csv(_EXTRACT)
    d = d.dropna(subset=[scenario])
    years = np.arange(T0, LAST_YEAR + 1)
    e = np.interp(years, d["year"].values, d[scenario].values / 1000.0)
    if d["year"].max() < LAST_YEAR:
        raise SystemExit("[ERROR] emission_paths.py: %s ends in %d, before %d"
                         % (scenario, d["year"].max(), LAST_YEAR))
    # Cumulative emissions must not fall, or the first crossing is not a single date.
    if (e < 0).any():
        raise SystemExit("[ERROR] emission_paths.py: %s has negative net CO2 emissions "
                         "in some year; the first-crossing rule needs a rule for them"
                         % scenario)
    return years, e


def deadline(mu_now, emissions):
    """Years from T0 until cumulative emissions reach mu_now (GtCO2), per draw.

    emissions: annual GtCO2/yr from T0 on. Draws never reached return +inf."""
    mu = np.asarray(mu_now, float)
    cum = np.concatenate(([0.0], np.cumsum(emissions)))
    n = np.searchsorted(cum, mu, side="left")          # cum[n-1] < mu <= cum[n]
    out = np.full(mu.shape, np.inf)
    reached = n < len(cum)
    k = np.clip(n[reached], 1, None) - 1               # the year in which it is reached
    e_k = np.asarray(emissions)[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(e_k > 0, (mu[reached] - cum[k]) / e_k, 0.0)
    out[reached] = k + frac
    out[mu <= 0.0] = 0.0
    return out


def deadline_constant(mu_now, e0):
    """The constant path: mu_now / E_0, with no horizon."""
    return np.asarray(mu_now, float) / float(e0)


def quantiles_with_unreached(t, qs=(10, 50, 90)):
    """Centiles with the unreached draws counted as +inf, by rank (no interpolation
    across the infinite tail): a centile whose rank lies among the unreached is +inf."""
    t = np.sort(np.asarray(t, float))
    n = len(t)
    res = []
    for q in qs:
        # same linear rule as np.percentile on the finite part
        pos = q / 100.0 * (n - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        if not np.isfinite(t[hi]):
            res.append(np.inf)
        else:
            res.append(t[lo] + (pos - lo) * (t[hi] - t[lo]))
    return res
