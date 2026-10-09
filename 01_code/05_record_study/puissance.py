r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/puissance.json ("puissance": power)
  FEEDS      the \rec* macros through record_detectability.py; the simulator is also imported
             by estimateurs.py, fenetres.py, horizon_np.py and mle_ditlevsen.py
  INPUTS     none (simulated records)
  SEED       one seed per cell, 1, 2, ... in the order of the job list; a record's bootstrap
             is seeded seed * 1000 + i
  RUNTIME    about 50 min on 6 worker processes
  IMPLEMENTS the power curve of the two trend estimators on records like the observed one
             (Appendix app:record)

puissance.py -- the power curve of the record.

Simulates records of 146 annual means, matching the observed record in length, in sampling
(annual means of a continuous process) and in the lag-one autocorrelation at mid-record (0.61,
that of the detrended record), the threshold being reached t_c years AFTER the end of the record
at a constant closing rate. Each simulated record goes through the estimators of
record_pipeline.py UNCHANGED:
  A: 40-year windows, linear detrending, CI by block bootstrap of the residuals (200);
  B: time-varying AR(1), CI by block bootstrap of pairs, blocks of 10 (150).
Power = P(the 95% CI of d log lambda/dt is entirely negative).

Models (continuous lambda(t)):
  lin-1/2 : linear OU, lambda = lambda_mid sqrt(mu/mu_mid)       (fold, without the nonlinearity)
  lin-1   : linear OU, lambda = lambda_mid (mu/mu_mid)           (transcritical)
  pli-3   : full fold dx = (mu - x^2) dt + sigma dW, eta* = 3 at mid-record
  pli-1.5 : same, eta* = 1.5 at mid-record; tipped paths are dropped and counted
t_c = inf is the null (constant lambda): the size test of the pipeline.
The model labels ("pli" = fold) are the keys of the output and are kept as they are.

Usage: python 01_code/05_record_study/puissance.py
"""
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_pipeline as dr                                 # noqa: E402

L = 146
YR = np.arange(1871, 1871 + L).astype(float)
BURN = 50
DT = 0.01


def annual_ac1(lam):
    """AC1 of annual means of a continuous OU with rate lam (per year)."""
    return (1 - np.exp(-lam)) ** 2 / (2 * (lam - 1 + np.exp(-lam)))


def lam_for_ac1(target):
    lo, hi = 0.01, 5.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if annual_ac1(m) > target else (lo, m)
    return 0.5 * (lo + hi)


def simulate(model, tc, lam_mid, R, rng):
    """R annual-mean records; time s runs from -BURN to L, the threshold at s = L + tc."""
    spy = int(round(1 / DT)); nsteps = (BURN + L) * spy
    mu_mid = (lam_mid / 2) ** 2
    nu = 0.0 if not np.isfinite(tc) else mu_mid / (tc + L / 2)

    def mu_at(s):
        return mu_mid if nu == 0 else nu * (tc + L - s)
    x = np.zeros(R); alive = np.ones(R, bool); ann = np.zeros((R, L)); acc = np.zeros(R)
    if model.startswith("pli"):
        eta = float(model.split("-")[1])
        sig = (np.sqrt(mu_mid) / eta) ** 1.5
        x[:] = np.sqrt(mu_at(-BURN))
    else:
        sig = 1.0
    for k in range(nsteps):
        s = -BURN + k * DT
        mu = mu_at(s)
        if model == "lin-1/2":
            lam = 2 * np.sqrt(mu); x += -lam * x * DT + sig * np.sqrt(DT) * rng.standard_normal(R)
        elif model == "lin-1":
            lam = lam_mid * mu / mu_mid; x += -lam * x * DT + sig * np.sqrt(DT) * rng.standard_normal(R)
        else:
            x += (mu - x ** 2) * DT + sig * np.sqrt(DT) * rng.standard_normal(R)
            dead = x < -np.sqrt(max(mu, 1e-12)) - 3 * sig ** (2 / 3)
            alive &= ~dead; x[dead] = np.sqrt(max(mu, 1e-12))
        if k >= BURN * spy:
            acc += x
            if (k + 1) % spy == 0:
                ann[:, (k + 1) // spy - BURN - 1] = acc / spy; acc[:] = 0
    return ann[alive], alive.mean()


def pipeline_A(y, seed):
    """Estimator A on one record, exactly as on the observed record (w=40, linear)."""
    dr.rng = np.random.default_rng(seed)
    t = YR
    base = dr.poly_dt(y, t, 1)
    x0, x1 = base[:-1], base[1:]; tau = t[1:] - t[1:].mean()
    p = dr.tvar_fit(x0, x1, tau)
    e_std = (x1 - np.exp(-np.exp(p[0] + p[1] * tau)) * x0) / np.exp(p[2] + p[3] * tau)
    trend = y - base
    sl, sv, _ = dr.windowed(y, t, "lineaire", 40)
    bs = []
    for _ in range(200):
        xs = dr.resid_block_series(p, base[0], tau, e_std, 10)
        bs.append(dr.windowed(xs + trend, t, "lineaire", 40)[0])
    bs = np.array(bs)
    # AC1 >= 1 in a window makes log(lambda) undefined: the estimator does not provide for it
    # (it never occurs on the observed record). NaN replicates are dropped; a NaN point
    # estimate is a failure of the pipeline, counted as a non-rejection and reported.
    if not np.isfinite(sl) or np.isfinite(bs).sum() < 50:
        return np.nan, np.nan, np.nan
    d_lo, d_hi = np.nanpercentile(bs - np.nanmedian(bs), [2.5, 97.5])
    return sl, sl - d_hi, sl - d_lo


def pipeline_B(y, seed):
    dr.rng = np.random.default_rng(seed)
    r = dr.poly_dt(y, YR, 1)
    x0, x1 = r[:-1], r[1:]; tau = YR[1:] - YR[1:].mean()
    p, P = dr.pairs_block_boot(x0, x1, tau, 10, 150)
    lo, hi = np.percentile(P[:, 1], [2.5, 97.5])
    return p[1], lo, hi


def task(args):
    model, tc, est, R, seed = args
    rng = np.random.default_rng(seed)
    recs, surv = simulate(model, tc, LAM_MID_G, R, rng)
    f = pipeline_A if est == "A" else pipeline_B
    res = [f(y, seed * 1000 + i) for i, y in enumerate(recs)]
    res = np.array(res) if len(res) else np.zeros((0, 3))
    ok = np.isfinite(res[:, 0]) if len(res) else np.zeros(0, bool)
    return dict(model=model, tc=tc, est=est, n=len(res), surv=float(surv), nfail=int((~ok).sum()),
                power=float(np.mean(np.where(ok, res[:, 2], 1.0) < 0)) if len(res) else float("nan"),
                wrong=float(np.mean(np.where(ok, res[:, 1], -1.0) > 0)) if len(res) else float("nan"),
                slope_mean=float(np.nanmean(res[:, 0])) if ok.any() else float("nan"),
                slope_sd=float(np.nanstd(res[:, 0])) if ok.any() else float("nan"),
                ci_halfwidth=float(np.nanmean(res[:, 2] - res[:, 1]) / 2) if ok.any() else float("nan"))


LAM_MID_G = lam_for_ac1(0.61)


def true_slope(model, tc, lam_mid):
    """OLS slope of the true log lambda(t) on time over the record (continuous-time lambda)."""
    if not np.isfinite(tc):
        return 0.0
    s = np.arange(L) + 0.5
    mu = (lam_mid / 2) ** 2 / (tc + L / 2) * (tc + L - s)
    lam = 2 * np.sqrt(mu) if model != "lin-1" else lam_mid * mu / (lam_mid / 2) ** 2
    return np.polyfit(s, np.log(lam), 1)[0]


def main():
    t0 = time.time()
    print(f"continuous lambda at mid-record for AC1(annual means) = 0.61: {LAM_MID_G:.4f} /yr "
          f"(-ln 0.61 = {-np.log(0.61):.4f}: the annual estimator understates the continuous rate)")
    TCS = [3, 5, 10, 20, 40, 80, 156, 400, np.inf]
    jobs = []
    seed = 1
    for model in ["lin-1/2", "lin-1", "pli-3", "pli-1.5"]:
        for tc in TCS:
            jobs.append((model, tc, "A", 100, seed)); seed += 1
    for tc in TCS:
        jobs.append(("lin-1/2", tc, "B", 60, seed)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)
    for o in out:
        o["true_slope"] = true_slope(o["model"], o["tc"], LAM_MID_G)
        print(f"{o['est']} {o['model']:8s} t_c={o['tc']:>5}: n={o['n']:3d} failures={o['nfail']} survival={o['surv']:.2f} "
              f"true slope={100*o['true_slope']:+.2f}%/yr  estimated={100*o['slope_mean']:+.2f}±{100*o['slope_sd']:.2f}  "
              f"half-CI={100*o['ci_halfwidth']:.2f}  power={o['power']:.2f}  wrong sign={o['wrong']:.2f}")
    dr.dump_json(out, "puissance.json", indent=1, default=float)
    print(f"duration {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
