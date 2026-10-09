r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/estimateurs.json ("estimateurs": estimators)
  FEEDS      the \rec* macros through record_detectability.py (saturation of the trend
             statistic); fenetres.py imports its tabulation
  INPUTS     none (simulated records)
  SEED       1300, 1301, ... one per cell
  RUNTIME    about 1-2 min on 6 worker processes
  IMPLEMENTS trend, level and ramp statistics of the recovery rate on the same simulated
             records, with oracle critical values (Appendix app:record)

estimateurs.py -- trend, level and ramp on the same simulated records.

Records: puissance.simulate, UNCHANGED (146 annual means of a continuous OU, AC1 = 0.61 at
mid-record, threshold t_c years after the end, closing linear in mu):
  lin-1/2 : lambda = lambda_mid sqrt(mu/mu_mid)   (a = 1/2, linearised fold)
  lin-1   : lambda = lambda_mid mu/mu_mid         (a = 1,   transcritical)
Each record: global linear detrending (record_pipeline.poly_dt), sliding windows of w = 40 years
as in estimator A, AC1 per window (record_pipeline.ar1), lambda_w by the CORRECTED annual-mean
reading (annual_mean_ou.lam_from_ac1, tabulated). AC1 outside [0.02, 0.999]: clipped and
counted (otherwise lambda_w = 0 and log = -inf).

Statistics (for all of them, more negative = approaching):
  T      OLS slope of log lambda_w on the window centre           (Dakos et al. 2008)
  N_w    log lambda_last - log lambda_first, windows of w = 40, 20, 10 years   (level)
  R_1/2  OLS slope of lambda_w^2 on time: a linear ramp in lambda^2, the linear-ramp fold model
         of Ditlevsen & Ditlevsen 2023 read by their route (i) (windows, then ramp)
  R_1    OLS slope of lambda_w on time: the same ramp, transcritical form

Test: ORACLE. Critical value = 5% quantile of the statistic under the null (t_c = inf, constant
lambda, 4000 simulated records), one-sided rejection. Size 5% by construction. This is the
maximal power of each statistic at known lambda_mid, hence an UPPER BOUND on any feasible
(bootstrap) version: if the oracle saturates, the feasible version saturates too.

Also from the same pass: the standard deviation of T across records as a function of t_c; the
bias of lambda_w in the first and last window against the true mean of lambda over the window;
the share of clipped windows.

Length control: t_c = 1 and the null at L = 292 years.

Usage: python 01_code/05_record_study/estimateurs.py
"""
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_calibration"))
import annual_mean_ou as am                                  # noqa: E402
import record_pipeline as dr                                 # noqa: E402
import puissance as pu                                       # noqa: E402

W = 40
NW = (40, 20, 10)
AC_LO, AC_HI = 0.02, 0.999
_G = np.logspace(-5, np.log10(40.0), 40000)
_A = am.annual_ac1(_G)                                       # decreasing in lambda


def lam_tab(ac1):
    """annual_mean_ou.lam_from_ac1, tabulated (checked against it in main)."""
    return np.interp(-ac1, -_A, _G)


def windows_ac1(r, w):
    return np.array([dr.ar1(r[i:i + w]) for i in range(len(r) - w + 1)])


def true_lambda(model, tc, L, lam_mid):
    """lambda(s) on a fine grid, averaged over each record year (continuous-time truth)."""
    s = np.arange(0, L, 0.01) + 0.005
    mu_mid = (lam_mid / 2) ** 2
    if not np.isfinite(tc):
        lam = np.full_like(s, lam_mid)
    else:
        mu = mu_mid / (tc + L / 2) * (tc + L - s)
        lam = 2 * np.sqrt(mu) if model == "lin-1/2" else lam_mid * mu / mu_mid
    return lam.reshape(L, 100).mean(1)


def stats_one(y, L):
    t = np.arange(L, dtype=float)
    r = dr.poly_dt(y, t, 1)
    out, clip = {}, 0
    ac = windows_ac1(r, W)
    clip += int(np.sum((ac < AC_LO) | (ac > AC_HI)))
    lam = lam_tab(np.clip(ac, AC_LO, AC_HI))
    c = np.arange(len(lam)) + W / 2
    out["T"] = np.polyfit(c, np.log(lam), 1)[0]
    out["R_1/2"] = np.polyfit(c, lam ** 2, 1)[0]
    out["R_1"] = np.polyfit(c, lam, 1)[0]
    out["lam_first"], out["lam_last"], out["lam_mid_w"] = lam[0], lam[-1], lam[len(lam) // 2]
    for w in NW:
        a0 = np.clip(dr.ar1(r[:w]), AC_LO, AC_HI); a1 = np.clip(dr.ar1(r[-w:]), AC_LO, AC_HI)
        clip += int(a0 in (AC_LO, AC_HI)) + int(a1 in (AC_LO, AC_HI))
        out[f"N_{w}"] = np.log(lam_tab(a1)) - np.log(lam_tab(a0))
    out["clip"] = clip
    return out


def task(args):
    model, tc, R, seed, L = args
    pu.L = L; pu.YR = np.arange(L).astype(float)             # simulate reads the module globals
    rng = np.random.default_rng(seed)
    recs, surv = pu.simulate(model, tc, pu.LAM_MID_G, R, rng)
    rows = [stats_one(y, L) for y in recs]
    keys = rows[0].keys()
    return dict(model=model, tc=tc, L=L, seed=seed, data={k: np.array([r[k] for r in rows]).tolist() for k in keys})


STATS = ["T", "N_40", "N_20", "N_10", "R_1/2", "R_1"]


def main():
    t0 = time.time()
    for a in (0.2, 0.5, 0.61, 0.9, 0.99):
        assert abs(lam_tab(np.array(a)) - am.lam_from_ac1(a)) / am.lam_from_ac1(a) < 1e-4
    print("tabulation of lam_from_ac1: relative gap < 1e-4 on AC1 in {0.2 ... 0.99} [OK]")
    TCS = [0.5, 1, 2, 5, 10, 20, 40, 80, 156]
    jobs, seed = [], 1300
    for L in (146,):
        for k in range(4):                                     # null: 4 x 1000
            jobs.append(("lin-1", np.inf, 1000, seed, L)); seed += 1
        for m in ("lin-1/2", "lin-1"):
            for tc in TCS:
                jobs.append((m, tc, 1000, seed, L)); seed += 1
    for k in range(2):
        jobs.append(("lin-1", np.inf, 1000, seed, 292)); seed += 1
    for m in ("lin-1/2", "lin-1"):
        jobs.append((m, 1, 1000, seed, 292)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)

    res = dict(TCS=TCS, W=W, cells=[], null={}, crit={})
    for L in (146, 292):
        nul = [o for o in out if o["L"] == L and not np.isfinite(o["tc"])]
        D = {k: np.concatenate([np.array(o["data"][k]) for o in nul]) for k in STATS}
        crit = {k: float(np.quantile(D[k], 0.05)) for k in STATS}
        res["crit"][L] = crit
        res["null"][L] = {k: dict(mean=float(D[k].mean()), sd=float(D[k].std())) for k in STATS}
        print(f"\nL = {L}: null, {len(D['T'])} records; 5% critical values: "
              + ", ".join(f"{k} {crit[k]:+.4g}" for k in STATS))
        lam_true_mid = pu.LAM_MID_G
        nl = np.concatenate([np.array(o["data"]["lam_mid_w"]) for o in nul])
        print(f"  bias of lambda_w under the null (true lambda {lam_true_mid:.4f}): median {np.median(nl):.4f} "
              f"({100*(np.median(nl)/lam_true_mid-1):+.1f} %), mean {nl.mean():.4f}")
    print("\nORACLE power (binomial 95% CI +- ~0.03 at n = 1000) and the same-pass diagnostics:")
    hdr = "  model     t_c  L   | " + " ".join(f"{k:>6s}" for k in STATS) + " | SD(T) %/yr | bias lam_w first / last | clipped/rec."
    print(hdr)
    for o in sorted([o for o in out if np.isfinite(o["tc"])], key=lambda o: (o["L"], o["model"], o["tc"])):
        L = o["L"]; d = {k: np.array(v) for k, v in o["data"].items()}
        pw = {k: float(np.mean(d[k] < res["crit"][L][k])) for k in STATS}
        lt = true_lambda(o["model"], o["tc"], L, pu.LAM_MID_G)
        tf, tl = lt[:W].mean(), lt[-W:].mean()
        b_first = float(np.median(d["lam_first"]) / tf - 1); b_last = float(np.median(d["lam_last"]) / tl - 1)
        sd_T = float(d["T"].std())
        # true OLS slope of log lambda over window centres (window-averaged truth)
        lw = np.array([lt[i:i + W].mean() for i in range(L - W + 1)]); c = np.arange(len(lw)) + W / 2
        true_T = float(np.polyfit(c, np.log(lw), 1)[0])
        row = dict(model=o["model"], tc=o["tc"], L=L, power=pw, sd_T=sd_T, mean_T=float(d["T"].mean()),
                   true_T_windowed=true_T, lam_true_first=tf, lam_true_last=tl,
                   bias_first=b_first, bias_last=b_last,
                   lam_last_p05_p95=np.percentile(d["lam_last"], [5, 95]).tolist(),
                   clip_per_record=float(d["clip"].mean()),
                   sd={k: float(d[k].std()) for k in STATS}, mean={k: float(d[k].mean()) for k in STATS},
                   z={k: float((d[k].mean() - res["null"][L][k]["mean"]) / res["null"][L][k]["sd"]) for k in STATS})
        res["cells"].append(row)
        print(f"  {o['model']:8s} {o['tc']:>5} {L} | " + " ".join(f"{pw[k]:6.2f}" for k in STATS)
              + f" | {100*sd_T:5.3f} (true {100*true_T:+.2f}, mean {100*row['mean_T']:+.2f})"
              + f" | {100*b_first:+5.1f} % / {100*b_last:+6.1f} % (true {tl:.3f})"
              + f" | {row['clip_per_record']:.2f}")
    print(f"\nduration {time.time()-t0:.0f} s")
    dr.dump_json(res, "estimateurs.json", indent=1, default=float)


if __name__ == "__main__":
    main()
