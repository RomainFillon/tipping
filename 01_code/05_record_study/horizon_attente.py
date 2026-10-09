r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/horizon_attente.json ("attente": waiting)
  FEEDS      the \rec* macros through record_detectability.py (the horizon on the waiting
             diagonal, and the bound read at the AMOC's distance)
  INPUTS     none (exact Gaussian covariances and simulated draws)
  SEED       7000, 7001, ... one per cell
  RUNTIME    about 15 min on 6 worker processes
  IMPLEMENTS the detectability horizon WHEN ONE WAITS: the record lengthens while the threshold
             nears (Appendix app:record)

horizon_attente.py -- the detectability horizon when one waits.

horizon_np.py gives the NP bound at a fixed record length. But waiting for the threshold to
approach also accumulates record: if the threshold is t_c = 156 years away today with 146 years
of record, in Delta years it will be 156 - Delta away with 146 + Delta years of record. This script
follows that diagonal, along a physical path fixed in calendar time (lambda = lambda_mid in 1943,
73 years before the end of the current record, so 73 + Delta years before the end of the future
record).

Bound: NP against the family of nulls sigma0^2 = 1 + theta 2 (lambda_mid - lambda) v1, minimum
over a finer grid of theta (0 to 1 in steps of 0.1, plus 1.25); and, at L = 146, an even finer
search in theta at the key cells, to tell whether the minimum is reached.

Usage: python 01_code/05_record_study/horizon_attente.py
"""
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import record_pipeline as dr                                 # noqa: E402
import horizon_np as hn                                      # noqa: E402

TC_NOW, L_NOW = 156.0, 146
THETAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25]
DELTAS = [0, 20, 40, 60, 80, 100, 110, 120, 130, 140, 146, 150, 153]


def task(args):
    model, tc, L, anchor, thetas, seed = args
    hn.ANCHOR = anchor
    rng = np.random.default_rng(seed)
    lam1 = hn.lam_path(model, tc, L); one = np.ones_like(lam1)
    C1 = hn.cov_annual(lam1, one, L)
    v1 = hn.variance_path(lam1, one)
    lam0 = np.full_like(lam1, hn.LAM_MID)
    pw = {}
    for th in thetas:
        s0 = 1 + th * 2 * (hn.LAM_MID - lam1) * v1
        if s0.min() > 0:
            pw[th] = hn.np_power(C1, hn.cov_annual(lam0, s0, L), rng, L)
    th_min = min(pw, key=pw.get)
    return dict(model=model, tc=tc, L=L, anchor=anchor, by_theta=pw, bound=pw[th_min], theta_min=th_min)


def main():
    t0 = time.time()
    jobs, seed = [], 7000
    for model in ("lin-1/2", "lin-1"):
        for d in DELTAS:
            jobs.append((model, TC_NOW - d, L_NOW + d, 73.0 + d, THETAS, seed)); seed += 1
    fine = [round(x, 2) for x in np.arange(0.2, 1.01, 0.05)]
    for model in ("lin-1/2", "lin-1"):
        for tc in (0.5, 10, 156):
            jobs.append((model, tc, L_NOW, 73.0, fine, seed)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)
    res = dict(diagonal=[], fine=[])
    for model in ("lin-1/2", "lin-1"):
        cs = sorted([o for o in out if o["model"] == model and len(o["by_theta"]) <= len(THETAS)
                     and abs(o["tc"] + o["L"] - (TC_NOW + L_NOW)) < 1e-9], key=lambda o: -o["tc"])
        print(f"\n{model}: waiting (t_c = 156 - Delta, L = 146 + Delta)")
        for o in cs:
            print(f"   Delta = {156 - o['tc']:5.0f}: t_c = {o['tc']:5.0f}, L = {o['L']:3d} -> bound {o['bound']:.3f} "
                  f"(theta* {o['theta_min']}); theta 0: {o['by_theta'][0.0]:.3f}, theta 1: {o['by_theta'][1.0]:.3f}")
        tcs = np.array([o["tc"] for o in cs]); p = np.array([o["bound"] for o in cs])
        hz = {}
        for lev in (0.5, 0.8):
            h = float("nan")
            for i in range(len(p) - 1):
                if p[i] < lev <= p[i + 1]:
                    h = float(tcs[i] + (lev - p[i]) * (tcs[i + 1] - tcs[i]) / (p[i + 1] - p[i]))
                    break
            if p[0] >= lev:
                h = float(tcs[0])
            hz[lev] = h
            print(f"   {int(lev*100)}% horizon: t_c = {h:.1f} yr (that is, in {TC_NOW - h:.0f} years, with a record of {L_NOW + TC_NOW - h:.0f} years)")
        res["diagonal"].append(dict(model=model, cells=cs, h50=hz[0.5], h80=hz[0.8]))
    print("\nfine search in theta at L = 146:")
    for o in [o for o in out if len(o["by_theta"]) > len(THETAS)]:
        th = sorted(o["by_theta"]); pv = [o["by_theta"][t] for t in th]
        print(f"   {o['model']:8s} t_c={o['tc']:>5}: min {o['bound']:.3f} at theta = {o['theta_min']}; "
              + " ".join(f"{t}:{v:.3f}" for t, v in zip(th, pv)))
        res["fine"].append(o)
    print(f"duration {time.time()-t0:.0f} s")
    dr.dump_json(res, "horizon_attente.json", indent=1, default=float)


if __name__ == "__main__":
    main()
