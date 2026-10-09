r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/fenetres.json ("fenetres": windows)
  FEEDS      the \rec* macros through record_detectability.py (the ceiling of the level
             statistic as a function of the window width)
  INPUTS     none (simulated records)
  SEED       2000, 2001, 2002 for the null, then 2003, 2004, ... one per cell
  RUNTIME    under 1 min on 6 worker processes
  IMPLEMENTS the window-width family of the level statistic N(w1, w2) and the mechanism behind
             its ceiling (Appendix app:record)

fenetres.py -- the ceiling of N as a function of the window width.

Records: puissance.simulate UNCHANGED (lin-1/2 = fold, lin-1 = transcritical; 146 annual means;
AC1 = 0.61 at mid-record). Global linear detrending, AC1 by record_pipeline.ar1, lambda by the
corrected reading (annual_mean_ou, tabulated), AC1 clipped to [0.02, 0.999] and counted.

Family N(w1, w2) = log lambda_hat(last w2 years) - log lambda_hat(first w1 years), w1, w2 in W.
Symmetric (w1 = w2) and asymmetric (full grid).
ORACLE test: critical value = 5% quantile under the simulated null (6000 records), one-sided.
The best cell of the grid is chosen on half A of the records (1000) and its power is reported on
half B (1000): no winner's curse.

Mechanism: median of lambda_hat over the last window of w2 years as a function of tau = t_c,
against the true mean of lambda = c (T - t)^a over the window,
c [(tau+w)^(a+1) - tau^(a+1)] / ((a+1) w)  ->  c w^a / (a+1)  as tau -> 0,
with c = lambda_mid / (t_c + L/2)^a (lambda_mid is fixed at mid-record).

Usage: python 01_code/05_record_study/fenetres.py
"""
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import estimateurs as es                                     # noqa: E402  (tabulation, simulator)

pu, dr = es.pu, es.dr
L = 146
W = [4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 73]
TCS = [0.5, 1, 2, 5, 10, 20, 40, 156]


def per_record(y):
    r = dr.poly_dt(y, np.arange(L, dtype=float), 1)
    af = np.array([dr.ar1(r[:w]) for w in W]); al = np.array([dr.ar1(r[-w:]) for w in W])
    clip = int(np.sum((af < es.AC_LO) | (af > es.AC_HI)) + np.sum((al < es.AC_LO) | (al > es.AC_HI)))
    lf = es.lam_tab(np.clip(af, es.AC_LO, es.AC_HI)); ll = es.lam_tab(np.clip(al, es.AC_LO, es.AC_HI))
    return np.log(lf), np.log(ll), clip


def task(args):
    model, tc, R, seed = args
    rng = np.random.default_rng(seed)
    recs, _ = pu.simulate(model, tc, pu.LAM_MID_G, R, rng)
    out = [per_record(y) for y in recs]
    return dict(model=model, tc=tc, lf=np.array([o[0] for o in out]), ll=np.array([o[1] for o in out]),
                clip=float(np.mean([o[2] for o in out])))


def window_mean(a, tc, w):
    c = pu.LAM_MID_G / (tc + L / 2) ** a
    return c * ((tc + w) ** (a + 1) - tc ** (a + 1)) / ((a + 1) * w), c * w ** a / (a + 1)


def main():
    t0 = time.time()
    jobs, seed = [("lin-1", np.inf, 2000, s) for s in (2000, 2001, 2002)], 2003
    for m in ("lin-1/2", "lin-1"):
        for tc in TCS:
            jobs.append((m, tc, 2000, seed)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)
    nul = [o for o in out if not np.isfinite(o["tc"])]
    LF0 = np.concatenate([o["lf"] for o in nul]); LL0 = np.concatenate([o["ll"] for o in nul])
    # N[i, j] : first window W[i], last window W[j]
    N0 = LL0[:, None, :] - LF0[:, :, None]
    crit = np.quantile(N0, 0.05, axis=0)
    print(f"null: {len(LF0)} records; SD of N(w,w) under the null: "
          + ", ".join(f"w={w}: {N0[:, i, i].std():.2f}" for i, w in enumerate(W)))
    res = dict(W=W, TCS=TCS, crit=crit.tolist(), cells=[])
    for o in sorted([o for o in out if np.isfinite(o["tc"])], key=lambda o: (o["model"], o["tc"])):
        a = 0.5 if o["model"] == "lin-1/2" else 1.0
        N = o["ll"][:, None, :] - o["lf"][:, :, None]
        half = len(N) // 2
        PA = np.mean(N[:half] < crit, axis=0); PB = np.mean(N[half:] < crit, axis=0)
        i, j = np.unravel_index(np.argmax(PA), PA.shape)
        diag = np.array([np.mean(N[:, k, k] < crit[k, k]) for k in range(len(W))])
        kd = int(np.argmax(diag[:]))
        mech = []
        for k, w in enumerate(W):
            wm, lim = window_mean(a, o["tc"], w)
            lam_hat = np.exp(o["ll"][:, k])
            mech.append(dict(w=w, median=float(np.median(lam_hat)), mean=float(lam_hat.mean()),
                             window_mean=float(wm), limit=float(lim)))
        row = dict(model=o["model"], tc=o["tc"], a=a, clip=o["clip"],
                   diag=diag.tolist(), best_diag=dict(w=W[kd], power=float(diag[kd])),
                   best_grid=dict(w1=W[i], w2=W[j], power_select=float(PA[i, j]), power_validate=float(PB[i, j])),
                   grid_B=PB.tolist(), mech=mech)
        res["cells"].append(row)
        print(f"{o['model']:8s} t_c={o['tc']:>5}: N(w,w) " + " ".join(f"{W[k]}:{diag[k]:.2f}" for k in range(len(W)))
              + f" | best sym. w={W[kd]} {diag[kd]:.2f} | best of grid (w1={W[i]}, w2={W[j]}) selected {PA[i,j]:.2f}"
              f" validated {PB[i,j]:.2f} | clipped/rec {o['clip']:.2f}")
    print("\nmechanism: median of lambda_hat (last window) against the true mean over the window")
    for row in res["cells"]:
        s = " ".join(f"w={m['w']}: {m['median']:.3f}/{m['window_mean']:.3f}" for m in row["mech"] if m["w"] in (10, 20, 40))
        print(f"  {row['model']:8s} t_c={row['tc']:>5}: {s}   (limit c w^a/(a+1) for w=40: {row['mech'][W.index(40)]['limit']:.3f})")
    print(f"duration {time.time()-t0:.0f} s")
    dr.dump_json(res, "fenetres.json", indent=1, default=float)


if __name__ == "__main__":
    main()
