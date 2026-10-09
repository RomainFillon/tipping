r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/horizon_np.json
  FEEDS      the \rec* macros through record_detectability.py (the Neyman-Pearson bound);
             horizon_attente.py imports its functions
  INPUTS     none (exact Gaussian covariances and simulated draws)
  SEED       5000, 5001, ... one per cell
  RUNTIME    about 25-30 min on 6 worker processes
  IMPLEMENTS the detectability horizon: the Neyman-Pearson bound on the power of any test, as a
             function of t_c (Appendix app:record)

horizon_np.py -- the detectability horizon: the Neyman-Pearson bound as a function of t_c.

A Gaussian record of annual means of a continuous OU, exact covariance, NP test simulated on
20,000 draws per hypothesis:
  - t_c swept finely, fold (a = 1/2) and transcritical (a = 1);
  - the record length L varies ALONG A FIXED PHYSICAL PATH: lambda equals lambda_mid 73 years
    before the end of the record (the middle of the observed 146-year record), lambda = lambda_mid
    ((t_c + u)/(t_c + 73))^a, u = years before the end. A longer L is a record that starts
    earlier, the end (today) and the threshold staying fixed;
  - the least favourable null is searched in a one-parameter family:
    sigma0^2(s) = 1 + theta * 2 (lambda_mid - lambda(s)) v1(s), theta in {0, .5, .75, 1, 1.25, 1.5}
    (theta = 0: constant sigma; theta = 1: process variance matched). Each theta gives a valid
    bound; the minimum is kept. A partial search: the true bound may be lower.
Horizons: the t_c at which the bound crosses 50% and 80%, interpolated in log t_c.

Control: at L = 146, theta in {0, 1}, reproduces an earlier fixed-length computation of the same
bound (the four reference values in main).

Usage: python 01_code/05_record_study/horizon_np.py
"""
import os
import sys
import time

import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import record_pipeline as dr                                 # noqa: E402
import puissance as pu                                       # noqa: E402

BURN, SPY = 50, 20
DT = 1.0 / SPY
LAM_MID = pu.LAM_MID_G
ANCHOR = 73.0                                                # years before record end where lambda = lambda_mid
NDRAW = 20000
THETAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
TCS = [0.5, 1, 2, 5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 156, 200, 300]
LS = [100, 146, 200, 292]


def lam_path(model, tc, L):
    s = -BURN + (np.arange((BURN + L) * SPY) + 0.5) * DT       # s in [-BURN, L], record = [0, L]
    if not np.isfinite(tc):
        return np.full_like(s, LAM_MID)
    u = L - s
    r = (tc + u) / (tc + ANCHOR)
    return LAM_MID * (np.sqrt(r) if model == "lin-1/2" else r)


def cov_annual(lam, sig2, L):
    a = np.exp(-lam * DT)
    b2 = sig2 * (1 - a * a) / (2 * lam)
    n = len(lam)
    beta = np.empty(n); beta[0] = np.sqrt(sig2[0] / (2 * lam[0])); beta[1:] = np.sqrt(b2[:-1])
    S = np.concatenate([[0.0], np.cumsum(np.log(a[:-1]))])
    MB = np.empty((L, n))
    idx = np.arange(n)
    for k in range(L):                                          # one record year at a time
        rows = np.arange((BURN + k) * SPY, (BURN + k + 1) * SPY)
        E = np.exp(np.minimum(S[rows][:, None] - S[None, :], 0.0))
        E[rows[:, None] < idx[None, :]] = 0.0
        MB[k] = (E * beta[None, :]).mean(0)
    return MB @ MB.T


def variance_path(lam, sig2):
    a = np.exp(-lam * DT); v = np.empty(len(lam)); v[0] = sig2[0] / (2 * lam[0])
    for k in range(1, len(lam)):
        v[k] = a[k - 1] ** 2 * v[k - 1] + sig2[k - 1] * (1 - a[k - 1] ** 2) / (2 * lam[k - 1])
    return v


def np_power(C1, C0, rng, L):
    L1 = np.linalg.cholesky(C1); L0 = np.linalg.cholesky(C0)
    P1, P0 = np.linalg.inv(C1), np.linalg.inv(C0)
    D = P1 - P0
    ld = np.linalg.slogdet(C1)[1] - np.linalg.slogdet(C0)[1]
    lr = lambda Y: -0.5 * np.einsum("ij,jk,ik->i", Y, D, Y) - 0.5 * ld
    crit = np.quantile(lr(rng.standard_normal((NDRAW, L)) @ L0.T), 0.95)
    return float(np.mean(lr(rng.standard_normal((NDRAW, L)) @ L1.T) > crit))


def task(args):
    model, tc, L, seed = args
    rng = np.random.default_rng(seed)
    lam1 = lam_path(model, tc, L); one = np.ones_like(lam1)
    C1 = cov_annual(lam1, one, L)
    v1 = variance_path(lam1, one)
    lam0 = np.full_like(lam1, LAM_MID)
    pw = {}
    for th in THETAS:
        s0 = 1 + th * 2 * (LAM_MID - lam1) * v1
        if s0.min() <= 0:
            continue
        pw[th] = np_power(C1, cov_annual(lam0, s0, L), rng, L)
    th_min = min(pw, key=pw.get)
    return dict(model=model, tc=tc, L=L, by_theta=pw, bound=pw[th_min], theta_min=th_min)


def crossing(tcs, p, level):
    """largest t_c at which the bound is still >= level (bound decreasing in t_c), log interp."""
    tcs, p = np.array(tcs, float), np.array(p)
    for i in range(len(tcs) - 1, 0, -1):
        if p[i - 1] >= level > p[i]:
            x0, x1 = np.log(tcs[i - 1]), np.log(tcs[i])
            return float(np.exp(x0 + (level - p[i - 1]) * (x1 - x0) / (p[i] - p[i - 1])))
    return float(tcs[-1]) if p[-1] >= level else (float("nan") if p[0] < level else float(tcs[0]))


def main():
    t0 = time.time()
    jobs, seed = [], 5000
    for L in LS:
        for model in ("lin-1/2", "lin-1"):
            for tc in TCS:
                jobs.append((model, tc, L, seed)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)
    ctrl = {("lin-1/2", 0.5): (0.873, 0.850), ("lin-1/2", 156): (0.188, 0.139),
            ("lin-1", 20): (0.966, 0.954), ("lin-1", 156): (0.414, 0.281)}
    print("control against the fixed-length computation (L = 146; theta 0 and 1):")
    for (m, tc), (s0, s1) in ctrl.items():
        o = next(o for o in out if o["model"] == m and o["tc"] == tc and o["L"] == 146)
        print(f"  {m:8s} t_c={tc:>5}: theta=0 {o['by_theta'][0.0]:.3f} (was {s0}); theta=1 {o['by_theta'][1.0]:.3f} (was {s1})")
    res = dict(TCS=TCS, LS=LS, THETAS=THETAS, cells=out, horizons=[])
    for L in LS:
        for model in ("lin-1/2", "lin-1"):
            cs = sorted([o for o in out if o["model"] == model and o["L"] == L], key=lambda o: o["tc"])
            p = [o["bound"] for o in cs]
            h50, h80 = crossing(TCS, p, 0.5), crossing(TCS, p, 0.8)
            res["horizons"].append(dict(model=model, L=L, h50=h50, h80=h80))
            print(f"\nL = {L} {model}: NP bound (min over theta): "
                  + " ".join(f"{o['tc']}:{o['bound']:.2f}(th{o['theta_min']})" for o in cs))
            print(f"   50% horizon: t_c = {h50:.1f} yr; 80% horizon: t_c = {h80:.1f} yr")
    print(f"duration {time.time()-t0:.0f} s")
    dr.dump_json(res, "horizon_np.json", indent=1, default=float)


if __name__ == "__main__":
    main()
