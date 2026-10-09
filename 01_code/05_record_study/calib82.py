r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/calib82.json
  FEEDS      the \rec* macros through record_detectability.py (the size of the
             Ditlevsen-Ditlevsen test over the noise trends the record cannot exclude)
  INPUTS     02_output/record_study/identification.json and draws/B_1871-2016_lineaire_10.npy,
             draws/A_1871-2016_lineaire_40.npy (identification.py);
             02_output/record_study/mle_ditlevsen.json (mle_ditlevsen.py)
  SEED       8000, 8001, ... one per cell
  RUNTIME    about 25 min on 6 worker processes
  IMPLEMENTS the false-rejection rate of the Ditlevsen-Ditlevsen test, calibrated on the record
             (Appendix app:record)

calib82.py -- the false-rejection rate of the Ditlevsen-Ditlevsen test, calibrated on the record.

mle_ditlevsen.py measures that the constant-sigma Ditlevsen test rejects in about 82% of cases
under a CHOSEN null "constant lambda, rising sigma" (process variance matched to the fold at
t_c = 5). Here the null is taken from what the record does not exclude:

1. The joint region of identification.py (robust region, identification.robust_region), at the
   two leading specifications: B 1871-2016 linear, blocks of 10, and A 1871-2016 linear,
   w = 40; the saved draws are mapped to CORRECTED coordinates by the Jacobian J of the
   annual-mean reading (J_at below, lambda = 0.7928).
2. Slice of the region along d log lambda/dt = 0 (the null "constant lambda"): the interval
   [s_lo, s_hi] of slopes d log sigma^2/dt the record does not exclude under that null.
   Constrained point estimate: s* = c_s - S_sl / S_ll * c_l (centre of the region conditioned on
   d log lambda/dt = 0). Unconstrained point: c_s.
3. Effective slope of the 82% null: OLS slope of log sigma0^2(t) over the years of the record.
4. Simulation (mle_ditlevsen.simulate, UNCHANGED): lambda = lambda_mid constant,
   sigma^2(t) = exp(s (t - t_mid)), for each slope s; 800 records; statistics
   mle_ditlevsen.analyse UNCHANGED; rejection against the oracle critical values of
   mle_ditlevsen.json (null of constant lambda and constant sigma). The rejection rate is a
   FALSE-rejection rate: lambda does not move.

Usage: python 01_code/05_record_study/calib82.py
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
import mle_ditlevsen as md                                   # noqa: E402
import identification as idf                                 # noqa: E402

SPECS = [("B_1871-2016_lineaire_10", "draws"), ("A_1871-2016_lineaire_40", "theta")]
N_REC = 800

# ---------------------------------------------------------------- the Jacobian of the
# annual-mean reading: (d log lambda, d log sigma^2) read as point samples -> corrected
H = 1e-5


def e_of(l):
    f = lambda x: np.log(-np.log(am.annual_ac1(x)))
    return float((f(l * (1 + H)) - f(l * (1 - H))) / (np.log(1 + H) - np.log(1 - H)))


def k_of(l):
    f = lambda x: np.log(2 * x * am.var_ratio(x))
    return float((f(l * (1 + H)) - f(l * (1 - H))) / (np.log(1 + H) - np.log(1 - H)))


def J_at(lam_c):
    e, k = e_of(lam_c), k_of(lam_c)
    return np.array([[1 / e, 0.0], [(k - e) / e, 1.0]]), e, k


def region_c(key, centre_mode):
    J = J_at(0.7928)[0]
    old = dr.load_json("identification.json")
    est = key[0]
    R0 = next(r for r in old[est] if (f"{est}_{r['period']}_{r['detrend']}_{r['blk'] if est == 'B' else r['w']}") == key)
    D = np.load(os.path.join(dr.OUTDIR, "draws", key + ".npy"))
    th = J @ np.array(R0["theta"]); Dc = D @ J.T
    R = idf.robust_region(th, Dc, centre_mode)
    c, S, q = np.array(R["centre"]), np.array(R["S_rob"]), R["q_rob"]
    Si = np.linalg.inv(S)
    # slice at x = 0: p = (0, s) inside iff (p - c)' Si (p - c) <= q. Along that line the form is
    # minimised at s_star = c_s - S_sl/S_ll c_l (the conditional mean), so the axis cuts the
    # region iff f(s_star) <= 0, and the two edges bracket s_star.
    f = lambda s: (np.array([0 - c[0], s - c[1]]) @ Si @ np.array([0 - c[0], s - c[1]])) - q
    s_star = c[1] - S[0, 1] / S[0, 0] * c[0]
    if f(s_star) > 0:
        return dict(key=key, centre=c.tolist(), origin_axis_cut=False, s_star=float(s_star))
    from scipy.optimize import brentq
    lo = brentq(f, s_star - 1.0, s_star); hi = brentq(f, s_star, s_star + 1.0)
    return dict(key=key, centre=c.tolist(), S=S.tolist(), q=q, s_lo=float(lo), s_hi=float(hi),
                s_star=float(s_star), s_uncond=float(c[1]), axis_cut=True)


def task(args):
    label, slope, sig_path, R, seed = args
    rng = np.random.default_rng(seed)
    lam = md.lam_path("nul", np.inf)
    if sig_path is None:
        t = -md.BURN + (np.arange((md.BURN + md.L) * md.SPY) + 0.5) * md.DT
        sig2 = np.exp(slope * (t - md.L / 2))
    else:
        sig2 = sig_path
    Y = md.simulate(lam, sig2, R, rng)
    rows = [md.analyse(y) for y in Y]
    return dict(label=label, slope=slope, stats={k: [r[k] for r in rows] for k in rows[0] if k.startswith("stat_")})


def main():
    t0 = time.time()
    crit = dr.load_json("mle_ditlevsen.json")["crit"]
    regs = [region_c(k, m) for k, m in SPECS]
    for r in regs:
        print(f"{r['key']} (corrected coordinates): centre (d log lambda, d log sigma^2) = "
              f"({100*r['centre'][0]:+.3f}, {100*r['centre'][1]:+.3f}) %/yr; slice at d log lambda = 0: "
              f"[{100*r['s_lo']:+.3f}, {100*r['s_hi']:+.3f}] %/yr; constrained point {100*r['s_star']:+.3f}, "
              f"unconstrained {100*r['s_uncond']:+.3f}")
    # effective slope of the 82 % null
    s0 = md.sig2_matched("lin-1/2", 5)
    t = -md.BURN + (np.arange(len(s0)) + 0.5) * md.DT
    rec = t >= 0
    ann_t = t[rec].reshape(md.L, md.SPY).mean(1); ann_s = np.log(s0[rec].reshape(md.L, md.SPY).mean(1))
    s82 = float(np.polyfit(ann_t, ann_s, 1)[0])
    ratio_end = float(s0[rec][-1] / s0[rec][0])
    print(f"the 82% null: effective slope of log sigma^2 over the record {100*s82:+.3f} %/yr "
          f"(sigma^2 end/start = {ratio_end:.2f}); not exponential, concentrated at the end of the record")
    jobs, seed = [], 8000
    pts = []
    for r in regs:
        for lab in ("s_lo", "s_star", "s_uncond", "s_hi"):
            pts.append((f"{r['key'][:1]}:{lab}", r[lab]))
    pts.append(("zero", 0.0))
    for lab, s in pts:
        jobs.append((lab, s, None, N_REC, seed)); seed += 1
    jobs.append(("nul82-exponentiel", s82, None, N_REC, seed)); seed += 1
    jobs.append(("nul82-forme-exacte", s82, s0, N_REC, seed)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)
    res = dict(regions=regs, slope_82=s82, crit=crit, cells=[])
    print("\nFALSE-rejection rate (constant lambda), oracle critical values at 5%:")
    for o in out:
        rr = {k: float(np.mean(np.array(v) > crit[k])) for k, v in o["stats"].items()}
        se = np.sqrt(0.05 * 0.95 / N_REC)
        res["cells"].append(dict(label=o["label"], slope=o["slope"], reject=rr, n=N_REC))
        print(f"  {o['label']:20s} sigma^2 slope {100*o['slope']:+.3f} %/yr: PLI-const {rr['stat_PLI_const']:.3f}"
              f"  TRANS-const {rr['stat_TRANS_const']:.3f}  PLI-free {rr['stat_PLI_libre']:.3f}  TRANS-free {rr['stat_TRANS_libre']:.3f}"
              f"   (SE at 5%: {se:.3f})")
    print(f"duration {time.time()-t0:.0f} s")
    dr.dump_json(res, "calib82.json", indent=1, default=float)


if __name__ == "__main__":
    main()
