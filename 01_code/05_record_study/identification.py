r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/identification.json and 02_output/record_study/draws/*.npy
             (120 arrays of bootstrap draws; calib82.py reads two of them)
  FEEDS      calib82.json, through calib82.py (the joint region the record cannot exclude)
  INPUTS     00_data/amoc/sg_index_hadisst.txt, through record_pipeline.load()
  SEED       record_pipeline.rng = default_rng(42), set at import; estimator B runs first, in
             the same order as the single-estimator run, so its draws are the same
  RUNTIME    about 40 min (one core)
  IMPLEMENTS the joint 95% confidence region of (d log lambda/dt, d log sigma^2/dt) on the
             record (Appendix app:record)

identification.py -- what the record cannot separate.

Reuses the estimators and block bootstraps of record_pipeline.py unchanged, and builds the
joint 95% confidence region of the pair (d log lambda/dt, d log sigma^2/dt), with
sigma^2 = 2 lambda Var the continuous-time noise of the paper.

For each specification:
  - bootstrap draws of the pair (the same bootstrap as the single-estimator run);
  - elliptical region {theta : (theta - theta_hat)' S^-1 (theta - theta_hat) <= q}, S the
    bootstrap covariance, q = chi2_2(0.95) = 5.99 and, as a control, q calibrated on the draws
    themselves (95% quantile of their Mahalanobis distances);
  - the ray {d log sigma^2/dt = 0} cuts the region iff |theta_s| / sqrt(S_ss) <= sqrt(q) (and
    likewise for lambda): a marginal test at the Scheffe level, wider than the marginal 95%
    interval, which is reported too (percentiles);
  - the origin (neither approach nor rising noise): inside the region or not.
Retained region (robust_region): shape from a trimmed covariance, level from the empirical 95%
quantile of the draws. The raw covariance is dominated by a few replicates of B that overflow
(correlation ~ 1), which makes the ellipse artificially wide; both are written out.

Usage: python 01_code/05_record_study/identification.py
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record_pipeline as dr                                 # noqa: E402

Q_CHI2 = stats.chi2.ppf(0.95, 2)
PERIODS = {"1871-2016": (1871, 2016), "1880-2016": (1880, 2016), "1900-2016": (1900, 2016),
           "1920-2016": (1920, 2016), "1871-1990": (1871, 1990)}


def dlogsig2_B(p):
    """d log(2 lambda Var)/dt at tau = 0 for the TV-AR(1) parameters p."""
    return p[1] + dr.dlogvar_from(p, 0.0)


def region(theta, draws):
    """theta: point (2,), draws: (B,2) centred bootstrap replicates of theta."""
    d = draws[np.all(np.isfinite(draws), 1)]
    S = np.cov(d.T)
    Si = np.linalg.inv(S)
    m = d - d.mean(0)
    maha = np.einsum("ij,jk,ik->i", m, Si, m)
    q_emp = np.quantile(maha, 0.95)
    z = theta / np.sqrt(np.diag(S))
    origin = float(theta @ Si @ theta)
    # basic (reflected) percentile marginal CIs
    lo = 2 * theta - np.quantile(d, 0.975, axis=0)
    hi = 2 * theta - np.quantile(d, 0.025, axis=0)
    return dict(theta=theta.tolist(), S=S.tolist(), q_emp=float(q_emp), z=z.tolist(),
                ray_sig2_in=bool(abs(z[1]) <= np.sqrt(Q_CHI2)), ray_lam_in=bool(abs(z[0]) <= np.sqrt(Q_CHI2)),
                ray_sig2_in_emp=bool(abs(z[1]) <= np.sqrt(q_emp)), ray_lam_in_emp=bool(abs(z[0]) <= np.sqrt(q_emp)),
                origin_in=bool(origin <= Q_CHI2), origin_maha=origin,
                ci_lam=[float(lo[0]), float(hi[0])], ci_sig2=[float(lo[1]), float(hi[1])],
                corr=float(S[0][1] / np.sqrt(S[0][0] * S[1][1])), nboot=int(len(d)))


def robust_region(theta, draws, centre="draws"):
    """Non-parametric region: shape = trimmed covariance (iterated, 97.5% of the draws under
    chi2_2), level = empirical 95% quantile of the draws' distances (the scale of the shape
    cancels). centre="draws": percentile region (the cloud itself, as the interval of
    estimator B); centre="theta": draws already recentred on theta (basic, for A)."""
    d = draws[np.all(np.isfinite(draws), 1)]
    c = np.median(d, 0)
    S = np.cov(d.T)
    for _ in range(5):
        Si = np.linalg.inv(S); m = d - c
        dist = np.einsum("ij,jk,ik->i", m, Si, m)
        keep = dist <= stats.chi2.ppf(0.975, 2)
        S = np.cov(d[keep].T); c = np.median(d[keep], 0)
    Si = np.linalg.inv(S); m = d - c
    dist = np.einsum("ij,jk,ik->i", m, Si, m)
    q = np.quantile(dist, 0.95)
    zc = c / np.sqrt(np.diag(S))                    # distance of each axis-line to the centre
    origin = float(c @ Si @ c)
    lo = np.quantile(d, 0.025, axis=0); hi = np.quantile(d, 0.975, axis=0)
    return dict(centre=c.tolist(), S_rob=S.tolist(), q_rob=float(q),
                ray_sig2_in_rob=bool(zc[1] ** 2 <= q), ray_lam_in_rob=bool(zc[0] ** 2 <= q),
                origin_in_rob=bool(origin <= q), zc=zc.tolist(), sqrt_q=float(np.sqrt(q)),
                pct_lam=[float(lo[0]), float(hi[0])], pct_sig2=[float(lo[1]), float(hi[1])],
                corr_rob=float(S[0][1] / np.sqrt(S[0][0] * S[1][1])))


def main():
    yr, sst, src = dr.load()
    print(f"Series: {len(sst)} values ({src}); a single SST fingerprint in the file.")
    print(f"q = chi2_2(0.95) = {Q_CHI2:.3f}, Scheffe threshold |z| <= {np.sqrt(Q_CHI2):.3f}\n")
    out = {"B": [], "A": []}
    draws = dr.out_path("draws")
    os.makedirs(draws, exist_ok=True)

    print("=== (B) time-varying AR(1); moving-block bootstrap of pairs (1000) ===")
    for pn, (a, b) in PERIODS.items():
        m = (yr >= a) & (yr <= b)
        for dn in ["lineaire", "quadratique", "noyau50"]:
            r = dr.DETRENDS[dn](sst[m], yr[m]); t = yr[m]
            x0, x1 = r[:-1], r[1:]; tau = t[1:] - t[1:].mean()
            for blk in [10, 20]:
                p, P = dr.pairs_block_boot(x0, x1, tau, blk, 1000)
                th = np.array([p[1], dlogsig2_B(p)])
                D = np.array([[q[1], dlogsig2_B(q)] for q in P])
                R = region(th, D); R.update(period=pn, detrend=dn, blk=blk)
                R.update(robust_region(th, D, "draws"))
                np.save(os.path.join(draws, f"B_{pn}_{dn}_{blk}.npy"), D)
                out["B"].append(R)
                print(f"   robust: ray σ²=0 inside: {R['ray_sig2_in_rob']} | ray λ=0 inside: {R['ray_lam_in_rob']} | "
                      f"origin inside: {R['origin_in_rob']} | centre/sqrt(S)=({R['zc'][0]:+.2f},{R['zc'][1]:+.2f}) sqrt(q)={R['sqrt_q']:.2f} "
                      f"| pct λ [{100*R['pct_lam'][0]:+.2f},{100*R['pct_lam'][1]:+.2f}]")
                print(f"{pn} {dn:11s} blk={blk}: dlogλ={100*th[0]:+.2f}  dlogσ²={100*th[1]:+.2f} %/yr  "
                      f"z=({R['z'][0]:+.2f},{R['z'][1]:+.2f}) corr={R['corr']:+.2f} | ray σ²=0 inside: {R['ray_sig2_in']}"
                      f" | ray λ=0 inside: {R['ray_lam_in']} | origin inside: {R['origin_in']} "
                      f"| marg. 95% CI σ² [{100*R['ci_sig2'][0]:+.2f},{100*R['ci_sig2'][1]:+.2f}]")

    print("\n=== (A) sliding windows; block bootstrap of the residuals (300), same pipeline ===")
    for pn, (a, b) in PERIODS.items():
        m = (yr >= a) & (yr <= b); y, t = sst[m], yr[m]
        base = dr.poly_dt(y, t, 1)
        x0, x1 = base[:-1], base[1:]; tau = t[1:] - t[1:].mean()
        p = dr.tvar_fit(x0, x1, tau)
        e_std = (x1 - np.exp(-np.exp(p[0] + p[1] * tau)) * x0) / np.exp(p[2] + p[3] * tau)
        trend = y - base
        for dn in dr.DETRENDS:
            for w in [30, 40, 50, 60, 70]:
                if w > len(y) / 2:
                    continue
                sl, sv, _ = dr.windowed(y, t, dn, w)
                bs = []
                for _ in range(300):
                    xs = dr.resid_block_series(p, base[0], tau, e_std, 10)
                    bl, bv, _ = dr.windowed(xs + trend, t, dn, w)
                    bs.append([bl, bl + bv])
                bs = np.array(bs)
                th = np.array([sl, sl + sv])
                D = bs - np.median(bs, 0) + th            # centred as for estimator A (basic bootstrap)
                R = region(th, D); R.update(period=pn, detrend=dn, w=w)
                R.update(robust_region(th, D, "theta"))
                np.save(os.path.join(draws, f"A_{pn}_{dn}_{w}.npy"), D)
                out["A"].append(R)
                print(f"{pn} {dn:12s} w={w}: dlogλ={100*th[0]:+.2f}  dlogσ²={100*th[1]:+.2f} %/yr  "
                      f"z=({R['z'][0]:+.2f},{R['z'][1]:+.2f}) corr={R['corr']:+.2f} | σ²=0 inside: {R['ray_sig2_in']}"
                      f" | λ=0 inside: {R['ray_lam_in']} | origin inside: {R['origin_in']}"
                      f" || robust: σ²=0 {R['ray_sig2_in_rob']} λ=0 {R['ray_lam_in_rob']} origin {R['origin_in_rob']}")
    dr.dump_json(out, "identification.json", indent=1)


if __name__ == "__main__":
    main()
