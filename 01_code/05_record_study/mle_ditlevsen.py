r"""
REPLICATION HEADER
  PRODUCES   02_output/record_study/mle_ditlevsen.json
  FEEDS      the \rec* macros through record_detectability.py (size and power of the
             Ditlevsen-Ditlevsen test); calib82.py reads its critical values
  INPUTS     none (simulated records)
  SEED       6000 for the simulator control, then 6001, 6002, ... one per cell
  RUNTIME    about 35 min on 6 worker processes
  IMPLEMENTS the model-based maximum-likelihood route (ii) of Ditlevsen and Ditlevsen (2023),
             without sliding windows: its power, its size and the date it returns
             (Appendix app:record)

mle_ditlevsen.py -- route (ii) of Ditlevsen & Ditlevsen (2023): maximum likelihood on the model.

MODEL (D&D 2023, linearised around the stable state): continuous restoring rate alpha(t), noise
sigma, a ramp over the whole record. FOLD ("PLI"): alpha(t)^2 linear (their lambda(t) linear,
alpha = 2 sqrt(A |lambda|)). TRANSCRITICAL ("TRANS"): alpha(t) linear. NULL ("nul"): alpha
constant. Parameters: alpha at the start and at the end of the record (positive, so the ramp is
positive throughout), log sigma; variant "free sigma" ("libre"): log sigma(t) linear (one more
parameter, in all three models). The labels in quotes are the keys of the output.

LIKELIHOOD: quasi-likelihood of the annual means, a time-varying AR(1) with
phi_k = annual AC1(alpha_k) (annual_mean_ou.annual_ac1), marginal variance
V_k = sigma_k^2 (alpha_k - 1 + e^-alpha_k)/alpha_k^3 (variance of the annual means of an OU),
innovation V_k (1 - phi_k^2). This is the locally stationary approximation of D&D, adapted to
annual means; the mean of the state (the drift of x* with lambda, which D&D also use) does not
exist in these linear models and is not used.

TEST: likelihood ratio ramp / null, signed (alpha_end < alpha_start), ORACLE critical value
(95% quantile under the null of constant lambda and constant sigma). SIZE under the null
"rising sigma, constant lambda" (process variance matched to the fold alternative, t_c = 5):
the null the joint region shows the variance cannot tell apart.

DATE: zero of the fitted ramp (a solution at the boundary alpha_end -> 0 gives -0.5 yr: the
threshold at the last year of the record; counted separately); reported from the END of the
record (today) and from its MIDDLE, with the ratio to the truth in both frames.

RECORDS: exact-discretisation simulator (20 steps/yr), same lambda(s) as puissance.simulate
(lambda = lambda_mid at the middle of the 146-year record), any sigma(s). Control: the annual AC1
under the null against the formula.

Usage: python 01_code/05_record_study/mle_ditlevsen.py
"""
import os
import sys
import time

import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "01_calibration"))
import annual_mean_ou as am                                  # noqa: E402
import record_pipeline as dr                                 # noqa: E402
import puissance as pu                                       # noqa: E402

L, BURN, SPY = 146, 50, 20
DT = 1.0 / SPY
LAM_MID = pu.LAM_MID_G
TAU = (np.arange(L) + 0.5 - L / 2) / L                      # centred, in record lengths
TCS = [2, 5, 20, 50, 156]


# ---------------------------------------------------------------- simulation
def lam_path(model, tc):
    s = -BURN + (np.arange((BURN + L) * SPY) + 0.5) * DT
    if model == "nul":
        return np.full_like(s, LAM_MID)
    r = (tc + L - s) / (tc + L / 2)
    return LAM_MID * (np.sqrt(r) if model == "lin-1/2" else r)


def sig2_matched(model, tc):
    """sigma^2(s) giving a constant-lambda process the PROCESS variance path of the alternative."""
    lam = lam_path(model, tc); a = np.exp(-lam * DT)
    v = np.empty(len(lam)); v[0] = 1 / (2 * lam[0])
    for k in range(1, len(lam)):
        v[k] = a[k - 1] ** 2 * v[k - 1] + (1 - a[k - 1] ** 2) / (2 * lam[k - 1])
    return 1 + 2 * (LAM_MID - lam) * v


def simulate(lam, sig2, R, rng):
    a = np.exp(-lam * DT); b = np.sqrt(sig2 * (1 - a * a) / (2 * lam))
    x = rng.standard_normal(R) * np.sqrt(sig2[0] / (2 * lam[0]))
    out = np.empty((R, L)); acc = np.zeros(R)
    for n in range(len(lam)):
        x = a[n] * x + b[n] * rng.standard_normal(R)
        if n >= BURN * SPY:
            acc += x
            if (n + 1 - BURN * SPY) % SPY == 0:
                out[:, (n + 1 - BURN * SPY) // SPY - 1] = acc / SPY; acc[:] = 0
    return out


# ---------------------------------------------------------------- likelihood
def alpha_path(kind, la0, la1):
    a0, a1 = np.exp(la0), np.exp(la1)
    w = TAU + 0.5                                             # 0 at record start, 1 at end
    if kind == "nul":
        return np.full(L, a0)
    if kind == "PLI":
        return np.sqrt(a0 ** 2 + (a1 ** 2 - a0 ** 2) * w)
    return a0 + (a1 - a0) * w


def negll(p, y, kind, free_sigma):
    if kind == "nul":
        al = alpha_path("nul", p[0], p[0]); rest = p[1:]
    else:
        al = alpha_path(kind, p[0], p[1]); rest = p[2:]
    if np.any(al > 30) or np.any(al < 1e-4):
        return 1e10
    ls = rest[0] + (rest[1] * TAU if free_sigma else 0.0)
    g = al - 1 + np.exp(-al)
    V = np.exp(2 * ls) * g / al ** 3
    phi = am.annual_ac1(al)
    s2 = V * (1 - phi ** 2)
    e = y[1:] - phi[1:] * y[:-1]
    return 0.5 * (np.log(V[0]) + y[0] ** 2 / V[0]) + 0.5 * np.sum(np.log(s2[1:]) + e ** 2 / s2[1:])


def fit(y, kind, free_sigma, start):
    best = None
    for x0 in start:
        r = minimize(negll, x0, args=(y, kind, free_sigma), method="Nelder-Mead",
                     options=dict(xatol=1e-6, fatol=1e-8, maxiter=4000, maxfev=4000))
        if best is None or r.fun < best.fun:
            best = r
    return best


def analyse(y):
    y = dr.poly_dt(y, np.arange(L, dtype=float), 1)
    h = L // 2
    l0 = np.log(am.lam_from_ac1(np.clip(dr.ar1(y[:h]), 0.05, 0.98)))
    l1 = np.log(am.lam_from_ac1(np.clip(dr.ar1(y[h:]), 0.05, 0.98)))
    lall = np.log(am.lam_from_ac1(np.clip(dr.ar1(y), 0.05, 0.98)))
    lsd = np.log(y.std() * 1.5)
    out = {}
    for fs in (False, True):
        tail = [lsd] + ([0.0] if fs else [])
        n = fit(y, "nul", fs, [[lall] + tail])
        tag = "libre" if fs else "const"
        out[f"nll_nul_{tag}"] = n.fun
        for kind in ("PLI", "TRANS"):
            r = fit(y, kind, fs, [[l0, l1] + tail, [lall, lall - 0.5] + tail])
            a0, a1 = np.exp(r.x[0]), np.exp(r.x[1])
            lr = 2 * (n.fun - r.fun)
            out[f"stat_{kind}_{tag}"] = max(lr, 0.0) if a1 < a0 else 0.0
            if a1 < a0:
                # zero crossing of the fitted ramp; w = 0 at record start (s = 0.5), 1 at end (s = L - 0.5)
                if kind == "PLI":
                    wz = a0 ** 2 / (a0 ** 2 - a1 ** 2)
                else:
                    wz = a0 / (a0 - a1)
                s_star = 0.5 + wz * (L - 1)
                out[f"date_{kind}_{tag}"] = s_star - L            # from the end
            else:
                out[f"date_{kind}_{tag}"] = np.inf
    return out


def task(args):
    label, model, tc, sigmode, R, seed = args
    rng = np.random.default_rng(seed)
    lam = lam_path(model, tc)
    sig2 = np.ones_like(lam) if sigmode == "const" else sig2_matched(*sigmode)
    Y = simulate(lam, sig2, R, rng)
    rows = [analyse(y) for y in Y]
    return dict(label=label, model=model, tc=tc, rows={k: [r[k] for r in rows] for k in rows[0]})


def main():
    t0 = time.time()
    rng = np.random.default_rng(6000)
    Y = simulate(lam_path("nul", np.inf), np.ones((BURN + L) * SPY), 400, rng)
    ac = np.mean([dr.ar1(y) for y in Y])
    print(f"simulator control: mean annual AC1 under the null {ac:.4f} (formula {am.annual_ac1(LAM_MID):.4f}, "
          f"expected Kendall bias ~ -0.02)")
    jobs, seed = [], 6001
    for k in range(4):
        jobs.append(("nul", "nul", np.inf, "const", 500, seed)); seed += 1
    for k in range(2):
        jobs.append(("nul-sigma-monte", "nul", np.inf, ("lin-1/2", 5), 500, seed)); seed += 1
    for model in ("lin-1/2", "lin-1"):
        for tc in TCS:
            jobs.append((f"{model}", model, tc, "const", 500, seed)); seed += 1
    with Pool(6) as pool:
        out = pool.map(task, jobs, chunksize=1)
    nul = [o for o in out if o["label"] == "nul"]
    keys = [k for k in nul[0]["rows"] if k.startswith("stat_")]
    crit = {k: float(np.quantile(np.concatenate([o["rows"][k] for o in nul]), 0.95)) for k in keys}
    print(f"\nnull (2000 records): 95% critical values: " + ", ".join(f"{k} {v:.2f}" for k, v in crit.items()))
    sm = [o for o in out if o["label"] == "nul-sigma-monte"]
    size = {k: float(np.mean(np.concatenate([o["rows"][k] for o in sm]) > crit[k])) for k in keys}
    print("SIZE under the null 'rising sigma, constant lambda' (variance matched to the fold at t_c = 5), nominal 0.05: "
          + ", ".join(f"{k} {v:.2f}" for k, v in size.items()))
    res = dict(crit=crit, size_sigma_rising=size, cells=[])
    print("\noracle power and dates (median [Q1, Q3]; 'past' = zero before the end; inf = no crossing)")
    for o in out:
        if o["label"] not in ("lin-1/2", "lin-1"):
            continue
        tc = o["tc"]; right = "PLI" if o["model"] == "lin-1/2" else "TRANS"
        row = dict(model=o["model"], tc=tc)
        pw = {k: float(np.mean(np.array(o["rows"][k]) > crit[k])) for k in keys}
        row["power"] = pw
        for tag in ("const", "libre"):
            for kind in ("PLI", "TRANS"):
                d = np.array(o["rows"][f"date_{kind}_{tag}"])
                fin = np.isfinite(d)
                row[f"date_{kind}_{tag}"] = dict(
                    median_end=float(np.median(d)), q1=float(np.percentile(d, 25)), q3=float(np.percentile(d, 75)),
                    ratio_end=float(np.median(d) / tc), ratio_mid=float((np.median(d) + L / 2) / (tc + L / 2)),
                    share_past=float(np.mean(d < 0)), share_nocross=float(1 - fin.mean()),
                    share_boundary=float(np.mean(np.abs(d + 0.5) < 1e-2)),
                    per_record_ratio_end=np.percentile(np.where(fin, d / tc, np.inf), [10, 25, 50, 75, 90]).tolist())
        res["cells"].append(row)
        dc = row[f"date_{right}_const"]; wrong = "TRANS" if right == "PLI" else "PLI"; dw = row[f"date_{wrong}_const"]
        print(f"  truth {o['model']:7s} t_c={tc:>4}: power PLI-const {pw['stat_PLI_const']:.2f} TRANS-const {pw['stat_TRANS_const']:.2f}"
              f" PLI-free {pw['stat_PLI_libre']:.2f} TRANS-free {pw['stat_TRANS_libre']:.2f}"
              f" | date, right class ({right}) {dc['median_end']:.1f} [{dc['q1']:.0f}, {dc['q3']:.0f}] x{dc['ratio_end']:.2f} end, x{dc['ratio_mid']:.2f} mid"
              f" | wrong ({wrong}) {dw['median_end']:.1f} [{dw['q1']:.0f}, {dw['q3']:.0f}] x{dw['ratio_end']:.2f} end, x{dw['ratio_mid']:.2f} mid,"
              f" past {100*dw['share_past']:.0f}% (of which at the boundary alpha_end=0 {100*dw['share_boundary']:.0f}%) no crossing {100*dw['share_nocross']:.0f}%"
              f" | right class at the boundary {100*dc['share_boundary']:.0f}%")
    print(f"duration {time.time()-t0:.0f} s")
    dr.dump_json(res, "mle_ditlevsen.json", indent=1, default=float)


if __name__ == "__main__":
    main()
