r"""
REPLICATION HEADER
  PRODUCES   nothing on its own; a module imported by the record-study scripts
  FEEDS      identification.json directly (through identification.py), and every other study
             output through the estimators it defines
  INPUTS     00_data/amoc/sg_index_hadisst.txt (read by load(), md5-checked), for
             identification.py only; the other scripts use the estimators on simulated records
  SEED       the module-level generator rng = default_rng(42); callers that simulate reseed it
  RUNTIME    negligible
  IMPLEMENTS the two trend estimators of the recovery rate on the AMOC record (Appendix
             app:record), and the shared paths of the study

record_pipeline.py -- the two estimators of a trend in the recovery rate, as run on the record.

The record is the AMOC SST fingerprint read by 01_calibration/ews_calibration.py (subpolar-gyre
index derived from HadISST, Caesar et al. 2018, 1871-2016). Two estimators of the slope
d log lambda / dt, where lambda is the AR(1) recovery rate:

  (A) sliding windows (the estimator of ews_calibration.py): lambda_w = -ln AC1_w, OLS slope of
      log lambda_w on the window centre; confidence interval by block bootstrap of the
      residuals of a fitted time-varying AR(1), pushed back through the same pipeline
      (detrending, then windows).
  (B) time-varying AR(1) by Gaussian maximum likelihood:
      x_t = exp(-lambda(t)) x_{t-1} + s(t) e_t,  log lambda(t) = a0 + a1 tau,
      log s(t) = b0 + b1 tau,  so a1 = d log lambda / dt directly;
      confidence interval by moving-block bootstrap of the pairs (t, x_{t-1}, x_t), each pair
      keeping its time.

The detrending keys of DETRENDS ("lineaire", "quadratique", "noyau50", "noyau30",
"dans-fenetre") are the labels the study outputs carry, and are kept as they are.
"""
import hashlib
import json
import os
import sys

import numpy as np
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
PKG = os.path.dirname(CODE)
SERIES = os.path.join(PKG, "00_data", "amoc", "sg_index_hadisst.txt")
OUTDIR = os.path.join(PKG, "02_output", "record_study")
sys.path.insert(0, CODE)
from inputs_literature import DERIVATION_INPUTS  # noqa: E402

rng = np.random.default_rng(42)


def out_path(name):
    """Where a study output is written and read: 02_output/record_study/<name>."""
    os.makedirs(OUTDIR, exist_ok=True)
    return os.path.join(OUTDIR, name)


def dump_json(obj, name, **kw):
    """Write a study output with LF line endings on every platform."""
    with open(out_path(name), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, **kw)


def load_json(name):
    p = os.path.join(OUTDIR, name)
    if not os.path.exists(p):
        raise SystemExit(f"[ERROR] {p} is missing: it is written by an earlier script of the "
                         "record study. Run the study in the order run_all.sh gives.")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load():
    """The AMOC record, read from the local copy that fetch_pik_sg_index.py downloads."""
    if not os.path.exists(SERIES):
        raise SystemExit(f"[ERROR] record_pipeline.py: the AMOC series is missing at {SERIES}.\n"
                         "        It is not redistributed (no licence published by PIK). Download it once with\n"
                         "            python 00_data/amoc/fetch_pik_sg_index.py\n"
                         "        before run_all.sh; see 00_data/amoc/PROVENANCE.md.")
    with open(SERIES, "rb") as fh:
        data = fh.read()
    md5 = hashlib.md5(data).hexdigest()
    expected = DERIVATION_INPUTS["amoc_series_md5"][0]
    if md5 != expected:
        raise SystemExit(f"[ERROR] record_pipeline.py: {SERIES} has md5 {md5}, not the {expected}\n"
                         "        of the series the paper was computed on (00_data/amoc/PROVENANCE.md).")
    raw = data.decode()
    v = np.array([float(l) for l in raw.splitlines() if l.strip() and not l.startswith("#")])
    return np.arange(1871, 1871 + len(v)).astype(float), v, "00_data/amoc (md5 checked)"


# ---------------------------------------------------------------- detrendings
def poly_dt(y, t, k):
    tc = (t - t.mean()) / t.std()
    return y - np.polyval(np.polyfit(tc, y, k), tc)


def kern_dt(y, t, fwhm):
    bw = fwhm / 2.355
    w = np.exp(-0.5 * ((t[:, None] - t[None, :]) / bw) ** 2)
    return y - (w / w.sum(1, keepdims=True)) @ y


DETRENDS = {
    "lineaire": lambda y, t: poly_dt(y, t, 1),
    "quadratique": lambda y, t: poly_dt(y, t, 2),
    "noyau50": lambda y, t: kern_dt(y, t, 50.0),
    "noyau30": lambda y, t: kern_dt(y, t, 30.0),
    "dans-fenetre": None,          # linear within each window, raw series otherwise
}


def ar1(z):
    z = z - z.mean()
    return np.dot(z[:-1], z[1:]) / np.dot(z[:-1], z[:-1])


# ---------------------------------------------------------------- estimator A
def windowed(y, t, detrend, w):
    r = y - y.mean() if detrend == "dans-fenetre" else DETRENDS[detrend](y, t)
    L, V, T = [], [], []
    for i in range(len(r) - w + 1):
        s = r[i:i + w]
        if detrend == "dans-fenetre":
            s = poly_dt(s, t[i:i + w], 1)
        ph = ar1(s)
        if ph <= 0.02:                       # lambda undefined: window dropped, and counted
            continue
        L.append(-np.log(ph)); V.append(s.var()); T.append(t[i] + (w - 1) / 2)
    L, V, T = map(np.array, (L, V, T))
    return (stats.linregress(T, np.log(L)).slope, stats.linregress(T, np.log(V)).slope,
            len(r) - w + 1 - len(L))


# ---------------------------------------------------------------- estimator B
def tvar_negll(p, x0, x1, tau):
    a0, a1, b0, b1 = p
    lam = np.exp(a0 + a1 * tau)
    s = np.exp(b0 + b1 * tau)
    e = x1 - np.exp(-lam) * x0
    return np.sum(np.log(s) + 0.5 * (e / s) ** 2)


def tvar_fit(x0, x1, tau, p0=None):
    if p0 is None:
        ph = np.dot(x0, x1) / np.dot(x0, x0)
        p0 = [np.log(-np.log(np.clip(ph, 0.05, 0.95))), 0.0, np.log(np.std(x1 - ph * x0)), 0.0]
    res = optimize.minimize(tvar_negll, p0, args=(x0, x1, tau), method="Nelder-Mead",
                            options=dict(xatol=1e-7, fatol=1e-9, maxiter=20000, maxfev=20000))
    return res.x


def dlogvar_from(p, tau_eval):
    """d log Var/dt of the stationary AR(1) Var = s^2/(1-phi^2), at tau_eval (numerical)."""
    a0, a1, b0, b1 = p
    def lv(tt):
        lam = np.exp(a0 + a1 * tt); ph = np.exp(-lam); s = np.exp(b0 + b1 * tt)
        return np.log(s ** 2 / (1 - ph ** 2))
    h = 1e-3
    return (lv(tau_eval + h) - lv(tau_eval - h)) / (2 * h)


def pairs_block_boot(x0, x1, tau, blk, B):
    n = len(x0); out = []
    p_hat = tvar_fit(x0, x1, tau)
    nb = int(np.ceil(n / blk))
    for _ in range(B):
        starts = rng.integers(0, n - blk + 1, nb)
        idx = np.concatenate([np.arange(s, s + blk) for s in starts])[:n]
        out.append(tvar_fit(x0[idx], x1[idx], tau[idx], p0=p_hat))
    return p_hat, np.array(out)


def resid_block_series(p, x_first, tau_full, resid_std, blk):
    """Regenerate a series from the fitted TV-AR(1) with block-resampled std. residuals."""
    n = len(tau_full); nb = int(np.ceil(n / blk)); m = len(resid_std)
    starts = rng.integers(0, m - blk + 1, nb)
    e = np.concatenate([resid_std[s:s + blk] for s in starts])[:n]
    a0, a1, b0, b1 = p
    x = np.empty(n + 1); x[0] = x_first
    for k in range(n):
        x[k + 1] = np.exp(-np.exp(a0 + a1 * tau_full[k])) * x[k] + np.exp(b0 + b1 * tau_full[k]) * e[k]
    return x


def date(slope, a):
    return -a / slope if slope < 0 else np.inf
