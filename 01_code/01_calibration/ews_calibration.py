r"""
REPLICATION HEADER
  PRODUCES   02_output/figures/fig_ews_amoc_calibration.pdf; the SUMMARY block parsed into computed.json
  FEEDS      \thetaAMOChat \VarAMOChat \sigmaAMOChat \sigmaAMOCPten \sigmaAMOCPninety \HadISSTyears
  INPUTS     00_data/amoc/sg_index_hadisst.txt, the HadISST subpolar-gyre SST index of Caesar et al.
             2018, downloaded once from the PIK server by 00_data/amoc/fetch_pik_sg_index.py (not
             redistributed, md5 checked here); the run itself makes no network call
  SEED       none: the script draws no random numbers
  RUNTIME    about 7 s
  IMPLEMENTS sigma^2 = 2 lambda Var, the stationarity relation of sec:noise,
             with (lambda, Var) the CONTINUOUS-TIME OU parameters recovered from annual means
             by annual_mean_ou.py

ews_calibration.py
Calibrates sigma from AMOC early warning signals using Caesar (2018) SST fingerprint.

THE RECORD IS A SERIES OF ANNUAL MEANS. An earlier version of this script read the
restoring rate as lambda = -ln AC1 and the variance as the variance of the series, which is
right for a process sampled once a year and wrong for one averaged over the year, which is
what an SST index is. The AC1 of annual means of an OU of rate lambda is
(1 - e^-lambda)^2 / (2 (lambda - 1 + e^-lambda)), and their variance understates Var(X_t) by
lambda^2 / (2 (lambda - 1 + e^-lambda)). Both are now inverted by annual_mean_ou.py; the
point-sample readings are still printed, under labels collect_computed.py does not parse,
so the size of the correction stays visible. 05_record_study/puissance.py simulates
the record this way; this makes the calibration agree with it.

Formula (from fold SDE stationarity):
  sigma = 2 * |lambda/2|^{1/2} * sqrt(Var(X_t))
  equivalently: sigma = 2 * mu^{1/4} * sqrt(Var(X_t))  [mu = |lambda|/2]

Data: subpolar gyre SST index, HadISST, 1871-2016
      Caesar et al. (2018) Nature, PIK server
"""

import numpy as np
from scipy import stats
from scipy.signal import detrend as scipy_detrend
import re
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annual_mean_ou import lam_from_ac1, var_ratio  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 1. Read the data (local copy, md5-checked)
# ─────────────────────────────────────────────────────────────────────────────
# The series is read from the copy shipped in 00_data/amoc/, byte for byte the
# file the PIK server returns (00_data/amoc/PROVENANCE.md; fetch_pik_sg_index.py is the
# provenance tool that downloaded it, outside the run). The run touches no network. The
# synthetic fallback that used to replace a failed download is gone: a missing file stops
# the run. The years are read from the file's own header ("# for the years 1871-2016 in K")
# and checked against the number of values, rather than starting from a typed 1871.
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "00_data", "amoc", "sg_index_hadisst.txt")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs_literature import DERIVATION_INPUTS  # noqa: E402
if not os.path.exists(DATA):
    raise SystemExit(f"[ERROR] ews_calibration.py: the AMOC series is missing at {DATA}.\n"
                     "        It is not redistributed (no licence published by PIK). Download it once with\n"
                     "            python 00_data/amoc/fetch_pik_sg_index.py\n"
                     "        before run_all.sh; see 00_data/amoc/PROVENANCE.md.")
with open(DATA, "rb") as fh:
    _bytes = fh.read()
_md5 = hashlib.md5(_bytes).hexdigest()
_md5_expected = DERIVATION_INPUTS["amoc_series_md5"][0]
if _md5 != _md5_expected:
    raise SystemExit(f"[ERROR] ews_calibration.py: {DATA} has md5 {_md5}, not the {_md5_expected}\n"
                     "        of the series the paper was computed on (00_data/amoc/PROVENANCE.md).\n"
                     "        PIK may have updated the file; the numbers would not be the paper's.")
raw = _bytes.decode()
hdr = re.search(r"^#.*?years\s+(\d{4})\s*-\s*(\d{4})", raw, re.MULTILINE)
if not hdr:
    raise SystemExit("[ERROR] ews_calibration.py: no '# for the years YYYY-YYYY' header in the data file")
first_year, last_year = int(hdr.group(1)), int(hdr.group(2))
lines = [l.strip() for l in raw.strip().splitlines() if l.strip() and not l.startswith('#')]
sst   = np.array([float(l) for l in lines])
years = np.arange(first_year, first_year + len(sst), dtype=float)
if int(years[-1]) != last_year:
    raise SystemExit(f"[ERROR] ews_calibration.py: {len(sst)} values from {first_year} end in "
                     f"{int(years[-1])}, the header says {last_year}")
print(f"  Read: {len(years)} annual values, {years[0]:.0f}–{years[-1]:.0f} (local copy, 00_data/amoc/)")
print(f"  AMOC record years: first = {first_year}, last = {last_year}")
print(f"  Raw  Var(X) = {np.var(sst):.4f} K²   std = {np.std(sst):.4f} K")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Detrend: remove linear trend over full period
# ─────────────────────────────────────────────────────────────────────────────
slope, intercept, r, p, se = stats.linregress(years, sst)
trend = slope * years + intercept
sst_dt = sst - trend
var_raw_ann = float(np.var(sst))       # variance of the ANNUAL MEANS, raw
var_dt_ann  = float(np.var(sst_dt))    # variance of the ANNUAL MEANS, detrended
print(f"\n--- Detrending ---")
print(f"  Linear trend: {slope*10:.4f} K/decade  (p = {p:.3e})")
print(f"  Raw  Var of annual means = {var_raw_ann:.4f} K²")
print(f"  Det. Var of annual means = {var_dt_ann:.4f} K²")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Estimate restoring rate lambda from the AC1 of the detrended annual means
#    AC1 of annual means of an OU of rate lambda:
#        (1 - e^-lambda)^2 / (2 (lambda - 1 + e^-lambda))      -> inverted numerically
#    -ln(AC1) is the point-sample reading, printed for comparison only.
# ─────────────────────────────────────────────────────────────────────────────
phi, intercept_ar, r_ar, p_ar, se_ar = stats.linregress(sst_dt[:-1], sst_dt[1:])
if not 0 < phi < 1:
    raise SystemExit(f"AC1 = {phi:.4f} outside (0,1): no continuous-time restoring rate")
lam_pointsample = -np.log(phi)
lam_est = lam_from_ac1(phi)
vr = float(var_ratio(lam_est))
var_raw = var_raw_ann * vr             # process variance Var(X_t), raw
var_dt  = var_dt_ann * vr              # process variance Var(X_t), detrended

print(f"\n--- AR(1) fit on annual means ---")
print(f"  phi (AC1)  = {phi:.4f}")
print(f"  -ln AC1 (point-sample reading, superseded) = {lam_pointsample:.4f} yr⁻¹")
print(f"  |lambda|   = {lam_est:.4f} yr⁻¹   (continuous restoring rate, annual-mean inversion)")
print(f"  1/|lambda| = {1/lam_est:.1f} yr    (relaxation time)")
print(f"  process/annual-mean variance ratio = {vr:.4f}  ->  process variance, detrended: {var_dt:.4f} K²")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute sigma from fold SDE stationarity
#    Fold: dX = (mu - X²) dt + sigma dW
#    At stable eq X* = sqrt(mu), linearised restoring rate lambda = -2 sqrt(mu)
#    Stationarity: Var(X) = sigma² / (4 sqrt(mu)) = sigma² / (2 |lambda|)
#    => sigma = sqrt(2 |lambda| * Var)           [using full OU result]
#    => sigma = 2 (|lambda|/2)^{1/2} sqrt(Var)   [rewritten]
# ─────────────────────────────────────────────────────────────────────────────
sigma_clim_raw = np.sqrt(2 * lam_est * var_raw)   # K / sqrt(yr)
sigma_clim_dt  = np.sqrt(2 * lam_est * var_dt)    # K / sqrt(yr)

print(f"\n--- sigma from fold stationarity ---")
print(f"  sigma_clim (raw var)       = {sigma_clim_raw:.4f} K/sqrt(yr)")
print(f"  sigma_clim (detrended var) = {sigma_clim_dt:.4f} K/sqrt(yr)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Convert to fold-formula units
#    In the SCC formula sigma enters as sigma^{2/3} where sigma is the
#    noise in the fold SDE parameterised by emissions M (GtCO₂) as slow variable.
#    If M grows at rate E (GtCO₂/yr), then dW_t = dW_M * sqrt(E) (Brownian motion
#    conversion), so sigma_M = sigma_clim / sqrt(E).
# ─────────────────────────────────────────────────────────────────────────────
E_current = 40.0  # GtCO₂/yr  (current global emissions)

sigma_M_raw = sigma_clim_raw / np.sqrt(E_current)
sigma_M_dt  = sigma_clim_dt  / np.sqrt(E_current)

print(f"\n--- sigma in fold-formula units (emissions domain) ---")
print(f"  Emission rate E = {E_current} GtCO₂/yr")
print(f"  sigma_M = sigma_clim / sqrt(E)")
print(f"  sigma_M (raw var)       = {sigma_M_raw:.4f} K/sqrt(GtCO₂)")
print(f"  sigma_M (detrended var) = {sigma_M_dt:.4f} K/sqrt(GtCO₂)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. (removed) An "implied SCC" printout stood here. It evaluated the
#    amplitude by the abandoned Airy form, and no script or macro read its lines.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 7. Sliding-window EWS: show variance and AC1 trend (validates Boers 2021)
# ─────────────────────────────────────────────────────────────────────────────
win = 40  # 40-year window (Boers uses 60 yr, we use 40 for larger sample)
n = len(sst_dt)
w_years, w_var, w_ac1, w_lam = [], [], [], []

for i in range(n - win):
    seg = sst_dt[i:i+win]
    phi_w, _, _, _, _ = stats.linregress(seg[:-1], seg[1:])
    ac1_w = phi_w
    # annual-mean inversion, as for the full sample; a window with AC1 outside (0,1) has
    # no continuous rate and is dropped (never met on the HadISST record)
    lam_w = lam_from_ac1(phi_w) if 0 < phi_w < 1 else np.nan
    var_w = float(np.var(seg)) * (float(var_ratio(lam_w)) if np.isfinite(lam_w) else np.nan)
    w_years.append(float(years[i + win//2]))
    w_var.append(var_w)
    w_ac1.append(ac1_w)
    w_lam.append(lam_w)

w_years = np.array(w_years)
w_var   = np.array(w_var)
w_ac1   = np.array(w_ac1)
w_lam   = np.array(w_lam)
ok_w    = np.isfinite(w_lam)
print(f"\n  windows dropped (AC1 outside (0,1)): {int((~ok_w).sum())}")
w_years, w_var, w_ac1, w_lam = w_years[ok_w], w_var[ok_w], w_ac1[ok_w], w_lam[ok_w]

slope_var, _, r_var, p_var, _ = stats.linregress(w_years, w_var)
slope_lam, _, r_lam, p_lam, _ = stats.linregress(w_years, w_lam)

print(f"\n--- Sliding-window EWS ({win}-yr) ---")
print(f"  Variance  trend: {slope_var*10:.5f} K²/decade  (r={r_var:.3f}, p={p_var:.3e})")
print(f"  Lambda    trend: {slope_lam*10:.5f} /decade    (r={r_lam:.3f}, p={p_lam:.3e})")
print(f"  Current |lambda| (last window): {w_lam[-1]:.4f} yr⁻¹")

# Final sigma using last-window lambda
lam_last = w_lam[-1]
sigma_last = np.sqrt(2 * lam_last * var_dt)
sigma_M_last = sigma_last / np.sqrt(E_current)
print(f"\n  sigma_clim (last window, detrended): {sigma_last:.4f} K/sqrt(yr)")
print(f"  sigma_M    (last window):             {sigma_M_last:.4f} K/sqrt(GtCO₂)")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Figure: EWS trends + sigma evolution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.plot(years, sst, 'k-', lw=0.8, alpha=0.7)
ax.plot(years, trend, 'r--', lw=1.5, label=f'Trend {slope*10:.3f} K/dec')
ax.set_title('AMOC SST fingerprint (Caesar 2018, HadISST)')
ax.set_ylabel('SST subpolar gyre index (K)')
ax.legend(fontsize=8)
ax.set_xlabel('Year')

ax = axes[0, 1]
ax.plot(years, sst_dt, 'steelblue', lw=0.8, alpha=0.8)
ax.axhline(0, color='k', lw=0.5)
ax.set_title('Detrended fingerprint')
ax.set_ylabel('Anomaly (K)')
ax.set_xlabel('Year')

ax = axes[1, 0]
ax.plot(w_years, w_var, 'darkorange', lw=1.2)
slope_var_line = slope_var*(w_years - w_years[0]) + w_var[0]
ax.plot(w_years, slope_var_line, 'r--', lw=1, alpha=0.7,
        label=f'Trend p={p_var:.3f}')
ax.set_title(f'Sliding variance ({win}-yr window)')
ax.set_ylabel('Var(X_t)  [K²]')
ax.set_xlabel('Year')
ax.legend(fontsize=8)

ax = axes[1, 1]
sigma_slide = np.sqrt(2 * w_lam * var_dt)
sigma_M_slide = sigma_slide / np.sqrt(E_current)
ax.plot(w_years, sigma_slide, 'purple', lw=1.2, label='sigma_clim [K/√yr]')
ax.plot(w_years, sigma_M_slide, 'darkgreen', lw=1.2, label='sigma_M [K/√GtCO₂]')
ax.axhline(sigma_M_dt, color='darkgreen', lw=0.8, ls=':', alpha=0.5)
ax.set_title('Identified sigma (fold EWS formula)')
ax.set_ylabel('sigma')
ax.set_xlabel('Year')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('fig_ews_amoc_calibration.pdf', bbox_inches='tight')
plt.savefig('fig_ews_amoc_calibration.png', dpi=150, bbox_inches='tight')
print("\nFigure saved: fig_ews_amoc_calibration.pdf / .png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Summary table for paper
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — for paper calibration table")
print("="*60)
print(f"  Data: Caesar (2018) HadISST subpolar gyre index, {years[0]:.0f}–{years[-1]:.0f}")
print(f"  Var(X_t) detrended           = {var_dt:.3f} K²   (process variance, from annual means)")
print(f"  Var of annual means (detr.)  = {var_dt_ann:.3f} K²   (not parsed)")
print(f"  AC1 (annual)                 = {phi:.4f}")
print(f"  -ln AC1 (superseded reading) = {lam_pointsample:.4f} yr⁻¹   (not parsed)")
print(f"  |lambda| (full-sample)       = {lam_est:.4f} yr⁻¹")
print(f"  |lambda| (last {win}-yr window)  = {lam_last:.4f} yr⁻¹")
print(f"  sigma_clim (full sample)     = {sigma_clim_dt:.4f} K/sqrt(yr)")
print(f"  sigma_M    (full sample)     = {sigma_M_dt:.4f} K/sqrt(GtCO₂)")
print(f"  sigma_M    (last window)     = {sigma_M_last:.4f} K/sqrt(GtCO₂)")
print(f"  Variance trend p-value       = {p_var:.3e}  ({'significant' if p_var < 0.05 else 'not sig.'})")
print(f"  Lambda   trend p-value       = {p_lam:.3e}  ({'significant' if p_lam < 0.05 else 'not sig.'})")
print("="*60)
