#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   stdout only (captured by run_all.sh to 02_output/values/raw_runs/record_detectability.out)
  FEEDS      the computed \rec* macros of values.tex, via collect_computed.py; and
             fig_record_power.py, which imports compute() instead of re-reading the study
  INPUTS     02_output/record_study/*.json: seven of the eight outputs of the
             record-detectability study, written by 01_code/05_record_study/ (or copied
             from 00_data/record_study/ when run_all.sh runs with RECORD_STUDY=cached)
  SEED       none of its own. It READS results; it re-runs no simulation. The simulations
             behind the JSON files carry their own seeds, stated in 01_code/05_record_study/.
  RUNTIME    under 1 s
  IMPLEMENTS no equation of the paper. One closed form is CHECKED numerically, not fitted:
             the limiting OLS slope of a log(L - n) on n in [0, L] (the saturation appendix: n, time since the start of the record), which is -3a/L.

record_detectability.py — what the instrumental record can establish about the geometry.

Four groups of numbers, all read off the study outputs:

  1. The Neyman-Pearson bound: the power of the best test of ANY statistic, against the
     least favourable null, on a record of L years. Read at the AMOC's distance (t_c equal to
     the deadline bound at central inputs) and near the threshold, on the fine grid of the
     null parameter (horizon_attente.json, block "fine").
  2. Saturation: the OLS slope of log lambda-hat on a record ending at the threshold tends to
     -3a/L, bounded; its standard error falls as L^{-3/2}; so z grows as a sqrt(L). The
     constants are exact. The law holds while lambda.w stays above ~10 (estimateurs.json).
  3. The horizon on the waiting diagonal: the record lengthens while the threshold nears.
     Years before the threshold at which the fold reaches 50% and 80% power, years from now,
     record length then, and the share of the remaining budget already spent.
  4. The Ditlevsen-Ditlevsen (2023) test: its size under constant noise over the range of
     sigma^2 trends the record cannot exclude (specification B, the 95% joint region cut),
     its size with the noise left free, and its power with the noise free as a share of the
     bound at the same distance.

Every value is printed on a line "REC <macro> = <value>", with the rounding chosen here.
A missing input file is fatal and named. Nothing is defaulted.

Usage:  python 01_code/01_calibration/record_detectability.py
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent                    # replication/
STUDY = PKG / "02_output" / "record_study"  # written by 01_code/05_record_study/ or copied by run_all.sh


def load(name):
    p = STUDY / name
    if not p.exists():
        raise SystemExit("[ERROR] record_detectability.py: %s is missing. This script reads "
                         "the outputs of the record-detectability study (01_code/05_record_study/, "
                         "run by run_all.sh, or copied from 00_data/record_study/ with "
                         "RECORD_STUDY=cached) and "
                         "recomputes nothing; without them the \\rec* macros do not exist."
                         % p)
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x, small_dec=1):
    """A share as a percentage: one decimal below 10%, whole percent above. The studies run
    800 records per cell, a Monte-Carlo SE of ~0.4 point at 2% and ~1.8 point at 50%."""
    v = 100.0 * x
    return ("%%.%df" % small_dec) % v if v < 10 else "%.0f" % v


def compute(printing=False):
    """Every macro value, and the study outputs they were read from.

    Returns (out, ctx): out maps macro name -> formatted value, exactly what main() prints;
    ctx carries the loaded JSON and the two constants the conversions use (the record length
    L and the diagonal's starting t_c), so that fig_record_power.py draws from the SAME
    evaluation as the macros instead of repeating the conversion. With printing=True the
    control lines and the REC lines are printed, which is main()."""
    att = load("horizon_attente.json")
    hnp = load("horizon_np.json")
    mle = load("mle_ditlevsen.json")
    c82 = load("calib82.json")
    est = load("estimateurs.json")
    out = {}

    # ── 1. Neyman-Pearson bound, fine grid of the null, L = record length ──
    fine = {(c["model"], c["tc"]): c for c in att["fine"]}
    L = {c["L"] for c in att["fine"]}
    if len(L) != 1:
        raise SystemExit("[ERROR] the fine grid mixes record lengths %s" % sorted(L))
    L = L.pop()
    tc_now = max(tc for _, tc in fine)          # the AMOC's deadline bound, central inputs
    tc_near = min(tc for _, tc in fine)
    fold_now = fine[("lin-1/2", tc_now)]
    fold_near = fine[("lin-1/2", tc_near)]
    trans_now = fine[("lin-1", tc_now)]
    b_fold_now = min(fold_now["by_theta"].values())
    b_fold_near = min(fold_near["by_theta"].values())
    b_trans_now = min(trans_now["by_theta"].values())
    out["recRecordYears"] = "%d" % L
    out["recNPtcNow"] = "%.0f" % tc_now
    out["recNPtcNear"] = "%g" % tc_near
    out["recNPFoldNow"] = "%.2f" % b_fold_now
    out["recNPTransNow"] = "%.2f" % b_trans_now
    out["recNPFoldNear"] = "%.2f" % b_fold_near

    # ── 2. Saturation: the limiting slope, checked; the exponents, exact ──
    # OLS slope of log(L - n) on n uniform on [0, L]: cov(n, log(L-n)) = -L/4 and
    # var(n) = L^2/12, so the slope is -3/L. Checked on a fine midpoint grid.
    n = 100000
    xs = [(i + 0.5) / n for i in range(n)]
    ys = [math.log(1.0 - x) for x in xs]
    mx, my = sum(xs) / n, sum(ys) / n
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    if abs(slope + 3.0) > 1e-3:
        raise SystemExit("[ERROR] limiting slope %.5f, expected -3" % slope)
    out["recSlopeLimitCoef"] = "3"
    out["recSEExponent"] = "3/2"
    # validity of z ~ a sqrt(L): lambda at the last window times the window, w = est["W"]
    w = est["W"]
    lw146 = [c["lam_true_last"] * w for c in est["cells"] if c["L"] == 146]
    cells292 = {c["model"]: c for c in est["cells"] if c["L"] == 292}
    out["recWindowYears"] = "%d" % w
    out["recThetaWFloor"] = "10"   # the threshold the study states; evidence printed below
    out["recThetaWMinValid"] = "%.1f" % min(lw146)
    out["recThetaWFail"] = "%.1f" % (cells292["lin-1"]["lam_true_last"] * w)
    out["recBiasFailPct"] = "%.0f" % (100 * cells292["lin-1"]["bias_last"])
    out["recThetaWPass"] = "%.1f" % (cells292["lin-1/2"]["lam_true_last"] * w)
    out["recBiasPassPct"] = "%.0f" % (100 * cells292["lin-1/2"]["bias_last"])
    if not (min(lw146) < 10 < cells292["lin-1/2"]["lam_true_last"] * w
            and cells292["lin-1"]["lam_true_last"] * w < 10):
        raise SystemExit("[ERROR] the stated floor lambda.w ~ 10 no longer separates the cells")

    # ── 3. Horizon on the waiting diagonal ──
    diag = {d["model"]: d for d in att["diagonal"]}
    tc0 = diag["lin-1/2"]["cells"][0]["tc"]
    if tc0 != tc_now:
        raise SystemExit("[ERROR] the diagonal starts at t_c = %s, the fine grid at %s"
                         % (tc0, tc_now))
    # LaTeX command names take letters only, hence Fifty/Eighty rather than 50/80.
    for model, tag in (("lin-1/2", "Fold"), ("lin-1", "Trans")):
        for lvl, word in (("50", "Fifty"), ("80", "Eighty")):
            h = diag[model]["h" + lvl]
            out["recHorizon%s%sToThreshold" % (tag, word)] = "%.1f" % h if tag == "Fold" else "%.0f" % h
            out["recHorizon%s%sFromNow" % (tag, word)] = "%.0f" % (tc0 - h)
            if tag == "Fold":
                out["recHorizon%s%sRecord" % (tag, word)] = "%.0f" % (L + tc0 - h)
                out["recHorizon%s%sSpentPct" % (tag, word)] = "%.0f" % (100 * (1 - h / tc0))
    fixed = [x for x in hnp["horizons"] if x["model"] == "lin-1/2" and x["L"] == L][0]
    out["recHorizonFoldFiftyFixedRecord"] = "%.1f" % fixed["h50"]
    if not (isinstance(fixed["h80"], float) and math.isnan(fixed["h80"])):
        raise SystemExit("[ERROR] the fold now reaches 80%% at fixed L = %d" % L)

    # ── 4. Ditlevsen-Ditlevsen (2023): size and power ──
    cells = {c["label"]: c for c in c82["cells"]}
    region = [r for r in c82["regions"] if r["key"].startswith("B_")][0]
    out["recSigmaTrendLo"] = "%.2f" % (100 * region["s_lo"])
    out["recSigmaTrendHi"] = "%.2f" % (100 * region["s_hi"])
    out["recSigmaTrendCentre"] = "%.2f" % (100 * region["s_star"])
    s_const = "stat_PLI_const"   # the published test fits a fold with constant noise
    s_free = "stat_PLI_libre"
    out["recDDConstSizeLo"] = pct(cells["B:s_lo"]["reject"][s_const])
    out["recDDConstSizeHi"] = pct(cells["B:s_hi"]["reject"][s_const])
    out["recDDConstSizeCentreA"] = pct(cells["A:s_star"]["reject"][s_const])
    out["recDDConstSizeCentreB"] = pct(cells["B:s_star"]["reject"][s_const])
    out["recDDConstSizeSigmaRising"] = pct(mle["size_sigma_rising"][s_const])
    free = [c["reject"][s_free] for c in c82["cells"]] + [mle["size_sigma_rising"][s_free]]
    out["recDDFreeSizeLo"] = "%.0f" % (100 * min(free))
    out["recDDFreeSizeHi"] = "%.0f" % (100 * max(free))
    bound = {(c["model"], c["tc"]): c["bound"] for c in hnp["cells"] if c["L"] == L}
    share = []
    for c in mle["cells"]:
        if c["model"] == "lin-1/2" and c["tc"] in (2, 5):
            share.append(c["power"][s_free] / bound[("lin-1/2", c["tc"])])
    if len(share) != 2:
        raise SystemExit("[ERROR] expected the fold at t_c = 2 and 5 in mle_ditlevsen.json")
    # The class a test presupposes: fitting a fold or a transcritical to the same records.
    # Largest gap in power between the two, over every cell and both noise treatments. An
    # earlier summary said "within 0.03"; the cells give more (0.042, noise free, true
    # transcritical at t_c = 20), inside the +/-0.04 Monte-Carlo interval at 500 records.
    gaps = [abs(c["power"]["stat_PLI_" + v] - c["power"]["stat_TRANS_" + v])
            for c in mle["cells"] for v in ("const", "libre")]
    out["recClassPowerGap"] = "%.2f" % max(gaps)
    out["recDDFreeShareOfBoundLo"] = "%.0f" % (100 * min(share))
    out["recDDFreeShareOfBoundHi"] = "%.0f" % (100 * max(share))

    # ── the NP bound at t_c now depends on the grid of the least favourable null: the fine
    # grid (reported), the waiting diagonal and horizon_np give a range, quoted in the record appendix ──
    grids = [b_fold_now, diag["lin-1/2"]["cells"][0]["bound"], bound[("lin-1/2", tc_now)]]
    out["recNPFoldNowGridLo"] = "%.2f" % min(grids)
    out["recNPFoldNowGridHi"] = "%.2f" % max(grids)

    # ── optimal window: the best window on the diagonal grid, most frequent over the cells ──
    fen = load("fenetres.json")
    ws = [c["best_diag"]["w"] for c in fen["cells"]]
    w_opt = max(set(ws), key=ws.count)
    out["recWindowOptYears"] = "%d" % w_opt
    out["recWindowOptCells"] = "%d" % ws.count(w_opt)
    out["recWindowCells"] = "%d" % len(ws)
    out["recWindowOptSharePct"] = "%.0f" % (100.0 * w_opt / L)

    # ── The joint 95% region of the two trends (robust region, percentile CIs) ──
    p11 = STUDY / "identification.json"
    if not p11.exists():
        raise SystemExit("[ERROR] record_detectability.py: %s is missing" % p11)
    idf = json.loads(p11.read_text(encoding="utf-8"))
    E = idf["A"] + idf["B"]
    rays = [e["ray_sig2_in_rob"] and e["ray_lam_in_rob"] for e in E]
    out["recJointSpecs"] = "%d" % len(E)
    out["recJointRaysIn"] = "%d" % sum(rays)
    out["recJointOriginIn"] = "%d" % sum(e["origin_in_rob"] for e in E)
    out["recJointThetaDecline"] = "%d" % sum(e["pct_lam"][1] < 0 for e in E)
    out["recJointSigChange"] = "%d" % sum(e["pct_sig2"][0] > 0 or e["pct_sig2"][1] < 0 for e in E)

    # ── The record length a TREND statistic would need, for a fold, even at the
    # threshold. z is the mean |slope|/sd of the trend pipeline (record_pipeline.py) at t_c = 3 and 5
    # (both estimators) on L = 146; z grows as sqrt(L), so L_needed = L (z_target/z)^2.
    # Taken near the threshold, where z has saturated: at the present distance the
    # requirement is longer still.
    pw = load("puissance.json")
    zs = [abs(o["slope_mean"]) / o["slope_sd"] for o in pw
          if o["model"] == "lin-1/2" and o["tc"] in (3, 5)]
    if len(zs) != 4:
        raise SystemExit("[ERROR] expected four fold cells at t_c = 3, 5 in puissance.json")
    z146 = sum(zs) / len(zs)
    out["recTrendLengthFifty"] = "%.0f" % (L * (1.96 / z146) ** 2)

    for k in out:
        if not k.isalpha():
            raise SystemExit("[ERROR] macro name %r is not letters only" % k)
    ctx = {"att": att, "hnp": hnp, "L": L, "tc0": tc0,
           "b_fold_now": b_fold_now, "b_fold_near": b_fold_near}
    if not printing:
        return out, ctx

    # ── controls, printed so the capture shows what each rounding rests on ──
    print("record_detectability.py — what the record can establish, read from the study")
    print("  record L = %d yr; t_c now = %s yr; near = %s yr" % (L, tc_now, tc_near))
    print("  NP bound, fine grid: fold now %.4f (theta %s), trans now %.4f, fold near %.4f"
          % (b_fold_now, min(fold_now["by_theta"], key=fold_now["by_theta"].get),
             b_trans_now, b_fold_near))
    print("  NP bound, coarse grids at t_c now (control, not reported): diagonal %.4f, "
          "horizon_np %.4f" % (diag["lin-1/2"]["cells"][0]["bound"],
                                bound[("lin-1/2", tc_now)]))
    print("  limiting OLS slope x L = %.5f (exact -3)" % slope)
    print("  lambda.w at L=146: min %.2f; L=292: fold %.2f (bias %.3f), trans %.2f (bias %.3f)"
          % (min(lw146), cells292["lin-1/2"]["lam_true_last"] * w, cells292["lin-1/2"]["bias_last"],
             cells292["lin-1"]["lam_true_last"] * w, cells292["lin-1"]["bias_last"]))
    print("  horizon fold: h50 = %.3f, h80 = %.3f yr before threshold (t_c0 = %s)"
          % (diag["lin-1/2"]["h50"], diag["lin-1/2"]["h80"], tc0))
    print("  DD23 const size, spec B: s_lo %.4f  s_star %.4f  s_hi %.4f (fold); "
          "trans s_hi %.4f" % (cells["B:s_lo"]["reject"][s_const],
                               cells["B:s_star"]["reject"][s_const],
                               cells["B:s_hi"]["reject"][s_const],
                               cells["B:s_hi"]["reject"]["stat_TRANS_const"]))
    print("  DD23 free size: min %.4f  max %.4f" % (min(free), max(free)))
    print("  DD23 free power / bound, fold, t_c 2 and 5: %s" % ", ".join("%.3f" % x for x in share))
    print()
    for k in sorted(out):
        print("REC %-32s = %s" % (k, out[k]))
    return out, ctx


def main():
    compute(printing=True)


if __name__ == "__main__":
    main()
