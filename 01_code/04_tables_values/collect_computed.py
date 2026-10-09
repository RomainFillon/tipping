#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   02_output/computed.json
  FEEDS      every computed macro of values.tex, through make_tables_values.py
  INPUTS     02_output/values/raw_runs/*.out, captured by run_all.sh from this run
  SEED       none of its own; inherits each script's seed
  RUNTIME    under 1 s
  IMPLEMENTS no equation of its own, except four closed forms stated in the paper: eps_now, eps_dyn (self-consistent), the bound at central inputs, and the band share

collect_computed.py — produce 02_output/computed.json from the REAL outputs of the
generator scripts (seeded), never by manual transcription or re-implemented formulas.

Source, per script: the stdout run_all.sh captured while running it, in
02_output/values/raw_runs/<key>.out. This script executes nothing itself, so every number
in computed.json comes from the one execution the run performed.

The only values computed HERE (not parsed) are exact, verifiable closed forms stated in
the paper: the current proximity  eps_now = (DT* - DT_current)/DT*  (from sourced inputs),
and the sigma prior percentiles (lognormal of the EWS point estimate). Everything else is
PARSED from a script's printed result. Each entry records {value, source, requires_network}.

Usage:  python 01_code/04_tables_values/collect_computed.py   (after step 1/5 of run_all.sh)
"""
import os, re, sys, json, math, subprocess, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ROOT = CODE.parent
RAW  = ROOT / "02_output" / "values" / "raw_runs"
OUTJSON = ROOT / "02_output" / "computed.json"
sys.path.insert(0, str(CODE))
from inputs_literature import DERIVATION_INPUTS, LITERATURE_MACROS  # noqa: E402

SCRIPTS = {
    "calibration_ci": CODE / "01_calibration" / "calibration_ci.py",
    "eps_inf_ci":     CODE / "01_calibration" / "eps_inf_ci.py",
    "bhp_bound":      CODE / "01_calibration" / "bhp_bound.py",
    "crossover":      CODE / "01_calibration" / "crossover.py",
    "ews_calibration": CODE / "01_calibration" / "ews_calibration.py",
    "fig_forest_tc":  CODE / "03_figures" / "fig_forest_tc.py",
    "reduced_form_scc": CODE / "01_calibration" / "reduced_form_scc.py",
    "noise_reversal_frontier": CODE / "02_solvers" / "noise_reversal_frontier.py",
    "record_detectability": CODE / "01_calibration" / "record_detectability.py",
    "validation_bvp": CODE / "02_solvers" / "validation_bvp.py",
    "moving_budget":  CODE / "02_solvers" / "moving_budget.py",
}


def get_out(key):
    """Read the output run_all.sh captured for this generator. It does NOT run anything.

    One execution per script, in run_all.sh, captured there by tee. A missing capture is a
    failure with the file's name attached, never a silent re-run: a generator executed here
    would be a SECOND execution, and nothing in the output would then say which of the two
    a macro came from. That question has to have one answer, so there is one execution.
    """
    cache = RAW / (key + ".out")
    if not cache.exists():
        raise SystemExit(
            f"[ERROR] no captured output for '{key}' at {cache}.\n"
            f"        collect_computed.py reads what run_all.sh captured; it does not run\n"
            f"        generators itself. Run the package's entry point:\n"
            f"            bash run_all.sh")
    return cache.read_text(encoding="utf-8", errors="replace")


def main():
    argparse.ArgumentParser(
        description="Read the outputs run_all.sh captured and write computed.json."
    ).parse_args()

    calib   = get_out("calibration_ci")
    epsi    = get_out("eps_inf_ci")
    ews     = get_out("ews_calibration")
    forest  = get_out("fig_forest_tc")
    reduced = get_out("reduced_form_scc")

    C = {}  # macro -> dict

    def put(macro, value, source, network=False):
        C[macro] = {"value": str(value), "source": source, "requires_network": network}

    # ── parsed: calibration KEY NUMBERS (M*_P50, f, SCC P50 @ eps=0.15) ──
    # line: "AMOC: M*_P50=1742 GtCO2 | T_P50=100 yr | f(rhoT)=0.317 | SCC P50=$9 [$2, $31]"
    # Note: the per-element reconciliation SCC macros (\scc*recon, \sccTotalRecon) were
    # removed in an earlier cleanup (unused in paper.tex), so they are no
    # longer extracted here. Only M*_P50 and Omega(rho H) (printed f(rhoT)) are kept.
    cmap = {"AMOC": ("MstarAMOCmed", "OmegaAMOCmed"),
            "Amazon": ("MstarAmazonmed", "OmegaAmazonmed"),
            "WAIS": ("MstarWAISmed", "OmegaWAISmed")}
    for m in re.finditer(r"^\s*(\w+):\s*M\*_P50=(\d+).*?f\([^)]*\)=([\d.]+)",
                         calib, re.MULTILINE):
        el, mstar, f = m.groups()
        if el in cmap:
            mm, fm = cmap[el]
            put(mm, mstar, "calibration_ci.py — M*_P50 (MC seed=42)")
            # Two significant figures, chosen here (as for the F of tab:eps_dyn);
            # the two decimals of the text used to be copied from the previous values.tex.
            put(fm, f"{float(f):#.2g}", "calibration_ci.py — Omega(rho H) median, the realization factor")

    # ── parsed: the share of TCRE draws the rejection discards.
    # Printed by calibration_ci.py from tcre_draw.draw_tcre, seed 42; the calibration-inputs appendix quotes it.
    m = re.search(r"^TCRE DRAW: .*discarded share = ([\d.]+)%", calib, re.MULTILINE)
    if m:
        put("TCREdiscardPct", f"{float(m.group(1)):.1f}",
            "calibration_ci.py — tcre_draw.draw_tcre, share of the normal draws <= 0 discarded "
            "by rejection, % of all draws made (seed=42)")

    # ── parsed: ews_calibration SUMMARY ──
    def grab(pat, txt):
        m = re.search(pat, txt)
        return m.group(1) if m else None
    var = grab(r"Var\(X_t\) detrended\s*=\s*([\d.]+)", ews)
    lam = grab(r"\|lambda\|\s*\(full-sample\)\s*=\s*([\d.]+)", ews)
    sig = grab(r"sigma_clim\s*\(full sample\)\s*=\s*([\d.]+)", ews)
    yrs = grab(r"Read:\s*(\d+)\s*annual", ews)
    # The first and last year of the record, read by ews_calibration.py from the
    # data file's own header (E.2 wrote "1871--2016" by hand).
    y0 = grab(r"AMOC record years: first = (\d{4})", ews)
    y1 = grab(r"AMOC record years: first = \d{4}, last = (\d{4})", ews)
    if lam and not (y0 and y1):
        raise SystemExit("[ERROR] the ews_calibration capture lacks the 'AMOC record years' line; "
                         "rerun run_all.sh.")
    if y0:
        put("AMOCrecordFirstYear", y0, "ews_calibration.py — first year of the series, read from "
            "the header of 00_data/amoc/sg_index_hadisst.txt", True)
        put("AMOCrecordLastYear", y1, "ews_calibration.py — last year of the series, read from "
            "the header of 00_data/amoc/sg_index_hadisst.txt", True)
    if var: put("VarAMOChat", f"{float(var):.3f}", "ews_calibration.py — detrended variance (HadISST)", True)
    if lam: put("thetaAMOChat", f"{float(lam):.2f}", "ews_calibration.py — AR(1) restoring rate", True)
    if sig: put("sigmaAMOChat", f"{float(sig):.3f}", "ews_calibration.py — sigma_clim detrended", True)
    if yrs: put("HadISSTyears", yrs, "HadISST record length (Caesar 2018)", True)

    # ── closed form: what reading annual means as point samples would do to sigma-hat ──
    # sigma^2 = 2 lambda Var. The point-sample reading takes lambda = -ln AC1 and Var = the
    # variance of the annual means; the annual-mean inversion (annual_mean_ou.py) raises both.
    # The text reports the total factor and, in a note, the part that comes through lambda
    # alone. Both are ratios of lines ews_calibration.py prints at four decimals (not the
    # three-decimal SUMMARY, whose rounded variances would move the factor by 0.01).
    lam_ps = grab(r"-ln AC1 \(point-sample reading, superseded\)\s*=\s*([\d.]+)", ews)
    vratio = grab(r"process/annual-mean variance ratio\s*=\s*([\d.]+)", ews)
    lam4 = grab(r"\|lambda\|\s*=\s*([\d.]+)\s*yr", ews)
    if lam:
        if not (lam_ps and vratio and lam4):
            raise SystemExit("[ERROR] the ews_calibration capture lacks the point-sample lines "
                             "(-ln AC1, variance ratio): it predates the annual-mean inversion. "
                             "Rerun run_all.sh.")
        r_lam = float(lam4) / float(lam_ps)
        put("sigmaAggregationFactor", f"{math.sqrt(r_lam * float(vratio)):.2f}",
            f"closed form: sigma-hat (annual-mean inversion) / sigma-hat (point-sample reading) "
            f"= sqrt((lambda/-ln AC1) x Var(X_t)/Var(annual means)), lambda={lam4}, "
            f"-ln AC1={lam_ps}, variance ratio={vratio} (ews_calibration.py)", True)
        # 2 rho / lambda at the AMOC calibration: the ratio the sign of the noise response is
        # conditioned on (not reported in the paper), set against \kappaReversalCeiling in the proof of the sign of the noise response.
        put("kappaAMOC", f"{2 * float(LITERATURE_MACROS['rhoBase'][0]) / float(lam4):.3f}",
            f"closed form: 2 rho / lambda, rho={LITERATURE_MACROS['rhoBase'][0]} (rhoBase), "
            f"lambda={lam4} (ews_calibration.py)", True)
        put("sigmaAggregationFactorTheta", f"{math.sqrt(r_lam):.2f}",
            f"closed form: the same factor through the recovery rate alone, sqrt(lambda/-ln AC1), "
            f"lambda={lam4}, -ln AC1={lam_ps} (ews_calibration.py)", True)

    # ── the Monte-Carlo prior median must be the estimate tab:inputs reports ──
    # calibration_ci, eps_inf_ci and fig_forest_tc draw sigma_AMOC around
    # DERIVATION_INPUTS["sigma_mu_AMOC"], declared by hand because they run before the
    # network step. This is the one place that sees both, so the drift is caught here: in
    # earlier versions the draws were centred on 0.258, written out three times, and nothing checked it.
    if sig:
        s_mc = DERIVATION_INPUTS["sigma_mu_AMOC"][0]
        if abs(float(sig) - s_mc) > 5e-4:
            raise SystemExit(
                f"[ERROR] the Monte-Carlo AMOC sigma prior is centred on {s_mc} "
                f"(inputs_literature.DERIVATION_INPUTS['sigma_mu_AMOC']) but ews_calibration.py "
                f"estimates {sig}. Update the declared median, then rerun the three MC scripts.")

    # ── closed form: sigma prior P10/P90 (lognormal of the EWS point estimate) ──
    if sig:
        m_, cv = float(sig), DERIVATION_INPUTS["sigma_cv_AMOC"][0]
        s = math.sqrt(math.log(1 + cv * cv))
        p10 = m_ * math.exp(-1.2815515 * s)
        p90 = m_ * math.exp(+1.2815515 * s)
        src = f"closed form: lognormal(median={m_}, cv={cv}) percentiles [{DERIVATION_INPUTS['sigma_cv_AMOC'][1]}]"
        put("sigmaAMOCPten", f"{p10:.2f}", src, True)
        put("sigmaAMOCPninety", f"{p90:.2f}", src, True)

    # ── sourced input, emitted as a macro so the text stops writing it by hand ──
    # DT_0 feeds three derived macros (the proximities) and the deadline bound, and it was
    # the last sourced number the text still spelled out.
    dtc = DERIVATION_INPUTS["DT_current"][0]
    put("DTzero", f"{dtc}", f"{DERIVATION_INPUTS['DT_current'][1]}")

    # ── closed form: current proximity eps_now = (DT* - DT_current)/DT* ──
    for el, macro in [("AMOC", "epsAMOCnat"), ("Amazon", "epsAmazonnat"),
                      ("WAIS", "epsWAISnat")]:  # SPG removed (T-D)
        dts = LITERATURE_MACROS.get(f"DTstar{el}")
        if dts:
            v = (float(dts[0]) - dtc) / float(dts[0])
            put(macro, f"{v:.2f}", f"closed form: (DT*-{dtc})/DT*, DT*={dts[0]} (AM2022), DT_cur={dtc} (AR6)")

    # ── closed form: the dynamic scale eps_dyn, solved SELF-CONSISTENTLY ──
    # The condition E_0/(M* eps) << lambda compares two speeds, the budget draining against
    # the system's own recovery, and eps_dyn is where they meet. The two sides must be read
    # at the SAME proximity. lambda is the local recovery rate: eq:gen_sde defines
    # lambda(mu) = -f'(x*(mu)), and for the fold lambda = 2 sqrt(mu), which vanishes as the
    # threshold nears; since eps = mu/M*, lambda(eps) = lambda_hat sqrt(eps/eps_now).
    # Solving E_0/(M* eps) = lambda(eps) therefore gives
    #     eps_dyn = ( (E_0/(M* lambda_hat)) sqrt(eps_now) )^(2/3),
    # and the 2/3 exponent IS the critical slowing-down factor: it is what appears when the
    # recovery rate declines like sqrt(eps) instead of holding at today's value.
    #
    # The superseded form solved the same condition with lambda frozen at the measured
    # present rate, giving eps_dyn = E_0/(M* lambda) = 0.0089. That is not a different
    # modelling choice but the same equation solved half-way: it evaluates the left side at
    # the eps being solved for and the right side at today, and it assumes the system keeps
    # its present resilience for ever, which is the assumption the critical-slowing-down
    # section refutes. This quantity has carried four different values across the project
    # -- 0.013 hard-coded, 0.0089 on a frozen lambda, 0.115 in a root script running on
    # M* = 1745, and 0.0380 here -- which is why it is computed in one place from named
    # inputs rather than quoted.
    #
    # The same closed form for every element and recovery rate the paper reads it
    # at (the appendix on where the elements sit, tab:eps_dyn), on the three emission paths, with the largest factor by
    # which the geometry can lift the price before the budget drains faster than the element
    # recovers, F = (eps_now/eps_dyn)^{1/2} if eps_dyn < eps_now, and F = 1 otherwise (the
    # element does not reach the regime's lower edge before today). On an SSP path E_0 is
    # replaced by the highest annual rate met before the median date of the bound on that
    # path, or over the whole path when the median is not reached; fig_forest_tc.py prints it.
    # The highest rate gives the highest eps_dyn, hence the smallest F: the cautious reading.
    # Only the AMOC's rate is measured (\thetaAMOChat); the Amazon's is the time scale of a
    # vegetation index (Boulton et al. 2022), and the century is an assumption.
    sys.path.insert(0, str(CODE / "01_calibration"))
    import eps_dyn as _ed  # noqa: E402  (the one home of the formula; fig:crossover imports it too)
    e0 = float(LITERATURE_MACROS["EglobalRate"][0])
    emax = {(m.group(1), m.group(2)): float(m.group(3)) for m in re.finditer(
        r"^\s*(AMOC|Amazon|WAIS)\s+(SSPmid|SSPhigh)\s*:\s*E_max =\s*([\d.]+)", forest,
        re.MULTILINE)}
    if len(emax) != 6:
        raise SystemExit("[ERROR] the fig_forest_tc capture lacks the six 'E_max' lines of "
                         "the regime lower edge on each path; rerun run_all.sh.")
    kappa_b = 12.0 / DERIVATION_INPUTS["recovery_months_Amazon_Boulton"][0]
    put("kappaAmazonBoulton", f"{kappa_b:.2f}",
        f"closed form: 12 / (1/kappa in months), 1/kappa="
        f"{DERIVATION_INPUTS['recovery_months_Amazon_Boulton'][0]} months "
        f"[{DERIVATION_INPUTS['recovery_months_Amazon_Boulton'][1]}]; per year")
    t_cent = DERIVATION_INPUTS["recovery_time_century"][0]
    hill_lo = float(LITERATURE_MACROS["relaxWAISHillLo"][0])
    hill_hi = float(LITERATURE_MACROS["relaxWAISHillHi"][0])
    dyn_cases = []   # (suffix, element, theta, theta source, network)
    if lam:
        dyn_cases.append(("AMOC", "AMOC", float(lam),
                          f"theta={float(lam):.4f}/yr measured on the record (ews_calibration.py)", True))
    dyn_cases += [
        ("AmazonBoulton", "Amazon", kappa_b,
         f"theta={kappa_b:.3f}/yr, the vegetation-index time scale of Boulton et al. 2022", False),
        ("AmazonSlow", "Amazon", 1.0 / t_cent,
         f"theta=1/{t_cent:.0f}/yr, assumption [{DERIVATION_INPUTS['recovery_time_century'][1]}]", False),
        ("WAISfast", "WAIS", 1.0 / hill_lo,
         f"theta=1/{hill_lo:.0f}/yr, flux relaxation, Hill et al. 2023 (relaxWAISHillLo)", False),
        ("WAISslow", "WAIS", 1.0 / hill_hi,
         f"theta=1/{hill_hi:.0f}/yr, flux relaxation, Hill et al. 2023 (relaxWAISHillHi)", False),
        ("WAIScentury", "WAIS", 1.0 / t_cent,
         f"theta=1/{t_cent:.0f}/yr, slower modes (Robel et al. 2018), assumption "
         f"[{DERIVATION_INPUTS['recovery_time_century'][1]}]", False),
    ]
    _paths_dyn = {"": ("E_0", None), "SSPmid": ("SSP2-4.5", "SSPmid"),
                  "SSPhigh": ("SSP3-7.0", "SSPhigh")}
    EPS_DYN = {}
    for suf, el, theta, tsrc, net in dyn_cases:
        mk, ek = f"Mstar{el}med", f"eps{el}nat"
        if mk not in C or ek not in C:
            continue
        mstar = float(C[mk]["value"])
        eps_now = float(C[ek]["value"])
        for pkey, (plab, ekey) in _paths_dyn.items():
            e_rate = e0 if ekey is None else emax[(el, ekey)]
            eps_dyn = _ed.eps_dyn(e_rate, mstar, theta, eps_now)
            amp = _ed.amp(eps_dyn, eps_now)
            EPS_DYN[(suf, pkey)] = (eps_dyn, amp, eps_now)
            src = (f"closed form, self-consistent: ((E/(M* theta)) sqrt(eps_now))^(2/3), "
                   f"E={e_rate:g} GtCO2/yr ({plab}"
                   + ("" if ekey is None else ", highest rate before the median date, fig_forest_tc.py")
                   + f"), M*={mstar:.0f} GtCO2 (calibration_ci.py), {tsrc}, "
                   f"eps_now={eps_now}; theta declines as sqrt(eps) along the descent")
            # Two significant figures: E_0 is known to one, theta off a noisy record or a
            # literature range. F to one decimal; F = 1 where eps_dyn >= eps_now.
            put(f"epsDyn{suf}{pkey}", f"{eps_dyn:#.2g}", src, net)
            C[f"epsDyn{suf}{pkey}"]["verbatim"] = True
            put(f"amp{suf}{pkey}", f"{amp:.1f}",
                "closed form: F = (eps_now/eps_dyn)^(1/2) if eps_dyn < eps_now, else 1; "
                "the factor by which the leading law lifts the price between today's "
                "proximity and eps_dyn, the width of the window (the lift of the full price "
                "only if the state is inside the noise layer today); " + src, net)
            C[f"amp{suf}{pkey}"]["verbatim"] = True
    # The words the text writes instead of a number are checked here: the appendix on where the elements sit says the ice
    # sheet's slower modes give a factor of "one", ssec:comparable that the Amazon's factor over a
    # century is "close to one" (the introduction's "barely more than one" was removed at
    # the same bound now carries ssec:comparable alone). (The controls -- eps_dyn AMOC 0.028, F 5.0, Amazon
    # 11, ice sheet 1.24 -- were checked once and are not asserted.)
    assert all(EPS_DYN[("WAIScentury", p)][1] == 1.0 for p in _paths_dyn), \
        "ssec:window says the ice sheet's slower modes give F = one"
    assert EPS_DYN[("AmazonSlow", "")][1] < 1.2, "ssec:comparable says 'close to one' for the century"
    if ("AMOC", "") in EPS_DYN:
        e_a, f_a, _ = EPS_DYN[("AMOC", "")]
        # fig:crossover shades eps < eps_dyn through the same module; the value it prints must be
        # this one, or the figure and \epsDynAMOC describe different things.
        fx = re.search(r"eps_dyn \(AMOC, E_0\)\s*=\s*([\d.]+)", get_out("fig_crossover"))
        if not fx or abs(float(fx.group(1)) - e_a) > 1e-6:
            raise SystemExit("[ERROR] fig:crossover shades a different eps_dyn than \\epsDynAMOC: "
                             f"{fx.group(1) if fx else 'no line'} against {e_a:.6f}")
        put("classFactorAtEpsDynAMOC", f"{e_a ** -0.5:.1f}",
            f"closed form: eps_dyn^(-1/2) for the AMOC at E_0, eps_dyn={e_a:.4f}; the class "
            "factor tied at eps = 1, as \\xoverClassFactorAtCentral", True)
        C["classFactorAtEpsDynAMOC"]["verbatim"] = True

    # ── closed form: where the ADDITIVITY condition of app:additivity bites ──
    # The additive decomposition of the price holds on eps >> E_0/(rho M*). This is NOT the
    # condition bounding the fixed-budget regime, eps_dyn above: different exponent, and the
    # discount rate appears here and not there. Both must hold, so the binding one is the
    # larger, and at the calibration that is this one, by a factor of about four.
    #
    # M* is the PRE-INDUSTRIAL-to-threshold budget of eq:Mstar_coupling, 9158 GtCO2, giving
    # M*/E_0 = 229 yr. Using the REMAINING budget mu_now = eps_now M* = 6411 GtCO2 instead
    # gives 160 yr and hence 0.21 rather than 0.146. The two are easy to confuse: the
    # remaining budget is what draw_Mstar once returned under the name M*.
    # The conclusion is unchanged either way; the numeral is not.
    if "MstarAMOCmed" in C:
        e0 = float(LITERATURE_MACROS["EglobalRate"][0])
        mstar = float(C["MstarAMOCmed"]["value"])
        rho_b = float(LITERATURE_MACROS["rhoBase"][0])
        eps_add = e0 / (rho_b * mstar)
        put("epsAdditivity", f"{eps_add:.2f}",
            f"closed form: E_0/(rho M*), E_0={e0:.0f} GtCO2/yr, rho={rho_b}, "
            f"M*={mstar:.0f} GtCO2 pre-industrial to threshold (calibration_ci.py); "
            "M* is NOT the remaining budget eps_now M*", True)

    # ── the SAME condition, read as the relative error the propagation lemma bounds ──
    # |w_eps|/(L|phi_eps|) = O(E_0/(rho M* eps)). Evaluated at each element's CURRENT
    # proximity this is a number, not an asymptotic regime, and the three numbers differ
    # enough to matter: the split is usable for the AMOC and the Amazon and fails outright
    # for the ice sheet, whose smaller M* and smaller eps_now compound.
    #
    # Only the two usable ones become macros. A relative error above one is not reported as
    # a percentage -- it is reported as a failure, in words, in sec:formula. The WAIS value
    # is asserted here instead, so that the claim "it exceeds one" is CHECKED by the code
    # rather than asserted in prose: if a future calibration moves the ice sheet back inside
    # the condition, this run fails and the sentence gets revisited.
    _add_els = [("AMOC", "MstarAMOCmed", "epsAMOCnat"),
                ("Amazon", "MstarAmazonmed", "epsAmazonnat"),
                ("WAIS", "MstarWAISmed", "epsWAISnat")]
    if all(m in C and e in C for _, m, e in _add_els):
        e0 = float(LITERATURE_MACROS["EglobalRate"][0])
        rho_b = float(LITERATURE_MACROS["rhoBase"][0])
        _rel = {}
        for nm, mk, ek in _add_els:
            _rel[nm] = e0 / (rho_b * float(C[mk]["value"]) * float(C[ek]["value"]))
        assert _rel["WAIS"] > 1.0, (
            f"propagation-lemma relative error for WAIS is {_rel['WAIS']:.3f}, no longer "
            "above one; sec:formula states that the split 'does not hold at all' for that "
            "element and must be rewritten if this assertion fails")
        for nm in ("AMOC", "Amazon"):
            assert _rel[nm] < 1.0, f"{nm} relative error {_rel[nm]:.3f} is no longer below one"
            put(f"additivityErr{nm}", f"{_rel[nm]*100:.0f}",
                f"closed form: E_0/(rho M* eps_now) x 100, the propagation lemma's "
                f"relative error at current proximity; E_0={e0:.0f} GtCO2/yr, rho={rho_b}, "
                f"M*={float(C['Mstar'+nm+'med']['value']):.0f} GtCO2, "
                f"eps_now={C['eps'+nm+'nat']['value']}; WAIS is {_rel['WAIS']*100:.0f}%, "
                "above one, so it carries no macro", True)

    # ── closed form: what the Nordhaus-Stern discount range does to the tipping price ──
    # Impatience reaches the price through TWO channels that pull opposite ways, and the text
    # used to report only the first. Raising rho raises the amplitude Phi_fold, because delay
    # is worth more to a planner who discounts more; and it lowers the realization factor
    # f(rho T) = (1 - exp(-rho T))/(rho T), because a damage that unfolds slowly is discounted
    # harder. eq:fold_scc carries both: A = L f(rho T) Phi_fold / (sigma^{2/3} sqrt(M*)).
    #
    # So "the tipping premium varies by a factor of about five over rho in [1%, 5%]" is true
    # of the AMPLITUDE and not of the price. The price ratio depends on T, and it is the
    # element's realization timescale that decides which channel wins: fast realization keeps
    # the rise, millennial realization cancels it. Four macros rather than one hand-written
    # number, because one number cannot say that.
    sys.path.insert(0, str(CODE / "01_calibration"))
    from phi_fold import phi_true_scalar, realization_factor  # noqa: E402
    if sig:
        s_hat = float(sig)
        rho_lo, rho_hi = 0.01, 0.05
        # the range itself, so the text that quotes the ratios quotes it from here too
        put("rhoRangeLoPct", f"{100 * rho_lo:.0f}", "the Nordhaus-Stern range of the discount "
            "rate over which the \\priceRatioDiscount* ratios are taken, low end (%)")
        put("rhoRangeHiPct", f"{100 * rho_hi:.0f}", "the Nordhaus-Stern range of the discount "
            "rate over which the \\priceRatioDiscount* ratios are taken, high end (%)")
        phi_lo = phi_true_scalar(rho_lo / s_hat ** (2.0 / 3.0))
        phi_hi = phi_true_scalar(rho_hi / s_hat ** (2.0 / 3.0))
        put("phiRatioDiscountRange", f"{phi_hi/phi_lo:.1f}",
            f"closed form: Phi_fold(rho/sigma^(2/3)) at rho=5% over rho=1%, "
            f"sigma={s_hat} (ews_calibration.py). The AMPLITUDE ratio, not the price", True)
        for el, macro in (("AMOC", "priceRatioDiscountAMOC"),
                          ("Amazon", "priceRatioDiscountAmazon"),
                          ("WAIS", "priceRatioDiscountWAIS")):
            T = float(LITERATURE_MACROS[f"T{el}"][0])
            r = (phi_hi * realization_factor(rho_hi, T)) / \
                (phi_lo * realization_factor(rho_lo, T))
            put(macro, f"{r:.2f}",
                f"closed form: A(rho=5%)/A(rho=1%) with A ~ f(rho T) Phi_fold, "
                f"T={T:.0f} yr ({LITERATURE_MACROS[f'T{el}'][1]}), sigma={s_hat}; "
                "the two channels of impatience net out here", True)

    # ── closed form: the bound evaluated at the central inputs, so a reader can check it ──
    # This is eq:tc_bound with no Monte Carlo: (DT* - DT_0)/TCRE x 1000 / E_0. It differs
    # from the reported \tcAMOC, which is the MEDIAN of the bound over the drawn threshold
    # and transient response, and the median of a ratio is not the ratio of the medians.
    # Both are printed so the difference is visible rather than looking like an error.
    tcre_med = DERIVATION_INPUTS["TCRE"][0]
    dts_amoc = LITERATURE_MACROS.get("DTstarAMOC")
    if dts_amoc:
        e0 = float(LITERATURE_MACROS["EglobalRate"][0])
        avail = float(dts_amoc[0]) - dtc
        put("tcAMOCcentral", f"{avail/tcre_med*1000.0/e0:.0f}",
            f"closed form: (DT*-DT_0)/TCRE x 1000 / E_0 at central inputs, "
            f"DT*={dts_amoc[0]}, DT_0={dtc}, TCRE={tcre_med}, E_0={e0:.0f}")
        # The class factor eps^{-1/2} at the CENTRAL AMOC threshold, the counterpart of
        # \xoverClassFactorAtLow (the near end of the range). Two decimals, like that macro.
        eps_c = avail / float(dts_amoc[0])
        put("xoverClassFactorAtCentral", f"{eps_c ** -0.5:.2f}",
            f"closed form: eps_now^(-1/2) at the central AMOC threshold, eps_now=(DT*-DT_0)/DT*, "
            f"DT*={dts_amoc[0]}, DT_0={dtc}; the class factor tied at eps=1, as in crossover.py")

    # ── closed form: the share of the path that falls outside the fixed-budget window ──
    # The band runs from eps_dyn down to the floor eps_inf, and the floor lies inside it:
    # eps_inf is an order of magnitude below eps_dyn, at the median and at the P90. What
    # makes the reported deadline immune is not that the band is absent but that the bound
    # eq:tc_bound does not use the floor. This macro sizes the band as a share of the
    # starting proximity, so the text can state how much of the descent it costs.
    fl = re.search(r"Element: AMOC.*?unrounded: ε_∞ P50=([\d.eE+-]+)", epsi, re.DOTALL)
    if fl and "epsDynAMOC" in C and "epsAMOCnat" in C:
        eps_dyn_v = float(C["epsDynAMOC"]["value"])
        eps_inf_v = float(fl.group(1))
        eps_0_v = float(C["epsAMOCnat"]["value"])
        put("bandFractionPct", f"{100*(eps_dyn_v-eps_inf_v)/eps_0_v:.1f}",
            f"closed form: (eps_dyn - eps_inf)/eps_0 as a percentage, eps_dyn={eps_dyn_v}, "
            f"eps_inf={eps_inf_v:.3e} (eps_inf_ci.py, AMOC c_bar=100 P50), "
            f"eps_0={eps_0_v}", True)

    # ── WITHDRAWN: the SCC LEVEL macros that calibration_ci.py used to feed
    # (\sccPerElem*, \sccTotalRange*, \sccWAISraw, \sccWAISafterF).
    # They quantified the published level, which pairs a noise amplitude in the element's
    # own state variable with a budget in GtCO2. That pairing is dimensionally
    # inconsistent (mu has units of an inverse squared time; a carbon budget does not),
    # so the level it produces is an artefact of the omitted conversion. the AMOC appendix now
    # reports the level through the reduced form (reduced_form_scc.py) as a family indexed
    # by the physical margin. These macros are NOT re-emitted: a withdrawn number must not
    # stay reachable in the macro chain where a later edit could silently cite it again.
    # What survives the withdrawal (the eps^{-1/2} exponent, the noise sign, the
    # cross-partial, the eps-ratios, the t_c ordering) never went through these macros.

    # ── parsed: critical horizon t_c at the low backstop (c_bar=100) ──
    # eps_inf_ci.py prints per element: "AMOC: M*_med=6387  c=100: t_c=111.63yr (...) c=300: ..."
    # We take the c=100 column (the body forest / policy paragraph reports the low backstop),
    # rounded to a whole year, so \tc* match the forest plot. The appendix table this line
    # used to name, a table of dates, no longer exists in the paper: fig:forest_tc replaced it. The
    # generated tc.tex that held it was removed with it; this line is the only trail left,
    # which is why it is written out here rather than left implicit.
    # The paper reports the BOUND t_c <= (M*/E0) eps_0, not the floor-based date: the floor
    # eps_inf = (A/c_bar)^2 inherits the missing margin through A ~ L d_phys^{4/3}, whereas
    # the bound reduces to the remaining warming over TCRE over the emission rate. The bound
    # carries no c_bar, which is why it is parsed from its own line rather than a c=100 cell.
    tc_src = ("eps_inf_ci.py — deadline bound t_c <= (M*/E0) eps_0 = mu_now/E0 "
              "(median calibration, seed=42); free of the backstop cost and of the margin")
    for m in re.finditer(r"^\s*(\w+):\s*t_c_bound=([\d.]+)yr", epsi, re.MULTILINE):
        el, val = m.groups()
        if el in ("AMOC", "Amazon", "WAIS"):
            put(f"tc{el}", str(round(float(val))), tc_src)

    # ── parsed: WAIS immediate-action fraction (eps_inf >= eps_0) at c_bar=100, from the forest ──
    # fig_forest_tc.py prints: "WAIS  : t_c=  4.6 yr [...] immediate(eps_inf>=eps0)=21%"
    fm = re.search(r"WAIS\s*:.*?immediate\(eps_inf>=eps0\)=(\d+)%", forest)
    if fm:
        put("fracWAISimmediate", fm.group(1),
            "fig_forest_tc.py — WAIS draws with eps_inf>=eps_0 at c_bar=100 (seed=42)")

    # ── parsed: reduced-form level (not reported in the paper), the family that replaces the
    # dimensionally inconsistent published level. reduced_form_scc.py prints a
    # "MACROS ... END MACROS" block of "name = value" lines; lambda and Var inside it
    # come from the same cached ews run as \thetaAMOChat and \VarAMOChat, so these
    # macros inherit the network dependency of the HadISST record.
    red_src = ("reduced_form_scc.py — the reduced-form price at the median damage prior "
               "(lambda, Var from ews_calibration.py)")
    forest_src = ("fig_forest_tc.py — share of draws in which the t_c ordering holds "
                  "(seed=42, c_bar=100); the marginal P10/P90 cannot answer this")

    # One macro out of the forest block is not an ordering share and must not inherit the
    # block's source line: \floorShorteningMax is the largest percentage by which the floor
    # shortens the bound eq:tc_bound, across the three elements.
    # Three more are not ordering shares either: they describe the CONDITIONING under which
    # the ordering shares are computed. A draw whose threshold lies below present warming has
    # mu_now = 0, and the price prices a boundary not yet reached, so the model does not
    # describe it and the draw is excluded rather than entered at t_c = 0. These macros say
    # how many such draws there are, what the joint sample is left at, and what the excluded
    # draws would do to the lead if entered — a control on the exclusion, not an alternative.
    forest_overrides = {
        "floorShorteningMax": ("fig_forest_tc.py — largest % by which the floor shortens "
                               "the bound t_c <= mu_now/E0, over the three elements "
                               "(seed=42, c_bar=100)"),
        "fracWAISalreadyPct": ("fig_forest_tc.py — % of WAIS draws with DT* below present "
                               "warming, i.e. mu_now = 0 and the deadline already passed "
                               "(seed=42); these are excluded from the ordering sample"),
        "jointSamplePct":     ("fig_forest_tc.py — % of the N draws in which all three "
                               "elements are still un-crossed, the sample the ordering "
                               "shares are computed on (seed=42)"),
        "fracWAISfirstIncl":  ("fig_forest_tc.py — CONTROL on the exclusion: share in which "
                               "WAIS leads when the already-crossed draws are entered at "
                               "t_c = 0 instead of being excluded (seed=42)"),
        "tcAMOCPten":         ("fig_forest_tc.py — P10 of the deadline bound mu_now/E0 over "
                               "the draws with t_c > 0, the left end of the fig:forest_tc bar "
                               "(seed=42); same sample as the median, no floor"),
        "tcAmazonPten":       ("fig_forest_tc.py — P10 of the deadline bound mu_now/E0 over "
                               "the draws with t_c > 0, the left end of the fig:forest_tc bar "
                               "(seed=42); same sample as the median, no floor"),
        "tcWAISPten":         ("fig_forest_tc.py — P10 of the deadline bound mu_now/E0 over "
                               "the draws with t_c > 0, the left end of the fig:forest_tc bar "
                               "(seed=42); same sample as the median, no floor"),
    }

    # The bound on two emission paths. None of these is an ordering share either.
    _paths = {"SSPmid": "SSP2-4.5", "SSPhigh": "SSP3-7.0"}
    _rcmip = ("RCMIP v5.1.0 (Zenodo 4589756), Emissions|CO2, World, with the extensions of "
              "Meinshausen et al. 2020; extract in 00_data/emissions/")
    forest_overrides["emissionPathStartYear"] = (
        "fig_forest_tc.py via emission_paths.py — T0, the year the emission paths start "
        "from; 2024, the start year the paper adopts (E_0 and DT_0 carry no year of their own)")
    forest_overrides["emissionPathLastYear"] = (
        "fig_forest_tc.py via emission_paths.py — last year of the emission paths; both "
        "are zero from 2250, so not reached by then is not reached at all; " + _rcmip)
    forest_overrides["fracWAISbeforeAmazon"] = (
        "fig_forest_tc.py — share of the draws with both bounds finite (t_c > 0) in which the "
        "ice sheet's bound is no later than the Amazon's (seed=42)")
    forest_overrides["fracWAISbeforeAmazonIncl"] = (
        "fig_forest_tc.py — CONTROL on the exclusion: the same share with the already-crossed "
        "draws entered at t_c = 0 (seed=42)")
    forest_overrides["emissionPathSSPmidZeroYear"] = (
        "fig_forest_tc.py via emission_paths.py — first year from which SSP2-4.5 CO2 emissions "
        "stay at zero in its published extension (Meinshausen et al. 2020); " + _rcmip)
    for _k, _lab in _paths.items():
        forest_overrides[f"EglobalRate{_k}"] = (
            f"fig_forest_tc.py via emission_paths.py — {_lab} CO2 emissions in T0, GtCO2/yr, "
            f"linear between the reported years; " + _rcmip)
        for _el in ("AMOC", "Amazon", "WAIS"):
            for _suf, _what in (("", "median"), ("Pten", "P10"), ("Pninety", "P90")):
                forest_overrides[f"tc{_el}{_k}{_suf}"] = (
                    f"fig_forest_tc.py — {_what} of the deadline bound on {_lab}: years from "
                    f"T0 until emissions summed from T0 reach mu_now, same draws and sample "
                    f"as \\tc{_el} (seed=42); unreached draws count as +inf, a centile among "
                    f"them gets no macro")
            forest_overrides[f"tc{_el}{_k}NotReachedPct"] = (
                f"fig_forest_tc.py — % of the draws with t_c > 0 that {_lab} never brings to "
                f"mu_now before \\emissionPathLastYear (seed=42)")

    def parse_macros(txt, source, network, overrides=None):
        """Read a 'MACROS ... END MACROS' block of 'name = value' lines."""
        blk = re.search(r"^MACROS$(.*?)^END MACROS$", txt, re.MULTILINE | re.DOTALL)
        if not blk:
            return
        # values may carry the manuscript's thousands separator, e.g. 1{,}500
        for name, val in re.findall(r"^\s*(\w+)\s*=\s*([\d.{},]+)\s*$",
                                    blk.group(1), re.MULTILINE):
            put(name, val, (overrides or {}).get(name, source), network)

    parse_macros(reduced, red_src, True)
    parse_macros(forest, forest_src, False, forest_overrides)

    # ── parsed: the upper end of the window of ssec:gen_thm, where dPi/dsigma changes sign.
    # The frontier is located in the logarithm (Riccati + variational), the repository's BVP
    # underflowing well before eta* = 5. The cached ews run supplies lambda for the AMOC
    # point, and read_lambda() sys.exits without it, so these three are network: not one of them
    # exists offline. \etaStarReversal
    # in particular is evaluated AT kappa = 2 rho/lambda and moves with the estimate.
    frontier = get_out("noise_reversal_frontier")
    # \etaStarNow and \etaStarReversalRatio are no longer emitted by the script and so no
    # longer extracted here: both read eta* at the deduced margin d_phys = lambda, a
    # convention in years that the persistence bound excludes physically, and neither has a
    # caller left in paper.tex.
    frontier_overrides = {
        "etaStarReversal": ("noise_reversal_frontier.py — eta* at which dPi/dsigma changes "
                            "sign for the fold, AT THE AMOC CALIBRATION: it is not a "
                            "constant, moving with kappa = 2 rho/lambda, the kappa of the "
                            "AMOC calibration printed in section 6 of the capture"),
        "kappaReversalCeiling": ("noise_reversal_frontier.py — the kappa below which the "
                                 "frontier lies ABOVE the window ceiling eta* = 1, so that "
                                 "the ceiling alone certifies the sign of dPi/dsigma"),
        "reversalLawSlope": ("noise_reversal_frontier.py — slope of eta*_rev^3 in "
                             "log(1/kappa) at the small-kappa end, against the Kramers "
                             "prediction 3/8; the frontier is logarithmic, not a power law"),
    }
    parse_macros(frontier,
                 "noise_reversal_frontier.py — the upper end of the window of ssec:gen_thm "
                 "(ii), located in the logarithm (Riccati + variational equation)",
                 True, frontier_overrides)

    # The two numbers of the cross-partial figure (\crosspartialAsymptote, \crosspartialBpct)
    # were parsed here from paper3_cross_partial_figures.py. The figure and its appendix left
    # the paper and the script left the package, so there is no capture to read.

    # ── closed form: the slope of the amplitude at zero discounting. the derivative of Phi in app:amp at
    # rho_tilde = 0 has u_0 = e^{-eta^3/3} with u_0(0) = 1, so Phi'(0) = 2 int e^{-2 eta^3/3}
    # = 2 (3/2)^{1/3} Gamma(4/3). Exact, not fitted. It was verified against the amplitude
    # solved by crosspartial_mstar.py, which converged to 2.044253 at rho_tilde = 1e-4; that
    # script is no longer part of the package.
    put("PhiPrimeZero", f"{2 * (1.5 ** (1 / 3)) * math.gamma(4 / 3):.3f}",
        "closed form: Phi'(0) = 2 (3/2)^{1/3} Gamma(4/3), the amplitude's slope at zero "
        "discounting; verified against the solved amplitude, which reaches 2.044253 at "
        "rho_tilde = 1e-4")

    # The damage range [0, Lmax] as a flow, 100 rho Lmax % of GDP per year (calibration inputs).
    # Both inputs are the sourced constants of inputs_literature.py, so the number has no
    # second home; whole percent, as \rhoRangeLoPct.
    _lmax, _rho = float(LITERATURE_MACROS["Lmax"][0]), float(LITERATURE_MACROS["rhoBase"][0])
    put("LmaxFlowPct", f"{100 * _rho * _lmax:.0f}",
        f"closed form: 100 rho Lmax, rho={LITERATURE_MACROS['rhoBase'][0]} (rhoBase), "
        f"Lmax={LITERATURE_MACROS['Lmax'][0]} (Lmax), the flow loss at the top of the damage range (%)")

    # -- parsed: bhp_bound.py, the sixteen computed bhp* macros --
    # Lines read "BHP <macro> = <value>". The generator has already chosen each rounding
    # (two of them carry a sign, one is typeset as an exponent), so every value is marked
    # verbatim: match_precision would strip the "+" on the next run, silently, once the
    # macros are in values.tex. The ten SOURCED bhp* macros are not here; they are in
    # inputs_literature.LITERATURE_MACROS, with their source.
    bhp = get_out("bhp_bound")
    bhp_src = ("bhp_bound.py — closed form of eq:ceiling_body under the floored hazard "
               "lambda_1 = h_1T max(0, T - T*), inputs from van2026three's replication "
               "package (Zenodo 20075290)")
    n_bhp = 0
    for m in re.finditer(r"^BHP\s+(bhp\w+)\s*=\s*(\S+)\s*$", bhp, re.MULTILINE):
        put(m.group(1), m.group(2), bhp_src)
        C[m.group(1)]["verbatim"] = True
        n_bhp += 1
    if n_bhp == 0:
        raise SystemExit("[ERROR] bhp_bound.out carries no 'BHP <macro> = <value>' line; "
                         "the sixteen computed bhp* macros would be missing and the run "
                         "would fail later, on their names, in make_tables_values.py.")

    # -- parsed: crossover.py, the five computed xover* macros --
    # Lines read "XOVER <macro> = <value>". Same discipline as the bhp* block above: the
    # generator has already chosen each rounding, and three of the five are typeset inside
    # "a factor of ...", so every value is marked verbatim rather than re-rounded here.
    #
    # These are NOT bhp* macros and the prefix is deliberate. Every bhp* value comes from
    # van2026three's own calibration; these five come from the AMOC threshold range of
    # Armstrong McKay et al. 2022 and the current warming of IPCC AR6, and the exponents
    # this paper proves. One prefix with two provenances is the configuration the
    # manuscript has already been bitten by.
    xover = get_out("crossover")
    xover_src = ("crossover.py — the power law SCC ~ eps^(p-1) read at p=1/2 and p=1 on "
                 "the AMOC threshold range (Armstrong McKay et al. 2022, Table 1) and the "
                 "current warming (Forster et al. 2024); the class factor ties the two prefactors "
                 "at eps = 1")
    n_xover = 0
    for m in re.finditer(r"^XOVER\s+(xover\w+)\s*=\s*(\S+)\s*$", xover, re.MULTILINE):
        put(m.group(1), m.group(2), xover_src)
        C[m.group(1)]["verbatim"] = True
        n_xover += 1
    if n_xover != 6:
        raise SystemExit("[ERROR] crossover.out carries %d 'XOVER <macro> = <value>' lines, "
                         "expected 6. The introduction and the which-parameter-decides "
                         "subsection quote all six; a missing one fails later, in "
                         "make_tables_values.py." % n_xover)

    # -- parsed: record_detectability.py, the rec* macros --
    # Lines read "REC <macro> = <value>". The script reads the outputs of the
    # record-detectability study (01_code/05_record_study/) and recomputes none of them, so
    # each value is the study's, at the rounding the script chose and states. Verbatim for the
    # same reason as the bhp* and xover* blocks: two carry a sign, one is a fraction.
    rec = get_out("record_detectability")
    rec_src = ("record_detectability.py — read from the record-detectability study outputs "
               "in 02_output/record_study (written by 01_code/05_record_study/, or copied "
               "from 00_data/record_study with RECORD_STUDY=cached)")
    n_rec = 0
    for m in re.finditer(r"^REC\s+(rec[A-Za-z]+)\s*=\s*(\S+)\s*$", rec, re.MULTILINE):
        put(m.group(1), m.group(2), rec_src)
        C[m.group(1)]["verbatim"] = True
        n_rec += 1
    if n_rec == 0:
        raise SystemExit("[ERROR] record_detectability.out carries no 'REC <macro> = <value>' "
                         "line; the rec* macros would be missing.")

    # -- Words of the text that rest on a computed number, checked here --
    # Introduction: for the ice sheet "the rise has little room, because the budget soon drains
    # faster than the ice recovers" (it once read "the geometry adds little before the
    # budget outruns it"). Same threshold.
    assert float(C["ampWAISfast"]["value"]) < 1.5, (
        "the introduction says the rise has little room for the ice sheet; \\ampWAISfast = "
        + C["ampWAISfast"]["value"])
    # ssec:comparable: "for a quarter of the threshold values drawn from a distribution fitted to its
    # published range" the ice sheet's threshold already lies below current warming.
    assert 22.5 <= float(C["fracWAISalreadyPct"]["value"]) <= 27.5, (
        "ssec:comparable says 'a quarter'; \\fracWAISalreadyPct = " + C["fracWAISalreadyPct"]["value"])

    # coupled_folds.py (the coupled-elements appendix) left the package: its closed form used the
    # abandoned Airy amplitude and its Monte Carlo contradicted it. that appendix now states the result
    # without numbers, so there is nothing to parse here.

    # -- parsed: validation_bvp.py, the numerical validation of the numerical-methods appendix --
    # Lines read "VALIDATION <macro> = <value>", formatted by the script; verbatim, some carry
    # a sign. The script's tables say what each one is.
    val = get_out("validation_bvp")
    n_val = 0
    for m in re.finditer(r"^VALIDATION\s+(val[A-Za-z]+)\s*=\s*(\S+)\s*$", val, re.MULTILINE):
        # valSens* (section 5) read theta from the ews capture: they need the AMOC series.
        put(m.group(1), m.group(2), "validation_bvp.py — app:solver, BVP validation "
            "(solve_bvp tol 1e-8, domain >= 8 noise scales); see the script's printed tables",
            m.group(1).startswith("valSens")
            or m.group(1) in ("valAmocRhoTilde", "valLawRatioAmoc", "valBtermRatioAmoc"))
        C[m.group(1)]["verbatim"] = True
        n_val += 1
    # \valTcLimitDevPct (the limit of the derivative along x*(eps) against the
    # transcritical constant) is no longer emitted: at fixed state the price tends to zero.
    # \valPitchSlopeFix and \valTcSlopeFix, the slopes of the price at fixed state.
    # Section 6, the leading law at small rho~ (five macros).
    if n_val != 34:
        raise SystemExit("[ERROR] validation_bvp.out carries %d 'VALIDATION' lines, expected 34 "
                         "(25, and the 9 valSens* and valAmoc* that need the ews capture)." % n_val)
    # The amplitude-to-price and numerical-methods appendices: at the overturning circulation's rho~ the effect through B
    # alone would exceed the leading term, yet the exact price stays close to the leading law,
    # because the effect through u offsets most of it.
    assert float(C["valLawRatioAmoc"]["value"]) < 1.3, (
        "app:solver says the law is close at the AMOC's rho~; \\valLawRatioAmoc = " + C["valLawRatioAmoc"]["value"])
    assert float(C["valBtermRatioAmoc"]["value"]) > 1, (
        "the amplitude-to-price expansion says the B effect alone would be large; \\valBtermRatioAmoc = " + C["valBtermRatioAmoc"]["value"])
    # The numerical-methods appendix: "the price at a fixed state falls to zero as the budget runs out" for the
    # transcritical and the pitchfork: the slopes of the price in log eps are positive.
    for nm in ("valPitchSlopeFix", "valTcSlopeFix"):
        if not float(C[nm]["value"]) > 0:
            raise SystemExit("[ERROR] \\%s = %s is not positive: app:solver says the price falls "
                             "to zero." % (nm, C[nm]["value"]))
    # The numerical-methods appendix: "the error falls faster than the square root of the budget": the lower end of the
    # 95% interval of the slope of the log error is above one half.
    if not float(C["valLeadErrSlopeLo"]["value"]) > 0.5:
        raise SystemExit("[ERROR] \\valLeadErrSlopeLo = %s is not above 0.5: app:solver says the "
                         "error falls faster than the square root of the budget."
                         % C["valLeadErrSlopeLo"]["value"])

    # -- parsed: moving_budget.py, the price when the budget is spent --
    # Lines read "MOVING <macro> = <value>", formatted by the script; theta comes from the ews
    # capture through eps_dyn.py, so all eight need the AMOC series.
    mov = get_out("moving_budget")
    n_mov = 0
    for m in re.finditer(r"^MOVING\s+(\w+)\s*=\s*(\S+)\s*$", mov, re.MULTILINE):
        put(m.group(1), m.group(2), "moving_budget.py — the price at fixed state when the budget is "
            "spent at E_0, against the frozen price (AMOC, eta*_now in {0.1, 0.25, 0.5, 1}); "
            "see the script's tables", True)
        C[m.group(1)]["verbatim"] = True
        n_mov += 1
    if n_mov != 8:
        raise SystemExit("[ERROR] moving_budget.out carries %d 'MOVING' lines, expected 8." % n_mov)
    # ssec:which_parameter. The location factor is computed on the budget, and the crossing moved from
    # 0.168 to the last stretch of the window. An earlier assertion, "about a sixth", guarded
    # a sentence since removed; it is replaced by the three claims the new ssec:which_parameter makes.
    _xe, _ed = float(C["xoverEps"]["value"]), float(C["epsDynAMOC"]["value"])
    assert _xe > _ed, (
        "ssec:which_parameter says the crossing is just above the dynamic scale; \\xoverEps = %s, \\epsDynAMOC = %s"
        % (_xe, _ed))
    _cd, _rf = float(C["classFactorAtEpsDynAMOC"]["value"]), float(C["xoverRangeFactor"]["value"])
    assert _cd >= _rf, (
        "ssec:which_parameter says the class catches up near the end of the window; \\classFactorAtEpsDynAMOC = "
        "%s < \\xoverRangeFactor = %s" % (_cd, _rf))
    _cl = float(C["xoverClassFactorAtLow"]["value"])
    assert _cl < _rf, (
        "ssec:which_parameter says the class moves the price less than the whole disagreement at the near end; "
        "\\xoverClassFactorAtLow = %s, \\xoverRangeFactor = %s" % (_cl, _rf))

    # ssec:comparable: "The ice sheet's date barely depends on the path". The three median
    # bounds -- today's rate held, SSP2-4.5, SSP3-7.0 -- may differ by at most 40% of the
    # smallest of them.
    _wais = [float(C[k]["value"]) for k in ("tcWAIS", "tcWAISSSPmid", "tcWAISSSPhigh")]
    assert (max(_wais) - min(_wais)) / min(_wais) <= 0.40, (
        "ssec:comparable says the ice sheet's date barely depends on the path; medians %s" % _wais)

    # the appendix on where the elements sit: further out, "the lift exceeds F many times over" (it read "by orders of magnitude",
    # which F_BVP/F = 34 at eta* = 1.5 did not carry). validation_bvp.py prints F_BVP/F at
    # eta*_now = 1.5 (section 5); the sentence needs it above ten.
    far = re.search(r"^   further out: F_BVP/F at eta\*_now = ([\d.]+) = ([\d.eE+]+)\s*$", val, re.MULTILINE)
    if not far:
        raise SystemExit("[ERROR] validation_bvp.out carries no 'further out' line (section 5).")
    assert float(far.group(2)) > 10, (
        "ssec:window says the lift exceeds F many times over further out; F_BVP/F at eta* = "
        + far.group(1) + " is " + far.group(2))

    # LF on every platform, like every other file the run writes, so that a run on Windows
    # compares byte for byte with the committed file.
    OUTJSON.write_text(json.dumps(C, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    net = sum(1 for v in C.values() if v["requires_network"])
    print(f"Wrote {OUTJSON} : {len(C)} computed macros ({net} require network).")
    for k in sorted(C):
        print(f"  {k:22s} = {C[k]['value']:8s}  {'[net]' if C[k]['requires_network'] else ''}")


if __name__ == "__main__":
    main()
