r"""
REPLICATION HEADER
  PRODUCES   the MACROS block ("HAZARD name = value"), parsed into computed.json by collect_computed.py
  FEEDS      \hazardCeilGapMaxYears \hazardCeilGapMaxPct \hazardCeilCases \hazardEarlyLoYears
             \hazardEarlyHiYears \hazardMissCases \hazardFollowCostQSPct \hazardFollowCostFoldPct
             \hazardFollowCostMaxPct (none used by paper.tex yet)
  INPUTS     as fixed_ceiling.py; imports ceiling_classes.py for the true model (recomputed, no cache)
  SEED       none (deterministic)
  RUNTIME    measured and printed at the end (about 5 min on 6 worker processes)
  IMPLEMENTS the fixed ceiling in a hazard model calibrated on the path (step 53, pas 5 bis), against the
             true model at order 2 in eps; and the cost, in the true model, of following the hazard's ceiling

hazard_ceiling.py -- what a hazard can price: the ceiling, if it is calibrated on the path.

THE HAZARD. lambda(eps) = rho phi(eps)/(1 - phi(eps)), phi(eps) the frozen crossing value at the stable state
x*(eps) (moving_budget.steady on the z-grid of the class, n_z doubled). It is CONSTRUCTED from the geometry,
not estimated: the hazards in use (Cai et al. 2019, linear in temperature above a floor; van den Bremer et al.
2026) are not calibrated this way. With I(eps) = (1/E_0) int_eps^eps_now (rho + lambda) du,
  c_hazard(eps_bar) = int_{eps_bar}^{eps_now} (lambda/E_0) e^{-I} du + e^{-I(eps_bar)} [g + (1 - g) phi(eps_bar)]
                    = g U1h + (1 - g) U2h,   U1h = J + e^{-I},  U2h = J + e^{-I} phi,
so fixed_ceiling.analyse applies unchanged. Quadrature: trapezoid in log u on N_QUAD geometric points of
[1e-5, eps_now], repeated with 2 N_QUAD (convergence printed). eps_bar = 0: J(1e-5) + e^{-I(1e-5)}.
CONTROLS (step 53): (1) identity lambda/(rho + lambda) = phi to 1e-12; (2) anchor: c_hazard(eps_now) =
c_true(eps_now) (stop today: both are g + (1 - g) phi_frozen) to 1e-12.
THE COMPARISON with the true model of ceiling_classes.py (doubled grids, order 2), for the three classes,
eta*_now in {2, F}, g in {0.02, 0.1, 0.5}. "Quasi-static" cases: the true optimum is interior and more than 5
years before t_c; there the hazard's ceiling is compared through t_c - t*. Near exhaustion (all other cases):
the hazard "misses" when its t_c - t* is not within 10% of the true one (or, if the true one is below 1 year,
when its own is not). The advance (t_c - t*)_hazard - (t_c - t*)_true of the misses is reported.
THE COST OF FOLLOWING THE HAZARD (step 54, addendum A.1; three macros: the largest over the quasi-static cases,
over the six fold cases, and over all cases): c_true(eps_bar_hazard)/c_true(eps_bar*_true) - 1,
both marched exactly in the true model from their own ceiling (the level is inserted in the march grid; a
level the hazard puts at "never stop" or "today" takes the true value of that end).
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moving_budget as mb                               # noqa: E402
import fixed_ceiling as fc                               # noqa: E402
import ceiling_classes as cc                             # noqa: E402

ORDER = fc.ORDER
N_QUAD = 4000
N_Z = fc.GRIDS["doubled"][0]
TOL_REL_GAP = 0.10                                       # a miss: not within 10% of t_c - t*
END_YEARS = 1.0                                          # "end of the budget": t_c - t* < 1 yr
CTRL_TOL = 1e-12


def hazard_case(cls, eta, n):
    st = mb.setting(eta, cls)
    z = mb.zgrid_of(st, N_Z)
    xs = mb.CLASSES[cls].xs
    E = np.geomspace(fc.EBAR_MIN, st["eps_now"], n)
    phi = np.array([mb.at_x(z, mb.steady(z, e, st["sigma"], st["rho"], cls), e, xs(e), cls) for e in E])
    one_m = 1.0 - phi
    n_clip = int(np.sum(one_m <= 0))
    lam = np.where(one_m > 0, st["rho"] * phi / np.where(one_m > 0, one_m, 1.0), 1e300)
    ident = float(np.max(np.abs(np.where(one_m > 0, lam / (st["rho"] + lam), phi) - phi)))
    lu = np.log(E)
    a = (st["rho"] + lam) / st["E0"] * E
    seg = 0.5 * (a[1:] + a[:-1]) * np.diff(lu)
    I = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    b = lam / st["E0"] * np.exp(-I) * E
    segb = 0.5 * (b[1:] + b[:-1]) * np.diff(lu)
    J = np.concatenate([np.cumsum(segb[::-1])[::-1], [0.0]])
    eI = np.exp(-I)
    return dict(st=st, cls=cls, eta=eta, ebar=E, U1=J + eI, U2=J + eI * phi, never=float(J[0] + eI[0]), phi=phi,
                lam=lam, n_clip=n_clip, ident=ident)


def c_true_at(cls, eta, ebar, g, order=ORDER):
    """c in the true model for the fixed ceiling ebar, marched exactly from ebar (inserted as a level) on
    the doubled grids; ebar = 0 is the never-stop value, ebar = eps_now stopping today."""
    n_z, n_mu = fc.GRIDS["doubled"]
    if ebar <= 0:
        st, dyn, frz = mb.solve(eta, n_z, n_mu, cls, order=order)
        return float(dyn(st["eps_now"], mb.CLASSES[cls].xs(st["eps_now"])))
    st = mb.setting(eta, cls)
    z = mb.zgrid_of(st, n_z)
    x0 = mb.CLASSES[cls].xs(st["eps_now"])
    mus, _, _ = fc.ceiling_levels(st, n_mu)
    mus = np.unique(np.concatenate([mus[mus > ebar], [ebar]]))
    pb = mb.steady(z, ebar, st["sigma"], st["rho"], cls)
    cols = mb.march(z, mus, st["sigma"], st["rho"], st["E0"], cls, init=np.column_stack([np.ones(len(z)), pb]),
                    keep="last", order=order)
    U1, U2 = (mb.at_x(z, cols[:, j], st["eps_now"], x0, cls) for j in (0, 1))
    return g * U1 + (1 - g) * U2


def _cost_job(args):
    return args, c_true_at(*args)


def main():
    T0 = time.time()
    mb.order_test()
    true = {k[1:]: v for k, v in cc.compute(names=("doubled",)).items()}
    H = {(cls, eta): hazard_case(cls, eta, N_QUAD) for cls in cc.CLS for eta in cc.etas()}
    H2 = {(cls, eta): hazard_case(cls, eta, 2 * N_QUAD) for cls in cc.CLS for eta in cc.etas()}
    for k, h in H.items():
        print(f"{k[0]:13s} eta*_now {k[1]:5.3g}: phi >= 1 at {h['n_clip']} of {N_QUAD} points (lambda set to 1e300 there)")

    ident = max(h["ident"] for h in H.values())
    anchor = max(abs(fc.analyse(H[k], g)["cnow"] - fc.analyse(true[k], g)["cnow"]) for k in H for g in cc.GS)
    print(f"\ncontrol 1 (identity): max |lambda/(rho+lambda) - phi| = {ident:.2e}")
    print(f"control 2 (anchor): max |c_hazard(eps_now) - c_true(eps_now)| over the {len(H) * len(cc.GS)} cases = {anchor:.2e}")
    if ident > CTRL_TOL or anchor > CTRL_TOL:
        raise SystemExit("[ERROR] hazard_ceiling.py: a control of the hazard construction fails")

    st0 = true[("fold", cc.etas()[0])]["st"]
    Y, e_now = fc.years_per_eps(), st0["eps_now"]
    t_c = fc.t_c_now(e_now)
    print("\nTHE CALIBRATION PATH: lambda (per year) at eps_now, at eps = 0.1 and at eps = 1e-3")
    for (cls, eta), h in H.items():
        py = h["lam"] / h["st"]["tconv"]
        print(f"  {cls:13s} eta* {eta:5.3g}: lambda(eps_now) {py[-1]:.3g}/yr, at eps 0.1 "
              f"{py[np.argmin(np.abs(h['ebar'] - 0.1))]:.3g}/yr, at 1e-3 {py[np.argmin(np.abs(h['ebar'] - 1e-3))]:.3g}/yr")

    rows = {}
    for cls in cc.CLS:
        for eta in cc.etas():
            for g in cc.GS:
                at, ah = fc.analyse(true[(cls, eta)], g), fc.analyse(H[(cls, eta)], g)
                ah2 = fc.analyse(H2[(cls, eta)], g)
                gt = t_c if at["at_now"] else at["e_star"] * Y          # t_c - t*
                gh = t_c if ah["at_now"] else ah["e_star"] * Y
                quasi = (not at["at_now"]) and at["e_star"] > 0 and gt > fc.YEARS_GAP
                if gt < END_YEARS:
                    ok, rel = gh < END_YEARS, np.nan
                else:
                    rel = (gh - gt) / gt
                    ok = abs(rel) < TOL_REL_GAP
                eb_h = e_now if ah["at_now"] else ah["e_star"]
                eb_t = e_now if at["at_now"] else at["e_star"]
                rows[(cls, eta, g)] = dict(at=at, ah=ah, ah2=ah2, gt=gt, gh=gh, gap=gh - gt, rel=rel, ok=ok,
                                           quasi=quasi, eb_h=eb_h, eb_t=eb_t)
    # the cost of following the hazard: both ends marched exactly in the true model
    jobs = sorted({(k[0], k[1], r["eb_h"], k[2]) for k, r in rows.items()} |
                  {(k[0], k[1], r["eb_t"], k[2]) for k, r in rows.items()})
    with ProcessPoolExecutor(max_workers=fc.N_WORKERS) as ex:
        cost = dict(ex.map(_cost_job, jobs))
    for k, r in rows.items():
        ch, ct = cost[(k[0], k[1], r["eb_h"], k[2])], cost[(k[0], k[1], r["eb_t"], k[2])]
        r["follow"] = ch / ct - 1

    print(f"\nSIDE BY SIDE (t_c = {t_c:.1f} yr): true (order {ORDER}) against hazard; gap = (t_c-t*)_hazard - "
          f"(t_c-t*)_true; follow = c_true(eps_bar_hazard)/c_true(eps_bar*_true) - 1")
    print(f"{'class':13s} {'eta*':>5} {'g':>5} | {'e* true':>9} {'eta* t':>7} {'tc-t* t':>8} | {'e* haz':>9} "
          f"{'eta* h':>7} {'tc-t* h':>8} | {'gap yr':>7} {'rel':>7} {'within':>6} {'regime':>8} | {'c_min t':>9} "
          f"{'c_min h':>9} {'follow %':>9} {'e* haz 2N':>9}")
    for (cls, eta, g), r in rows.items():
        st = true[(cls, eta)]["st"]
        at, ah = r["at"], r["ah"]
        print(f"{cls:13s} {eta:5.3g} {g:5g} | {at['e_star']:9.4g} {fc.eta_at(cls, st, at['e_star']):7.3g} {r['gt']:8.1f} | "
              f"{ah['e_star']:9.4g} {fc.eta_at(cls, st, ah['e_star']):7.3g} {r['gh']:8.1f} | {r['gap']:+7.1f} "
              f"{r['rel']:+7.3f} {'yes' if r['ok'] else 'NO':>6} {'quasi' if r['quasi'] else 'end':>8} | "
              f"{at['cmin']:9.4g} {ah['cmin']:9.4g} {100 * r['follow']:9.3f} {r['ah2']['e_star']:9.4g}")
    conv = max(abs(rows[k]["ah2"]["e_star"] / rows[k]["ah"]["e_star"] - 1) for k in rows if rows[k]["ah"]["e_star"] > 0)
    print(f"\nconvergence of the quadrature (N = {N_QUAD} -> {2 * N_QUAD}): max |change| of eps_bar* {100 * conv:.3g}%")

    Q = [r for r in rows.values() if r["quasi"]]
    NQ = [r for r in rows.values() if not r["quasi"]]
    miss = [r for r in NQ if not r["ok"]]
    q_bad = [k for k, r in rows.items() if r["quasi"] and not r["ok"]]
    gap_y = max(abs(r["gap"]) for r in Q)
    gap_p = max(abs(r["rel"]) for r in Q)
    early = [r["gap"] for r in miss]
    print(f"quasi-static cases: {len(Q)} (" + ", ".join(f"{c} {sum(1 for k, r in rows.items() if r['quasi'] and k[0] == c)}"
                                                        for c in cc.CLS) + f"); within 10%: {len(Q) - len(q_bad)}; "
          f"largest |gap| {gap_y:.2f} yr, {100 * gap_p:.2f}% of t_c - t*")
    print(f"near exhaustion: {len(NQ)} cases, missed in {len(miss)}; advance of the misses (hazard earlier > 0): "
          + ", ".join(f"{x:+.1f}" for x in sorted(early)) + " yr")
    if any(x <= 0 for x in early):
        print("  NOTE: a miss in which the hazard stops LATER than the optimum")
    fol_q = max(r["follow"] for r in Q)
    fol_fold = max(r["follow"] for k, r in rows.items() if k[0] == "fold")
    k_max = max(rows, key=lambda k: rows[k]["follow"])
    fol_max = rows[k_max]["follow"]
    print(f"cost of following the hazard's ceiling in the true model: largest {100 * fol_max:.3f}% (all classes; "
          f"{k_max[0]}, eta*_now {k_max[1]:.3g}, g {k_max[2]:g}), {100 * fol_fold:.4f}% for the fold (its six cases), "
          f"{100 * fol_q:.4f}% over the {len(Q)} quasi-static cases")
    print(f"\nruntime {time.time() - T0:.0f} s ({fc.N_WORKERS} worker processes)")
    print("MACROS")
    print(f"HAZARD hazardCeilGapMaxYears = {gap_y:.1f}")
    print(f"HAZARD hazardCeilGapMaxPct = {100 * gap_p:.0f}")
    print(f"HAZARD hazardCeilCases = {len(Q)}")
    print(f"HAZARD hazardMissCases = {len(miss)}")
    print(f"HAZARD hazardEarlyLoYears = {min(early):.1f}")
    print(f"HAZARD hazardEarlyHiYears = {max(early):.1f}")
    print(f"HAZARD hazardFollowCostQSPct = {100 * fol_q:.2f}")
    print(f"HAZARD hazardFollowCostFoldPct = {100 * fol_fold:.2f}")
    print(f"HAZARD hazardFollowCostMaxPct = {100 * fol_max:.0f}")
    print("END MACROS")


if __name__ == "__main__":
    main()
