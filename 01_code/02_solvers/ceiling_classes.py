r"""
REPLICATION HEADER
  PRODUCES   the MACROS block ("CLASS name = value"), parsed into computed.json by collect_computed.py
  FEEDS      \classPitchEtaStop \classTcEtaStop \classFoldEtaStop (none used by paper.tex yet)
  INPUTS     as fixed_ceiling.py; the BVP solver of paper2_all_bifurcations.py for control (d)
  SEED       none (deterministic)
  RUNTIME    measured and printed at the end (about 8 min on 6 worker processes)
  IMPLEMENTS the fixed ceiling for the three classes of Table 1 (step 53, pas 5), at order 2 in eps,
             with the variant of the terminal condition

ceiling_classes.py -- at equal (eta*_now, g), does the pitchfork or the transcritical stop at the same margin
as the fold?

Same policy class and formula as fixed_ceiling.py (run_eta, analyse), with the class of moving_budget.py as a
parameter. THE UNITS (moving_budget.setting): each class recovers today at the measured rate theta, one model
time unit being |f'(x*(eps_now))|/theta years; l = d(eps_now)/(2 eta*_now). This choice is the author's (step
53) and must be stated wherever classes are compared at equal eta*.
CONTROL (d), before anything else: the frozen phi at x*(eps) of the pitchfork and the transcritical by the
finite differences of this package (n_z doubled) against the BVP of the paper's frozen solver
(paper2_all_bifurcations.solve_phi_at_xstar, domain >= 8 layers): within 1% relative or 1e-8 absolute (the
absolute floor was added in step 53 after the 1% control failed at 1.13% on a phi of 3.6e-9; it is <= 1e-4 of
the c_min at stake). The fold is the solver of moving_budget.py, checked there.
THE TERMINAL CONDITION. phi(x, 0) = 1 (crossing certain at exhaustion) is an assumption (R1.4). The variant
replaces the never-stop value by c(0+), the value of stopping at the smallest eps_bar at zero cost
(phi(x, 0+) = frozen phi): it moves "never stop", never an interior ceiling (step 53).
GATE: eps_bar* moves by less than 5% when the grids are doubled (fixed_ceiling.CONV_TOL).
CASES. Fold, pitchfork, transcritical; eta*_now in {2, F}; g in {0.02, 0.1, 0.5}; base and doubled grids.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moving_budget as mb                               # noqa: E402
import fixed_ceiling as fc                               # noqa: E402
import paper2_all_bifurcations as p2                     # noqa: E402

ORDER = fc.ORDER
CLS = ("fold", "pitchfork", "transcritical")
GS = (0.02, 0.1, 0.5)
TOL_BVP = 0.01
ABS_BVP = 1e-8
G_MACRO = 0.1           # the macros: margin at the optimal ceiling at eta*_now = 2, g = 0.1
ETA_MACRO = 2.0


def etas():
    return (2.0, fc.amoc_F())


def control_d():
    """Frozen phi at x*(eps), finite differences against the BVP, pitchfork and transcritical."""
    worst_rel, worst_abs, n, fails = 0.0, 0.0, 0, []
    print("control d: pitchfork and transcritical, frozen phi at x*(eps), FD (n_z doubled) against the BVP")
    for cls in ("pitchfork", "transcritical"):
        C = mb.CLASSES[cls]
        for eta in etas():
            st = mb.setting(eta, cls)
            z = mb.zgrid_of(st, fc.GRIDS["doubled"][0])
            for e in (st["eps_now"], 0.3, 0.1, 0.03, 0.01):
                x = C.xs(e)
                fd = mb.at_x(z, mb.steady(z, e, st["sigma"], st["rho"], cls), e, x, cls)
                if fd < fc.PHI_RES:
                    continue
                absr = max(5.0, 8.0 * st["l"] + C.margin(e))
                bvp = p2.solve_phi_at_xstar(e, st["sigma"], st["rho"], C.drift, 0.0, lambda m: x, 6.0, absr,
                                            tol=1e-8, max_nodes=200000)
                r_, a_ = abs(fd / bvp - 1), abs(fd - bvp)
                ok = r_ < TOL_BVP or a_ < ABS_BVP
                n += 1
                worst_rel, worst_abs = max(worst_rel, r_), max(worst_abs, a_)
                if not ok:
                    fails.append((cls, eta, e))
                print(f"    {'OK ' if ok else 'OUT'} {cls:13s} eta* {eta:5.3g} eps {e:6.3g}: FD {fd:.6g}  BVP {bvp:.6g}  "
                      f"rel {100 * r_:.3g}%  abs {a_:.2g}")
    if fails:
        raise SystemExit(f"[ERROR] ceiling_classes.py: control d fails at {fails}")
    print(f"    {n} points, all within {100 * TOL_BVP:g}% or {ABS_BVP:g}")


def compute(order=ORDER, names=tuple(fc.GRIDS), workers=fc.N_WORKERS):
    return fc.compute(etas(), classes=CLS, names=names, order=order, with_price=False, workers=workers)


def tables(res):
    """The per-class table, on the doubled grids: {(cls, eta, g): (analysis, t*, eta*(eps_bar*))}."""
    Y = fc.years_per_eps()
    out = {}
    for cls in CLS:
        for eta in etas():
            r = res[("doubled", cls, eta)]
            for g in GS:
                a = fc.analyse(r, g)
                tstar = 0.0 if a["at_now"] else (r["st"]["eps_now"] - a["e_star"]) * Y
                out[(cls, eta, g)] = (a, tstar, fc.eta_at(cls, r["st"], a["e_star"]))
    return out


def main():
    T0 = time.time()
    mb.order_test()
    control_d()
    res = compute()
    R = {k[1:]: v for k, v in res.items() if k[0] == "doubled"}
    E = etas()
    Y = fc.years_per_eps()
    e_now = R[("fold", E[0])]["st"]["eps_now"]
    t_c = fc.t_c_now(e_now)
    rho_y = float(mb.LITERATURE_MACROS["rhoBase"][0])
    print(f"\nscheme order {ORDER} in eps; t_c = {t_c:.1f} yr; years per unit of eps {Y:.2f}")
    print("\nSETTINGS (model units): l, sigma, years per model time unit, eps_dyn of the class")
    for cls in CLS:
        for eta in E:
            st = R[(cls, eta)]["st"]
            print(f"  {cls:13s} eta*_now {eta:5.3g}: l {st['l']:.4f}, sigma {st['sigma']:.4g}, rho_m {st['rho']:.4g}, "
                  f"E0_m {st['E0']:.4g}, yr/unit {st['tconv']:.3f}, eps_dyn {st['eps_dyn']:.4f}")
    print("\nCROSSING DATE tau_eq = -ln(phi)/rho (yr): stop today (frozen phi) and never stop")
    for cls in CLS:
        for eta in E:
            r = R[(cls, eta)]
            teq = lambda ph: f"{-np.log(ph) / rho_y:9.1f}" if ph > fc.PHI_RES else f"{'>691':>9}"
            print(f"  {cls:13s} {eta:5.3g}: phi_frozen {r['phi_frozen_now']:11.4g} {teq(r['phi_frozen_now'])}   "
                  f"phi_never {r['never']:11.4g} {teq(r['never'])}")

    T = tables(res)
    print("\nFIXED CEILING by class (doubled grids); c in units of L; eta*(e*) = d(e*)/(2 l)")
    print(f"{'class':13s} {'eta*':>5} {'g':>5} {'c(0)':>10} {'c(e_now)':>10} {'c_min':>10} {'eps_bar*':>9} "
          f"{'e*/e_dyn':>8} {'eta*(e*)':>8} {'t*':>6} {'t_c-t*':>7} {'category':>16}")
    for cls in CLS:
        for eta in E:
            st = R[(cls, eta)]["st"]
            for g in GS:
                a, tstar, ea = T[(cls, eta, g)]
                print(f"{cls:13s} {eta:5.3g} {g:5g} {a['c0']:10.5g} {a['cnow']:10.5g} {a['cmin']:10.5g} "
                      f"{a['e_star']:9.4g} {a['e_star'] / st['eps_dyn']:8.3g} {ea:8.3g} {tstar:6.1f} {t_c - tstar:7.1f} "
                      f"{fc.category(a):>16}")
                if len(a["loc"]) > 1:
                    print("        local minima (eps_bar, c): " + ", ".join(f"({e:.3g}, {c:.6g})" for e, c in a["loc"]))
    wc, we = fc.convergence(res, [(cls, eta) for cls in CLS for eta in E], GS)
    print(f"convergence (base -> doubled): max |change| of c_min {100 * wc:.3g}%, of eps_bar* {100 * we:.3g}%")
    if not we < fc.CONV_TOL:
        raise SystemExit(f"[ERROR] ceiling_classes.py: eps_bar* moves by {100 * we:.2f}% when the grids are doubled")

    print("\nSIDE BY SIDE at equal (eta*_now, g): eps_bar*, eta*(eps_bar*), t_c - t* (yr)")
    print(f"{'eta*':>5} {'g':>5} | " + " | ".join(f"{c:^28s}" for c in CLS))
    for eta in E:
        for g in GS:
            cells = [f"{T[(c, eta, g)][0]['e_star']:8.4g} {T[(c, eta, g)][2]:7.3g} {t_c - T[(c, eta, g)][1]:8.1f}   "
                     for c in CLS]
            print(f"{eta:5.3g} {g:5g} | " + " | ".join(cells))

    print("\nTERMINAL CONDITION: phi(x, 0) = 1 (default) against c(0+) = frozen phi at the smallest eps_bar")
    print(f"{'class':13s} {'eta*':>5} {'g':>5} {'c(0) phi=1':>11} {'c(0+)':>10} {'eps_bar* (phi=1)':>17} "
          f"{'eps_bar* (c(0+))':>17} {'category (phi=1)':>17} {'category (c(0+))':>17}")
    moved_interior = 0
    for cls in CLS:
        for eta in E:
            r = R[(cls, eta)]
            alt = float(r["U2"][0])
            for g in GS:
                a, b = fc.analyse(r, g), fc.analyse(r, g, never=alt)
                if fc.category(a) == "interior > 5 yr" and abs(a["e_star"] - b["e_star"]) > 0:
                    moved_interior += 1
                print(f"{cls:13s} {eta:5.3g} {g:5g} {r['never']:11.5g} {alt:10.5g} {a['e_star']:17.4g} "
                      f"{b['e_star']:17.4g} {fc.category(a):>17} {fc.category(b):>17}")
    print(f"interior ceilings (> 5 yr) moved by the terminal condition: {moved_interior}")

    em = {c: T[(c, ETA_MACRO, G_MACRO)][2] for c in CLS}
    print(f"\nmargin at the optimal ceiling, eta*_now = {ETA_MACRO:g}, g = {G_MACRO:g}: "
          + ", ".join(f"{c} {v:.3f}" for c, v in em.items()))
    print(f"\nruntime: {time.time() - T0:.0f} s ({fc.N_WORKERS} worker processes; per case "
          + ", ".join(f"{k[0][0]}:{k[1][:2]}{k[2]:.3g} {v['t']:.0f}" for k, v in res.items()) + " s)")
    print("MACROS")
    print(f"CLASS classPitchEtaStop = {em['pitchfork']:.2f}")
    print(f"CLASS classTcEtaStop = {em['transcritical']:.2f}")
    print(f"CLASS classFoldEtaStop = {em['fold']:.2f}")
    print("END MACROS")


if __name__ == "__main__":
    main()
