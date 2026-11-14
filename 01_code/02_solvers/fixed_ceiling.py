r"""
REPLICATION HEADER
  PRODUCES   the MACROS block ("CEIL name = value"), parsed into computed.json by collect_computed.py;
             02_output/planner/fixed_ceiling.json (U1, U2 on the doubled grids and the macro strings), which
             03_figures/fig_ceiling_map.py draws without recomputing anything
  FEEDS      \ceilTauEqMaxInLayer \ceilGSwitchEtaOne \ceilEtaCLo \ceilEtaCHi \ceilEtaCLoCheap
             \ceilEtaCHiCheap \ceilInteriorCount \ceilYearsBeforeTcFLo \ceilYearsBeforeTcFHi
             \ceilEpsOverEpsDynFLo \ceilEpsOverEpsDynFHi (none used by paper.tex yet)
  INPUTS     theta, M*, E_0, eps_now of the AMOC through 01_calibration/eps_dyn.py (the ews and
             calibration captures); rho from inputs_literature.py; the marcher of moving_budget.py
  SEED       none (deterministic)
  RUNTIME    measured and printed at the end (about 10 min on 6 worker processes)
  IMPLEMENTS the planner's problem with a constant unit abatement cost, in the class of fixed ceilings
             (step 53, pas 2 and 2 bis), at order 2 in eps; and the functions the other three
             step-54 scripts share (hjb_stop.py, ceiling_classes.py, hazard_ceiling.py)

fixed_ceiling.py -- "emit at E_0 until the budget reaches eps_bar, then stop for ever": the best eps_bar.

THE PROBLEM (step 53, referee 1). Model units of moving_budget.py (M* = 1, L = 1, mu = eps). The planner
pays gamma E_0 per unit of time while abating and L at the crossing:
    c(x, eps) = inf_z E[ int_0^tau e^{-rho t} gamma E_0 z_t dt + e^{-rho tau} L ],  d eps/dt = -E_0 (1 - z).
g = gamma E_0/(rho L) is the one new parameter. The abatement cost runs until tau (a convention).

THE FIXED CEILING. Start at x0 = x*(eps_now). The emitting phase lasts T = (eps_now - eps_bar)/E_0; after it
the budget is frozen and the value of stopping is c_stop(x, eps_bar) = g + (1 - g) phi_bar(x), phi_bar the
frozen crossing value at eps_bar (moving_budget.steady). Hence
    c(eps_bar) = E[ e^{-rho tau} 1{tau < T} + e^{-rho T} 1{tau >= T} c_stop(X_T, eps_bar) ]
               = g U1(eps_bar) + (1 - g) U2(eps_bar),
U1, U2 the draining problem of moving_budget.py marched from eps_bar to eps_now with data 1 and phi_bar
(march_cols, one column per candidate), read at x0. U1 and U2 do not depend on g, so c is linear in g.
eps_bar = eps_now is "stop today"; eps_bar = 0 is "never stop", the value of moving_budget.solve with
phi(x, 0) = 1 (an assumption, R1.4; ceiling_classes.py varies it).
The best ceiling eps_bar* minimises c on [0] + 200 candidates in [1e-5, eps_now], refined by a parabola in
log eps_bar. A case is "today" (eps_bar* = eps_now), "interior > 5 yr" (t_c - t* > 5 yr) or "< 5 yr"
(within five years of exhaustion, or never stop). t_c - t* = eps_bar* M*/E_0.

THE SCHEME. Order 2 in eps (BDF2, moving_budget.ORDER_TEST is its gate); the z-grid and the eps-levels
are those of step 53. Every number is computed on two grids, (n_z, n_mu) = (1500, 4000) and doubled; the
doubled grids are reported, and the run stops if eps_bar* moves by 5% or more between them.

THE SWEEP. eta*_now in {0.1, 0.25, 0.5, 1} (inside the noise layer) and {1.25, 1.5, 2, 3, F} (outside),
F = (eps_now/eps_dyn)^(1/2); g in {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}. Inside the layer the choice is binary,
stop today or never: g_switch = (phi_A3 - phi_frozen)/(1 - phi_frozen) at x0, and the certainty-
equivalent crossing date tau_eq = -ln(phi)/rho is far below t_c.

THE RULE (macros). The interior ceilings are the cases "interior > 5 yr" with eta*_now >= 1.25 (23 on the
doubled grids, \ceilInteriorCount). eta*(eps_bar*) = d(eps_bar*)/(2 l) is the margin at the ceiling in
widths of the noise layer: its range over all g (\ceilEtaCLo, \ceilEtaCHi) and over g <= 0.1 (\ceilEtaCLoCheap,
\ceilEtaCHiCheap). At eta*_now = F: t_c - t* and eps_bar*/eps_dyn over the six values of g.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moving_budget as mb                               # noqa: E402  (the marcher, the class, the inputs)
from price import price_fixed                            # noqa: E402

ORDER = 2                                 # the scheme in eps of every number this script emits
GRIDS = {"base": (mb.GRID["n_z"], mb.GRID["n_mu"]), "doubled": (2 * mb.GRID["n_z"], 2 * mb.GRID["n_mu"])}
GS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
ETAS_IN = (0.1, 0.25, 0.5, 1.0)
EBAR_MIN = 1e-5                           # smallest interior eps_bar
N_CAND = 200                              # interior candidates for eps_bar (every s-th march level)
YEARS_GAP = 5.0                           # "interior > 5 yr"
PHI_RES = 1e-9                            # below this phi is not resolved by the finite differences
TIE = 1e-6                                # c within TIE of c(0) counts as "never stop" (units of L)
PROM_MIN = 1e-7                           # a local minimum is reported if deeper than this
CONV_TOL = 0.05                           # gate: eps_bar* moves by less than 5% when the grids are doubled
N_WORKERS = max(1, min(6, os.cpu_count() or 1))
PLANNER_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "02_output", "planner")


def write_data(name, obj):
    """The data a figure draws, as JSON in 02_output/planner/ (floats written by repr, so they read back
    exactly; keys sorted, LF line ends: two runs write the same bytes)."""
    import json
    os.makedirs(PLANNER_DIR, exist_ok=True)
    path = os.path.join(PLANNER_DIR, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, indent=1)
        f.write("\n")
    print(f"wrote {os.path.relpath(path, os.path.dirname(PLANNER_DIR))}")


def floats(a):
    return [float(x) for x in np.asarray(a, float).ravel()]


def amoc_F():
    """F = (eps_now/eps_dyn)^(1/2), the largest eta*_now at which the law applies somewhere above eps_dyn."""
    e0, ms, th, en = mb.ed.amoc_inputs()
    return float(mb.ed.amp(mb.ed.eps_dyn(e0, ms, th, en), en))


def etas_out():
    return (1.25, 1.5, 2.0, 3.0, amoc_F())


def ceiling_levels(st, n_mu):
    """The march levels on [EBAR_MIN, eps_now], geometric, with the density per e-fold of the mu-grid of
    moving_budget.solve; the candidate entry levels (every s-th, and eps_now)."""
    e_now = st["eps_now"]
    dens = (n_mu - 1) / np.log(1.12 * e_now / 1e-9)
    K = int(np.ceil(dens * np.log(e_now / EBAR_MIN))) + 1
    mus = np.geomspace(EBAR_MIN, e_now, K)
    s = max(1, (K - 1) // N_CAND)
    starts = list(range(0, K - 1, s)) + [K - 1]
    return mus, starts, s


def run_eta(eta, n_z, n_mu, cls="fold", order=ORDER, with_price=True):
    """U1, U2 on the candidates, the never-stop value and the price along the flow, for one eta*_now."""
    t0 = time.time()
    st, dyn, frz, z = mb.solve(eta, n_z, n_mu, cls, with_z=True, order=order)
    sig, rho, E0, e_now = st["sigma"], st["rho"], st["E0"], st["eps_now"]
    C = mb.CLASSES[cls]
    x0 = C.xs(e_now)
    mus, starts, s = ceiling_levels(st, n_mu)
    init = [np.ones(len(z)) for _ in starts] + [mb.steady(z, mus[k], sig, rho, cls) for k in starts]
    cols = mb.march_cols(z, mus, sig, rho, E0, starts + starts, init, cls, order=order)
    m = len(starts)
    U1 = np.array([mb.at_x(z, cols[:, j], e_now, x0, cls) for j in range(m)])
    U2 = np.array([mb.at_x(z, cols[:, m + j], e_now, x0, cls) for j in range(m)])
    out = dict(st=st, cls=cls, eta=eta, ebar=mus[starts], U1=U1, U2=U2, never=dyn(e_now, x0),
               phi_frozen_now=float(U2[-1]), K=len(mus), s=s, order=order)
    if with_price:   # the shadow price along the flow (emit for ever), units of rho L/E_0
        ef = np.geomspace(e_now, EBAR_MIN, 140)
        out["ef"] = ef
        out["Pd"] = np.array([price_fixed(dyn, e, C.xs(e)) for e in ef]) * st["eps_rho"]
    out["t"] = time.time() - t0
    return out


def _run(args):
    name, eta, cls, order, with_price = args
    n_z, n_mu = GRIDS[name]
    return (name, cls, eta), run_eta(eta, n_z, n_mu, cls, order, with_price)


def compute(etas, classes=("fold",), names=tuple(GRIDS), order=ORDER, with_price=True, workers=N_WORKERS):
    """run_eta on every (grid, class, eta*_now), in parallel; {(grid, class, eta): result}."""
    jobs = [(name, eta, cls, order, with_price) for name in names for cls in classes for eta in etas]
    if workers <= 1:
        return dict(_run(j) for j in jobs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(_run, jobs))


def analyse(r, g, never=None):
    """c on [0, candidates]; global minimum; all local minima (refined by a parabola in log eps_bar).
    never: the value of eps_bar = 0 ("never stop"); default the value with phi(x, 0) = 1."""
    never = r["never"] if never is None else never
    c = g * r["U1"] + (1 - g) * r["U2"]
    e = r["ebar"]
    full_e = np.concatenate([[0.0], e])
    full_c = np.concatenate([[never], c])
    i_star = int(np.argmin(full_c))
    loc = []
    for i in range(len(full_c)):
        lft = full_c[i - 1] if i > 0 else np.inf
        rgt = full_c[i + 1] if i + 1 < len(full_c) else np.inf
        lmax = full_c[:i].max() if i > 0 else np.inf
        rmax = full_c[i + 1:].max() if i + 1 < len(full_c) else np.inf
        prom = min(lmax, rmax) - full_c[i]
        if i == i_star or (full_c[i] < lft and full_c[i] < rgt and prom > PROM_MIN):
            ee = full_e[i]
            if 1 < i < len(full_c) - 1 and full_c[i] < lft and full_c[i] < rgt:   # parabola in log eps_bar
                xs = np.log(full_e[i - 1:i + 2]); ys = full_c[i - 1:i + 2]
                a2, a1, _ = np.polyfit(xs, ys, 2)
                if a2 > 0:
                    ee = float(np.exp(-a1 / (2 * a2)))
            loc.append((ee, full_c[i]))
    e_star = [x for x in loc if x[1] == full_c[i_star]][0][0]
    if full_c[0] - full_c[i_star] < TIE:          # a tie with "never stop" at round-off level
        e_star, i_star = 0.0, 0
    dev = np.abs(c - never) > TIE
    flat = float(r["ebar"][np.argmax(dev)]) if dev.any() else np.nan
    return dict(c=c, c0=never, c0p=c[0], cnow=c[-1], cmin=full_c[i_star], e_star=e_star, flat=flat,
                at_now=(i_star == len(full_c) - 1), at_zero=(i_star == 0), loc=loc)


def years_per_eps():
    e0, mstar, _, _ = mb.ed.amoc_inputs()
    return mstar / e0                                   # years per unit of eps at E_0


def t_c_now(e_now):
    """t_c of the AMOC on the median budget (macro tcAMOCmedBudget), read from eps_dyn.tc_med_budget(),
    the one source moving_budget.setting reads too; checked, to the bit, against e_now M*/E_0 here."""
    t = mb.ed.tc_med_budget()
    if t != e_now * years_per_eps():
        raise SystemExit("[ERROR] t_c: eps_dyn.tc_med_budget() = %r, e_now M*/E_0 of this run = %r"
                         % (t, e_now * years_per_eps()))
    return t


def category(a):
    if a["at_now"]:
        return "today"
    return "interior > 5 yr" if a["e_star"] * years_per_eps() > YEARS_GAP else "< 5 yr"


def eta_at(cls, st, e):
    """eta*(eps) = d(eps)/(2 l), the margin in widths of the noise layer."""
    return mb.CLASSES[cls].margin(e) / (2.0 * st["l"]) if e > 0 else 0.0


def convergence(res, keys, gs=GS):
    """Largest relative change of c_min and of eps_bar* (cases with eps_bar* > 0 on both) from the base to
    the doubled grids."""
    wc = we = 0.0
    for k in keys:
        for g in gs:
            a, b = analyse(res[("doubled",) + k], g), analyse(res[("base",) + k], g)
            wc = max(wc, abs(b["cmin"] / a["cmin"] - 1))
            if a["e_star"] > 0 and b["e_star"] > 0:
                we = max(we, abs(b["e_star"] / a["e_star"] - 1))
    return wc, we


def fmt_sig(x, n=2):
    return mb.fmt_sig(x, n)


def main():
    T0 = time.time()
    mb.order_test()
    F = amoc_F()
    ETAS = ETAS_IN + etas_out()
    res = compute(ETAS)
    for (name, cls, eta), r in sorted(res.items(), key=lambda kv: (kv[0][0] != "base", kv[0][2])):
        print(f"[{name}] eta*_now = {eta:.4g}: {r['K']} march levels on [{EBAR_MIN:g}, eps_now], "
              f"{len(r['ebar'])} eps_bar (every {r['s']} levels), {r['t']:.1f} s")
    R = {eta: res[("doubled", "fold", eta)] for eta in ETAS}
    st1 = R[ETAS[0]]["st"]
    Y = years_per_eps()
    e_now, e_dyn, e_rho = st1["eps_now"], st1["eps_dyn"], st1["eps_rho"]
    t_c = t_c_now(e_now)
    rho_y = float(mb.LITERATURE_MACROS["rhoBase"][0])
    print(f"\nscheme: order {ORDER} in eps; F = {F:.4f}; years per unit of eps = M*/E_0 = {Y:.2f}; t_c = {t_c:.1f} yr; "
          f"eps_dyn = {e_dyn:.4f}; eps_rho = {e_rho:.4f}")

    print("\nINSIDE THE LAYER: tau_eq = -ln(phi)/rho (years) and the switch g_switch = (phi_A3 - phi_frozen)/(1 - phi_frozen)")
    print(f"{'eta*':>5} {'phi_frozen(x0)':>15} {'tau_eq stop':>12} {'phi_A3':>10} {'tau_eq never':>13} {'g_switch':>10} "
          f"{'P max (flow)':>12}")
    sw, teqs = {}, []
    for eta in ETAS_IN:
        r = R[eta]
        pf, pa = r["phi_frozen_now"], r["never"]
        sw[eta] = (pa - pf) / (1 - pf)
        t_stop, t_never = -np.log(pf) / rho_y, -np.log(pa) / rho_y
        teqs += [t_stop, t_never]
        print(f"{eta:5.3g} {pf:15.6g} {t_stop:12.2f} {pa:10.6g} {t_never:13.2f} {sw[eta]:10.4g} {r['Pd'].max():12.4g}")

    print("\nFIXED CEILING: c(eps_bar) in units of L, from x*(eps_now); eps_bar = 0 is 'never stop'")
    print(f"{'eta*':>5} {'g':>5} {'c(0)':>10} {'c(e_now)':>10} {'c_min':>10} {'eps_bar*':>9} {'t*':>6} "
          f"{'t_c-t*':>7} {'e*/e_dyn':>8} {'eta*(e*)':>8} {'category':>16}")
    table = {}
    for eta in ETAS:
        r = R[eta]
        for g in GS:
            a = analyse(r, g)
            cat = category(a)
            tstar = 0.0 if a["at_now"] else (e_now - a["e_star"]) * Y
            ea = eta_at("fold", r["st"], a["e_star"])
            table[(eta, g)] = (a, cat, tstar, ea)
            print(f"{eta:5.3g} {g:5g} {a['c0']:10.6g} {a['cnow']:10.6g} {a['cmin']:10.6g} {a['e_star']:9.4g} "
                  f"{tstar:6.1f} {t_c - tstar:7.1f} {a['e_star'] / e_dyn:8.3g} {ea:8.3g} {cat:>16}")
            if len(a["loc"]) > 1:
                print("        local minima (eps_bar, c): " + ", ".join(f"({e:.3g}, {c:.6g})" for e, c in a["loc"]))
    wc, we = convergence(res, [("fold", eta) for eta in ETAS])
    print(f"\nconvergence (base -> doubled grids): max |change| of c_min {100 * wc:.3g}%, of eps_bar* {100 * we:.3g}%")
    if not we < CONV_TOL:
        raise SystemExit(f"[ERROR] fixed_ceiling.py: eps_bar* moves by {100 * we:.2f}% (>= {100 * CONV_TOL:g}%) "
                         f"when the grids are doubled")
    n_int_in = sum(1 for eta in ETAS_IN for g in GS if table[(eta, g)][1] == "interior > 5 yr")
    print(f"inside the layer (eta* <= 1): {n_int_in} interior ceilings of {len(ETAS_IN) * len(GS)} (the choice is binary)")
    if n_int_in:
        raise SystemExit("[ERROR] fixed_ceiling.py: an interior ceiling inside the noise layer; the binary "
                         "reading of the layer (step 53, pas 2) no longer holds")

    # the rule: eta*(eps_bar*) at the interior ceilings outside the layer, by g
    inter = [(eta, g) for eta in etas_out() for g in GS if table[(eta, g)][1] == "interior > 5 yr"]
    print(f"\nTHE RULE: eta*(eps_bar*) at the {len(inter)} interior ceilings (> 5 yr) of eta*_now >= 1.25")
    for g in GS:
        v = [f"{table[(eta, g)][3]:.3g}" for eta in etas_out() if (eta, g) in inter]
        print(f"  g = {g:<5g}: " + "; ".join(v))
    G_CHEAP = 0.1                                    # the second pair of macros: g <= 0.1
    eta_c = [table[k][3] for k in inter]
    eta_cheap = [table[k][3] for k in inter if k[1] <= G_CHEAP + 1e-12]
    atF = [table[(F, g)] for g in GS if table[(F, g)][1] == "interior > 5 yr"]
    yrsF = [t_c - x[2] for x in atF]
    ratF = [x[0]["e_star"] / e_dyn for x in atF]
    print(f"  range over the {len(inter)} interior ceilings: {min(eta_c):.3f} to {max(eta_c):.3f}; "
          f"over the {len(eta_cheap)} with g <= {G_CHEAP:g}: {min(eta_cheap):.3f} to {max(eta_cheap):.3f}")
    print(f"  at eta*_now = F: t_c - t* from {min(yrsF):.2f} to {max(yrsF):.2f} yr; eps_bar*/eps_dyn from "
          f"{min(ratF):.3f} to {max(ratF):.3f} ({len(atF)} of {len(GS)} values of g interior)")

    print(f"\nruntime: {time.time() - T0:.0f} s ({N_WORKERS} worker processes; per case " +
          ", ".join(f"{k[0][0]}{k[2]:.3g} {v['t']:.0f}" for k, v in sorted(res.items(), key=lambda kv: kv[0][2])) + " s)")
    mac = {"ceilTauEqMaxInLayer": f"{max(teqs):.0f}",
           "ceilGSwitchEtaOne": fmt_sig(sw[1.0]),
           "ceilEtaCLo": f"{min(eta_c):.2f}",
           "ceilEtaCHi": f"{max(eta_c):.2f}",
           "ceilEtaCLoCheap": f"{min(eta_cheap):.2f}",
           "ceilEtaCHiCheap": f"{max(eta_cheap):.2f}",
           "ceilInteriorCount": f"{len(inter)}",
           "ceilYearsBeforeTcFLo": f"{min(yrsF):.0f}",
           "ceilYearsBeforeTcFHi": f"{max(yrsF):.0f}",
           "ceilEpsOverEpsDynFLo": f"{min(ratF):.1f}",
           "ceilEpsOverEpsDynFHi": f"{max(ratF):.1f}"}
    # the data of fig_ceiling_map.py: the doubled grids, from which every number above was read
    write_data("fixed_ceiling.json", {
        "order": ORDER, "grids": "doubled", "GS": list(GS), "ETAS_IN": list(ETAS_IN), "ETAS_OUT": list(etas_out()),
        "F": F, "eps_now": e_now, "eps_dyn": e_dyn, "years_per_eps": Y, "YEARS_GAP": YEARS_GAP,
        "g_switch": {f"{eta:g}": float(sw[eta]) for eta in ETAS_IN},
        "cases": [{"eta": float(eta), "l": float(R[eta]["st"]["l"]), "never": float(R[eta]["never"]),
                   "ebar": floats(R[eta]["ebar"]), "U1": floats(R[eta]["U1"]), "U2": floats(R[eta]["U2"])}
                  for eta in ETAS],
        "macros": mac})
    print("MACROS")
    for k, v in mac.items():
        print(f"CEIL {k} = {v}")
    print("END MACROS")


if __name__ == "__main__":
    main()
