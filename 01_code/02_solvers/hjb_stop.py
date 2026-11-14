r"""
REPLICATION HEADER
  PRODUCES   the MACROS block ("HJB name = value"), parsed into computed.json by collect_computed.py;
             02_output/planner/hjb_stop.json (the stopping curves, the prices along the branch, the stop
             points, and the macro strings), which 03_figures/fig_hjb_stop.py draws without recomputing
  FEEDS      \hjbGainMaxPct \hjbGainMinPct \hjbStopPriceMatchPct \hjbNEpsLevels \hjbOrderGapPosPct (none used
             by paper.tex yet); hjbGainMaxPctOrderOne and hjbGainMinPctOrderOne are diagnostic fields of
             computed.json, never emitted (REMOVED_MACROS)
  INPUTS     as fixed_ceiling.py (the AMOC inputs through eps_dyn.py; rho); imports fixed_ceiling.py and
             the marcher of moving_budget.py
  SEED       none (deterministic)
  RUNTIME    measured and printed at the end (about 6 min on 6 worker processes)
  IMPLEMENTS the full control problem for the fold (step 53, pas 3): the variational inequality marched in
             eps with policy iteration per level, at order 2 in eps, against the best fixed ceiling

hjb_stop.py -- the optimal policy is a trigger on the state; how much it gains over the best fixed ceiling.

THE EQUATIONS (step 53, pas_3.md section 1), model units of moving_budget.py, coordinate z = x - x_-(eps):
  emit:  E_0 dc/d eps|_z = (sigma^2/2) c_zz + [f + E_0 x_-'] c_z - rho c       (the draining operator, L_E)
  stop:  rho c = g rho + (sigma^2/2) c_zz + f c_z                             (the frozen operator, L_0)
  c = 1 at z = 0, c_z = 0 at z = Z, c(x, 0) = 1; at each eps-level the planner takes the smaller of the two
  (min over the policy). The level is solved by policy iteration (Howard): solve the linear system of the
  current policy, switch every node to the branch whose residual is smaller, repeat until the policy no
  longer changes (at most MAX_IT iterations, otherwise the run stops).
THE SCHEME IN eps. Order 2: in the emitting rows the time derivative is the variable-step BDF2 of
  moving_budget.step_bdf2, its first TWO levels (and any level whose step ratio exceeds
  moving_budget.BDF2_W_MAX) by backward Euler; the stopping rows carry no derivative in eps. Two Euler levels,
  not one: the data c(x, 0) = 1 jumps to the stopping value at the first level, and a BDF2 history that
  still holds level 0 adds to the emitting residual a term E_0 a2 (1 - c)/h of order 1e5, which empties the
  stop set and lets policy iteration rebuild it one node per iteration (it did not converge in 60). The levels are step 53's: 0, the
  geometric grid of moving_budget.solve and eps_now itself.
THE COMPARISON. The best fixed ceiling (fixed_ceiling.analyse) is recomputed on the same levels, so the gain
  (c_fixed* - c_HJB)/c_fixed* compares the two policies with the same discretisation.
WHAT IS READ OFF. The level eps_b at which the stable state x*(eps) enters the stop set; the stop curve
  eps_bar(x); the controlled price -dc/dM at x*(eps) (price.price_fixed on the marched c), which equals g
  just before the stop (the rule dc/dM = gamma, in units of rho L/E_0), against the frozen price and the
  price under the flow (emit for ever).
GATES (step 53, now assertions): c_HJB <= the best fixed ceiling (to 1e-9 relative); c non-increasing in
  the state (c_z <= 1e-12) and non-increasing in eps at fixed x (<= 1e-12); eps_b moves by less than 5%
  when the z-grid or the eps-grid is halved.
CASES. Fold, eta*_now in {2, F}, g in {0.02, 0.1, 0.5}; grids MAIN = (3000, 8000), checks n_z / 2 and n_mu / 2.
  MAIN has 8000 geometric eps-levels from 1e-9 to 1.12 eps_now, plus 0 and eps_now (\hjbNEpsLevels = 8000).
THE ORDER-1 DIAGNOSTIC. The HJB and the fixed ceiling are also run at order 1 (backward Euler, the scheme of
  step 53) on MAIN: the gains at order 1 and the largest change of the positions eps_bar*, eps_b between the
  two orders on the interior ceilings (\hjbOrderGapPosPct). The paper quotes order 2 only.
THE PRICE AT THE STOP (\hjbStopPriceMatchPct). The rule dc/dM = gamma, i.e. P = g in units of rho L/E_0, read
  at the stop point itself from the emitting side (stop_price_emit_side): largest |P/g - 1| over the cases
  with a stop. The reading of step 53, at 1.05 eps_b, is printed for comparison.
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import solve_banded

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moving_budget as mb                               # noqa: E402
import fixed_ceiling as fc                               # noqa: E402
from price import price_fixed                            # noqa: E402

ORDER = fc.ORDER
CLS = "fold"
GS = (0.02, 0.1, 0.5)
MAIN = fc.GRIDS["doubled"]
CHECKS = {"x/2": (fc.GRIDS["base"][0], MAIN[1]), "eps/2": (MAIN[0], fc.GRIDS["base"][1])}
MAX_IT = 60
MONO_TOL = 1e-12
GAIN_TOL = 1e-9
CONV_TOL = 0.05


def etas():
    return (2.0, fc.amoc_F())


def levels(st, n_mu):
    e_now = st["eps_now"]
    mus = np.unique(np.concatenate([[0.0], np.geomspace(1e-9, 1.12 * e_now, n_mu), [e_now]]))
    return mus, int(np.searchsorted(mus, e_now))


def hjb(st, z, mus, g, order=ORDER, store=True, k_keep=None):
    """March the variational inequality; c at every level (if store) and at level k_keep, the policy at
    every level, and the number of policy iterations per level."""
    sig, rho, E0 = st["sigma"], st["rho"], st["E0"]
    n = len(z) - 1
    c = np.ones(n + 1)
    c_old = None
    pol = np.zeros(n - 1, dtype=bool)                    # stop at interior nodes 1..n-1
    sol = np.empty((len(mus), n + 1)) if store else None
    P = np.zeros((len(mus), n - 1), dtype=bool)
    if store:
        sol[0] = c
    its = []
    c_keep = None
    for k in range(1, len(mus)):
        h = mus[k] - mus[k - 1]
        if order == 2 and k >= 3 and mb.bdf2_ok(mus, k):   # two Euler levels first (see THE SCHEME)
            w = h / (mus[k - 1] - mus[k - 2])
            a0, a1, a2 = (1 + 2 * w) / (1 + w), 1 + w, w * w / (1 + w)
            hist = E0 / h * (a1 * c[1:n] - a2 * c_old[1:n])     # E_0 (a1 c^k - a2 c^{k-1})/h
        else:
            a0 = 1.0
            hist = E0 / h * c[1:n]
        loE, diE, upE = mb.operator(z, mus[k], sig, rho, E0, CLS)
        lo0, di0, up0 = mb.operator(z, mus[k], sig, rho, 0.0, CLS)
        for it in range(1, MAX_IT + 1):
            di = np.where(pol, -di0, E0 * a0 / h - diE)
            lo = np.where(pol, lo0, loE)
            up = np.where(pol, up0, upE)
            ab = np.zeros((3, n))
            ab[1, :n - 1] = di; ab[0, 1:n] = -up; ab[2, :n - 2] = -lo[1:]
            ab[1, n - 1] = 1.0; ab[2, n - 2] = -1.0
            rhs = np.empty(n)
            rhs[:n - 1] = np.where(pol, g * rho, hist); rhs[n - 1] = 0.0
            rhs[0] += lo[0] * 1.0
            cn = np.concatenate([[1.0], solve_banded((1, 1), ab, rhs)])
            Fe = hist - E0 * a0 / h * cn[1:n] + loE * cn[:-2] + diE * cn[1:-1] + upE * cn[2:]
            Fs = g * rho + lo0 * cn[:-2] + di0 * cn[1:-1] + up0 * cn[2:]
            new = np.where(Fs < Fe, True, np.where(Fe < Fs, False, pol))
            if np.array_equal(new, pol):
                break
            pol = new
        else:
            raise SystemExit(f"[ERROR] hjb_stop.py: policy iteration did not converge at level {k}")
        c_old, c = c, cn
        its.append(it)
        P[k] = pol
        if store:
            sol[k] = c
        if k == k_keep:
            c_keep = c
    return dict(sol=sol, pol=P, its=np.array(its), c_keep=c_keep)


def fixed_on_levels(st, z, mus, k_now, order=ORDER):
    """The best fixed ceiling on the same levels: U1, U2 at (x*(eps_now), eps_now) for eps_bar >= 1e-5."""
    k0 = int(np.searchsorted(mus, fc.EBAR_MIN))
    sub = mus[k0:k_now + 1]
    K = len(sub)
    s = max(1, (K - 1) // fc.N_CAND)
    starts = list(range(0, K - 1, s)) + [K - 1]
    init = [np.ones(len(z)) for _ in starts] + [mb.steady(z, sub[k], st["sigma"], st["rho"], CLS) for k in starts]
    cols = mb.march_cols(z, sub, st["sigma"], st["rho"], st["E0"], starts + starts, init, CLS, order=order)
    m = len(starts)
    x0 = np.sqrt(st["eps_now"])
    U1 = np.array([mb.at_x(z, cols[:, j], st["eps_now"], x0) for j in range(m)])
    U2 = np.array([mb.at_x(z, cols[:, m + j], st["eps_now"], x0) for j in range(m)])
    never = mb.at_x(z, mb.march(z, mus[:k_now + 1], st["sigma"], st["rho"], st["E0"], CLS, keep="last",
                                order=order), st["eps_now"], x0)
    return dict(st=st, ebar=sub[starts], U1=U1, U2=U2, never=never)


def stop_curve(z, mus, P, X):
    """eps_bar(x): the largest level at which the policy is 'stop' at x (nearest interior node)."""
    zi = z[1:-1]
    eb = np.zeros(len(X))
    for k in range(1, len(mus)):
        if not P[k].any():
            continue
        zz = X + np.sqrt(mus[k])
        inside = (zz > zi[0]) & (zz < zi[-1])
        idx = np.clip(np.searchsorted(zi, zz), 1, len(zi) - 1)
        near = np.where(np.abs(zi[idx] - zz) < np.abs(zi[idx - 1] - zz), idx, idx - 1)
        stop = inside & P[k][near]
        eb = np.where(stop, np.maximum(eb, mus[k]), eb)
    return eb


def branch_stop(z, mus, P):
    """The largest level at which the stable state x*(mu) = sqrt(mu) is in the stop set."""
    zi = z[1:-1]
    best = 0.0
    for k in range(1, len(mus)):
        if P[k].any():
            zz = 2.0 * np.sqrt(mus[k])
            if P[k][int(np.argmin(np.abs(zi - zz)))]:
                best = max(best, mus[k])
    return best


def monotonicity(z, mus, sol):
    """max of c_z, and max of c(x, mu_k+1) - c(x, mu_k) at fixed x on sampled levels (both should be <= 0)."""
    vz = float(np.max(np.diff(sol[1:], axis=1)))
    vm = 0.0
    ks = np.unique(np.linspace(1, len(mus) - 2, 200).astype(int))
    X = np.linspace(0.0, z[-1] / 2, 80)
    for k in ks:
        a = np.interp(X + np.sqrt(mus[k]), z, sol[k])
        b = np.interp(X + np.sqrt(mus[k + 1]), z, sol[k + 1])
        vm = max(vm, float(np.max(b - a)))
    return vz, vm


def stop_price_emit_side(st, z, mus, sol, P, eb):
    """The controlled price -dc/dM at the stop point of the branch, eps_b, from the emitting side, in units of
    rho L/E_0. Along the branch, as for the price everywhere else, at the state x_j = x*(mu_j): for the two
    first levels above eps_b, j = k_b+2 and k_b+3, the backward difference at fixed x_j,
    -[c_j(x_j) - c_(j-1)(x_j)]/(mu_j - mu_(j-1)), placed at the midpoint of its interval; the two are extended
    linearly in eps to eps_b. Both levels of each difference must be emitting at x_j (checked): above eps_b
    the stable state emits, and x_j, further from the boundary than x*(mu_(j-1)), emits at level j-1.
    (At fixed x_b itself the levels just above eps_b can still be in the stop set: the trigger.)
    Returns the price and the two differences."""
    kb = int(np.argmin(np.abs(mus - eb)))
    zi = z[1:-1]
    pts = []
    for j in (kb + 2, kb + 3):
        xj = np.sqrt(mus[j])
        for k in (j - 1, j):
            if P[k][int(np.argmin(np.abs(zi - (xj + np.sqrt(mus[k])))))]:
                raise SystemExit(f"[ERROR] hjb_stop.py: level {k} is not emitting at x*(mu_{j})")
        d = -(mb.at_x(z, sol[j], mus[j], xj, CLS) - mb.at_x(z, sol[j - 1], mus[j - 1], xj, CLS)) / (mus[j] - mus[j - 1])
        pts.append((0.5 * (mus[j] + mus[j - 1]), d * st["eps_rho"]))
    (m1, p1), (m2, p2) = pts
    return float(p1 + (p1 - p2) * (eb - m1) / (m1 - m2)), (p1, p2)


def run_case(eta, g, grids, full=True, order=ORDER):
    """The HJB on one grid: c at (x0, eps_now), eps_b, and (if full) everything read off the solution."""
    t0 = time.time()
    st = mb.setting(eta, CLS)
    z = mb.zgrid_of(st, grids[0])
    mus, k_now = levels(st, grids[1])
    H = hjb(st, z, mus, g, order=order, store=full, k_keep=k_now)
    x0 = np.sqrt(st["eps_now"])
    out = dict(eta=eta, g=g, st=st, its=H["its"], eb=branch_stop(z, mus, H["pol"]),
               c_hjb=mb.at_x(z, H["c_keep"], st["eps_now"], x0))
    if not full:
        return out
    sol, P = H["sol"], H["pol"]
    out["vz"], out["vm"] = monotonicity(z, mus, sol)
    X = np.linspace(-0.5 * np.sqrt(st["eps_now"]), np.sqrt(st["eps_now"]) + 4 * st["l"], 241)
    out["X"], out["ebx"] = X, stop_curve(z, mus, P, X)
    eb = out["eb"]
    dyn_c = mb.Dynamic(z, mus, sol, CLS)
    e_now, e_dyn, e_rho = st["eps_now"], st["eps_dyn"], st["eps_rho"]
    lo_e = max(1.05 * eb, 2e-4) if eb > 0 else 2e-4
    EP = np.geomspace(e_now, lo_e, 60)
    out["EP"] = EP
    out["Pc"] = np.array([price_fixed(dyn_c, e, np.sqrt(e)) for e in EP]) * e_rho
    out["Pc_at_stop"] = float(price_fixed(dyn_c, 1.05 * eb, np.sqrt(1.05 * eb)) * e_rho) if eb > 0 else np.nan
    win = (EP >= max(2 * eb, e_dyn)) & (EP <= e_now) & (out["Pc"] > 0)
    out["slope"] = float(np.polyfit(np.log(EP[win]), np.log(out["Pc"][win]), 1)[0]) if win.sum() >= 3 else np.nan
    out["Pc_stop_emit"], out["Pc_stop_pts"] = (stop_price_emit_side(st, z, mus, sol, P, eb) if eb > 0
                                                 else (np.nan, None))
    out["n_any"] = int(sum(1 for k in range(1, len(mus)) if P[k].any()))
    out["n_mixed"] = int(sum(1 for k in range(1, len(mus)) if P[k].any() and not P[k].all()))
    out["t"] = time.time() - t0
    return out


def _job(args):
    kind = args[0]
    if kind == "main":
        _, eta, g, order = args
        return args, run_case(eta, g, MAIN, full=True, order=order)
    if kind == "check":
        _, eta, g, name, order = args
        r = run_case(eta, g, CHECKS[name], full=False, order=order)
        return args, dict(eb=r["eb"], c_hjb=r["c_hjb"])
    if kind == "fixed":
        _, eta, order = args
        st = mb.setting(eta, CLS)
        z = mb.zgrid_of(st, MAIN[0])
        mus, k_now = levels(st, MAIN[1])
        return args, fixed_on_levels(st, z, mus, k_now, order)
    if kind == "flow":
        _, eta, order = args
        st, dyn, frz = mb.solve(eta, MAIN[0], MAIN[1], CLS, order=order)
        e_now = st["eps_now"]
        EP = np.geomspace(e_now, 2e-4, 60)
        Pa = np.array([price_fixed(dyn, e, np.sqrt(e)) for e in EP]) * st["eps_rho"]
        Pf = np.array([price_fixed(frz, e, np.sqrt(e)) for e in EP]) * st["eps_rho"]
        return args, dict(EP=EP, Pa=Pa, Pf=Pf)
    raise ValueError(kind)


def compute(order=ORDER, workers=fc.N_WORKERS, cases=None, light=False):
    """Every job of the run, in parallel: the HJB on MAIN and on the two check grids for each (eta, g),
    the best fixed ceiling on the MAIN levels and the prices under the flow for each eta. light: the HJB
    on MAIN and the fixed ceiling only (the order-1 diagnostic)."""
    E = etas()
    cases = [(e, g) for e in E for g in GS] if cases is None else cases
    ee = sorted({e for e, _ in cases})
    jobs = [("main", e, g, order) for e, g in cases] + [("fixed", e, order) for e in ee]
    if not light:
        jobs += [("check", e, g, n, order) for e, g in cases for n in CHECKS] + [("flow", e, order) for e in ee]
    if workers <= 1:
        out = dict(_job(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = dict(ex.map(_job, jobs))
    res = {}
    for e, g in cases:
        R = out[("main", e, g, order)]
        R["checks"] = {} if light else {n: out[("check", e, g, n, order)] for n in CHECKS}
        R["fixed"] = out[("fixed", e, order)]
        R["flow"] = None if light else out[("flow", e, order)]
        res[(e, g)] = R
    return res


def summarise(res):
    """Per case: the fixed ceiling on the same levels, the gain, the convergence of eps_b and of c."""
    rows = []
    for (eta, g), R in res.items():
        a = fc.analyse(R["fixed"], g)
        gain = (a["cmin"] - R["c_hjb"]) / a["cmin"]
        eb = R["eb"]
        conv = {n: (abs(c["eb"] / eb - 1) if eb > 0 else abs(c["eb"]), abs(c["c_hjb"] / R["c_hjb"] - 1))
                for n, c in R["checks"].items()}
        rows.append(dict(eta=eta, g=g, R=R, a=a, gain=gain, eb=eb, conv=conv))
    return rows


def main():
    T0 = time.time()
    mb.order_test()
    res = compute()
    rows = summarise(res)
    rows1 = summarise(compute(order=1, light=True))      # the order-1 diagnostic (step 53's scheme)
    st0 = rows[0]["R"]["st"]
    Y, e_now, e_dyn, e_rho = fc.years_per_eps(), st0["eps_now"], st0["eps_dyn"], st0["eps_rho"]
    t_c = fc.t_c_now(e_now)
    print(f"fold; scheme order {ORDER} in eps; t_c = {t_c:.1f} yr; eps_dyn = {e_dyn:.4f}; eps_rho = {e_rho:.4f}; "
          f"MAIN grids {MAIN}; checks {CHECKS}")
    fails = []
    for r in rows:
        R, a, eb, g, eta = r["R"], r["a"], r["eb"], r["g"], r["eta"]
        eta_b = np.sqrt(eb) / R["st"]["l"] if eb > 0 else 0.0
        print(f"\n=== eta*_now {eta:.4g}, g {g:g}: HJB {R['t']:.0f} s; policy iterations per level: max "
              f"{R['its'].max()}, mean {R['its'].mean():.2f}")
        print(f"  c_HJB(x*, eps_now) = {R['c_hjb']:.6g}; best fixed ceiling c = {a['cmin']:.6g} at eps_bar* = "
              f"{a['e_star']:.4g} (t_c - t* {(a['e_star'] if not a['at_now'] else e_now) * Y:.1f} yr); "
              f"gain (fixed - HJB)/fixed = {100 * r['gain']:+.3f}%")
        print(f"  branch stop: eps_b = {eb:.4g}, eta* there {eta_b:.3g}, t_c - t* = {eb * Y:.1f} yr; "
              f"eps_bar*/eps_dyn fixed {a['e_star'] / e_dyn:.3g}, HJB {eb / e_dyn:.3g}")
        print(f"  levels with a non-empty stop set: {R['n_any']}; of them mixed: {R['n_mixed']}")
        print(f"  monotonicity: max c_z {R['vz']:.2e}, max c(mu_k+1) - c(mu_k) at fixed x {R['vm']:.2e} (both <= {MONO_TOL:g})")
        if eb > 0:
            print(f"  controlled price at the stop point (x_b, eps_b) from the emitting side: {R['Pc_stop_emit']:.5g} "
                  f"(g = {g:g}, |P/g - 1| = {100 * abs(R['Pc_stop_emit'] / g - 1):.3f}%); one-sided differences on "
                  f"the first two emitting intervals {R['Pc_stop_pts'][0]:.5g}, {R['Pc_stop_pts'][1]:.5g}")
        print(f"  controlled price at x*(eps), units rho L/E0: at eps_now {R['Pc'][0]:.4g}, just above the stop "
              f"(1.05 eps_b) {R['Pc_at_stop']:.4g} (g = {g:g}); frozen {R['flow']['Pf'][0]:.4g}, under the flow "
              f"{R['flow']['Pa'][0]:.4g} at eps_now; log-log slope on [max(2 eps_b, eps_dyn), eps_now] {R['slope']:.3f}")
        print("  convergence: " + "; ".join(f"{n}: |d eps_b| {100 * v[0]:.3g}%, |d c| {100 * v[1]:.3g}%"
                                            for n, v in r["conv"].items()))
        if r["gain"] < -GAIN_TOL:
            fails.append(f"HJB above the fixed ceiling at ({eta:.3g}, {g:g})")
        if R["vz"] > MONO_TOL or R["vm"] > MONO_TOL:
            fails.append(f"monotonicity at ({eta:.3g}, {g:g})")
        if max(v[0] for v in r["conv"].values()) >= CONV_TOL:
            fails.append(f"eps_b not converged at ({eta:.3g}, {g:g})")

    print("\nSUMMARY")
    print(f"{'eta*':>5} {'g':>5} {'c_fixed*':>10} {'c_HJB':>10} {'gain %':>8} {'eps_fix*':>9} {'eps_b HJB':>9} "
          f"{'tc-t* fix':>9} {'tc-t* HJB':>9} {'eta*_b':>7} {'slope':>7} {'conv x':>7} {'conv eps':>8} {'max it':>6}")
    for r in rows:
        a = r["a"]
        print(f"{r['eta']:5.3g} {r['g']:5g} {a['cmin']:10.5g} {r['R']['c_hjb']:10.5g} {100 * r['gain']:8.3f} "
              f"{a['e_star']:9.4g} {r['eb']:9.4g} {a['e_star'] * Y:9.1f} {r['eb'] * Y:9.1f} "
              f"{np.sqrt(r['eb']) / r['R']['st']['l']:7.3f} {r['R']['slope']:7.3f} "
              f"{100 * r['conv']['x/2'][0]:6.2f}% {100 * r['conv']['eps/2'][0]:7.2f}% {r['R']['its'].max():6d}")
    if fails:
        raise SystemExit("[ERROR] hjb_stop.py: " + "; ".join(fails))
    print("gates passed: HJB <= fixed ceiling, monotonicity, convergence of eps_b < 5%")

    inter = [r for r in rows if fc.category(r["a"]) == "interior > 5 yr"]   # the interior fixed ceilings
    one = {(r["eta"], r["g"]): r for r in rows1}
    gmax, gmin = max(r["gain"] for r in rows), min(r["gain"] for r in inter)
    gmax1 = max(r["gain"] for r in rows1)
    gmin1 = min(one[(r["eta"], r["g"])]["gain"] for r in inter)
    match = max(abs(r["R"]["Pc_stop_emit"] / r["g"] - 1) for r in rows if r["eb"] > 0)
    match105 = max(abs(r["R"]["Pc_at_stop"] / r["g"] - 1) for r in rows if r["eb"] > 0)
    # positions, order 1 against order 2, on the interior ceilings: eps_bar* (fixed) and eps_b (HJB); the
    # dates t_c - t* and eps_bar*/eps_dyn move by the same fraction, eta* at the stop by half of it
    gap_pos = max(max(abs(r["a"]["e_star"] / one[(r["eta"], r["g"])]["a"]["e_star"] - 1),
                      abs(r["eb"] / one[(r["eta"], r["g"])]["eb"] - 1)) for r in inter)
    print("\nORDER 1 (diagnostic, the scheme of step 53) against ORDER 2")
    print(f"{'eta*':>5} {'g':>5} {'eps_fix* 1':>11} {'eps_fix* 2':>11} {'eps_b 1':>9} {'eps_b 2':>9} {'c_fix 1':>10} "
          f"{'c_fix 2':>10} {'c_HJB 1':>10} {'c_HJB 2':>10} {'gain 1 %':>9} {'gain 2 %':>9}")
    for r in rows:
        q = one[(r["eta"], r["g"])]
        print(f"{r['eta']:5.3g} {r['g']:5g} {q['a']['e_star']:11.5g} {r['a']['e_star']:11.5g} {q['eb']:9.5g} "
              f"{r['eb']:9.5g} {q['a']['cmin']:10.5g} {r['a']['cmin']:10.5g} {q['R']['c_hjb']:10.5g} "
              f"{r['R']['c_hjb']:10.5g} {100 * q['gain']:9.3f} {100 * r['gain']:9.3f}")
    print(f"largest |order 2 / order 1 - 1| on the positions (eps_bar*, eps_b) of the {len(inter)} interior "
          f"ceilings: {100 * gap_pos:.3f}%")
    print(f"\ngain of the HJB over the best fixed ceiling: largest {100 * gmax:.3f}% (order 1: {100 * gmax1:.3f}%), "
          f"smallest over the {len(inter)} interior ceilings (> 5 yr) {100 * gmin:.3f}% (order 1: {100 * gmin1:.3f}%)")
    print(f"controlled price at the stop point from the emitting side against g: largest |P/g - 1| = "
          f"{100 * match:.3f}%; at 1.05 eps_b (step 53's reading, not a macro): {100 * match105:.2f}%")
    print(f"\nruntime {time.time() - T0:.0f} s ({fc.N_WORKERS} worker processes)")
    mac = {"hjbGainMaxPct": f"{gmax * 100:.2f}",
           "hjbGainMinPct": f"{gmin * 100:.2f}",
           "hjbStopPriceMatchPct": fc.fmt_sig(100 * match),
           "hjbNEpsLevels": f"{MAIN[1]}",
           "hjbOrderGapPosPct": f"{100 * gap_pos:.2f}",
           "hjbGainMaxPctOrderOne": f"{gmax1 * 100:.2f}",
           "hjbGainMinPctOrderOne": f"{gmin1 * 100:.2f}"}
    # the data of fig_hjb_stop.py (order 2, MAIN grids): per case the stopping curve eps_bar(x), the controlled
    # price along the branch and the stop point; per eta*_now the frozen price and the price under the flow
    flows = {}
    for r in rows:
        flows.setdefault(f"{r['eta']:.6g}", {"eta": float(r["eta"]), "EP": fc.floats(r["R"]["flow"]["EP"]),
                                             "Pf": fc.floats(r["R"]["flow"]["Pf"]),
                                             "Pa": fc.floats(r["R"]["flow"]["Pa"])})
    fc.write_data("hjb_stop.json", {
        "order": ORDER, "grids": list(MAIN), "GS": list(GS), "F": fc.amoc_F(), "eps_now": e_now, "eps_dyn": e_dyn,
        "cases": [{"eta": float(r["eta"]), "g": float(r["g"]), "l": float(r["R"]["st"]["l"]),
                   "X": fc.floats(r["R"]["X"]), "ebx": fc.floats(r["R"]["ebx"]), "EP": fc.floats(r["R"]["EP"]),
                   "Pc": fc.floats(r["R"]["Pc"]), "eps_b": float(r["eb"]),
                   "P_stop": float(r["R"]["Pc_stop_emit"]) if r["eb"] > 0 else None,
                   "eps_fixed": float(r["a"]["e_star"])} for r in rows],
        "flows": [flows[k] for k in sorted(flows, key=float)],
        "macros": mac})
    print("MACROS")
    for k, v in mac.items():
        print(f"HJB {k} = {v}")
    print("END MACROS")


if __name__ == "__main__":
    main()
