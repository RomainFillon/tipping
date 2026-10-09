#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_inventory.py — write the generated sections of README.md: the title, the map from each
exhibit of the paper to the code that produces it, the list of programs, the macros of
values.tex by producing script, and the counts the data availability statement and the notes
quote.

  COMPUTES   nothing; it reads what the paper displays and the run's outputs
  READS      what the paper displays, through paper_manifest.load() (manuscript/paper.tex
             when manuscript/ is present, 02_output/paper_manifest.json otherwise);
             02_output/values/values_generated.tex, 02_output/tables/*.tex,
             02_output/computed.json, 02_output/values/counts.json, run_all.sh
  WRITES     the blocks between the GENERATED markers in README.md
  FEEDS      no exhibit; it is the documentation of what feeds them

The tables are GENERATED, never typed: a mapping kept by hand is a second description of the
package, and drifts from the first. What keeps them honest is run_all.sh step 4/5, which
harvests the exhibits from paper.tex (or its manifest) and fails if any was not produced by
the run. If these tables and that check ever disagree, paper.tex is the authority.

Commented-out material is stripped before scanning, so the tables describe the compiled
document only.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPL = CODE.parent
sys.path.insert(0, str(HERE))
from make_tables_values import figure_macros  # noqa: E402  (the FIGURE_MACROS of each figure script)
import paper_manifest  # noqa: E402
from paper_manifest import strip_comments  # noqa: E402

# The run's own outputs. When manuscript/ is present its copies equal them byte for byte
# (run_all.sh checks it), so the README reads the same files with or without the manuscript.
VALUES = REPL / "02_output" / "values" / "values_generated.tex"
TABLES = REPL / "02_output" / "tables"
COMPUTED = REPL / "02_output" / "computed.json"
COUNTS = REPL / "02_output" / "values" / "counts.json"
RUNALL = REPL / "run_all.sh"
OUT = REPL / "README.md"

BLOCKS = ("TITLE", "MAP", "COUNTS", "PROGRAMS", "MACROS")


# ── the producer of every figure file, declared once and reconciled with run_all.sh ──
# (file -> (script under 01_code/, what the script reads))
FIGURE_PRODUCERS = {
    "fig_intro_two_boundaries.pdf": ("03_figures/fig_intro_two_boundaries.py", "no data (closed forms)"),
    "fig_record_power.pdf": ("03_figures/fig_record_power.py",
                             "record-study outputs (the study reads the AMOC series)"),
    "fig_bifurcation_intro.pdf": ("03_figures/fig_bifurcation_intro.py", "no data (closed forms)"),
    "fig_crossover.pdf": ("03_figures/fig_crossover.py", "AMOC series, via the recovery-rate estimate"),
    "fig_forest_tc.pdf": ("03_figures/fig_forest_tc.py", "RCMIP emission paths; literature constants"),
}
TABLE_PRODUCERS = {
    "disagreements.tex": "04_tables_values/make_tables_values.py",
    "eps_dyn.tex": "04_tables_values/make_tables_values.py",
}

# ── every script of 01_code/, with its role. Reconciled with the directory below. ──
SCRIPTS = {
    "inputs_literature.py": "module: the sourced constants, each with its citation",
    "tcre_draw.py": "module: the TCRE draw shared by the three Monte-Carlo scripts",
    "check_exhibits.py": "control (step 4/5): every exhibit paper.tex displays was produced by this run",
    "01_calibration/calibration_ci.py": "Monte Carlo over the calibration inputs: budgets M* and their intervals",
    "01_calibration/eps_inf_ci.py": "Monte Carlo: the deadline bound t_c of each element",
    "01_calibration/bhp_bound.py": "the ceiling of ssec:ceiling under the floored hazard (eq:ceiling_body)",
    "01_calibration/record_detectability.py": "reads the record-study outputs; the \\rec* macros",
    "01_calibration/crossover.py": "which disagreement moves the price (ssec:which_parameter): the \\xover* macros",
    "01_calibration/ews_calibration.py": "reads the AMOC series: recovery rate, variance and noise of the overturning circulation",
    "01_calibration/annual_mean_ou.py": "module: annual means -> continuous OU parameters (imported by ews_calibration.py and the record study)",
    "01_calibration/phi_fold.py": "module: the fold amplitude Phi of app:amp",
    "01_calibration/eps_dyn.py": "module: the lower edge eps_dyn of the fixed-budget regime and the factor F (ssec:window)",
    "01_calibration/emission_paths.py": "module: the deadline bound on SSP2-4.5 and SSP3-7.0 (RCMIP extract)",
    "01_calibration/reduced_form_scc.py": "the reduced-form price level (diagnostic macros)",
    "02_solvers/price.py": "module: the price at fixed state, eq:gen_scc, and its leading law",
    "02_solvers/noise_reversal_frontier.py": "BVP: where the noise reversal of the theorem (ssec:gen_thm) ends",
    "02_solvers/validation_bvp.py": "BVP: the numerical validation of app:solver",
    "02_solvers/moving_budget.py": "the price while the budget is being spent (ssec:window, app:additivity)",
    "02_solvers/paper2_all_bifurcations.py": "module: the BVP solvers of the three geometries (imported by validation_bvp.py)",
    "03_figures/fig_intro_two_boundaries.py": "Figure fig:two_boundaries",
    "03_figures/fig_record_power.py": "Figure fig:record_power (imports record_detectability.compute())",
    "03_figures/fig_bifurcation_intro.py": "Figure fig:bifurcation_intro",
    "03_figures/fig_crossover.py": "Figure fig:crossover (imports crossover.py and eps_dyn.py)",
    "03_figures/fig_forest_tc.py": "Figure fig:forest_tc and the deadline macros on the emission paths",
    "04_tables_values/collect_computed.py": "step 2/5: collects the printed results into computed.json",
    "04_tables_values/make_tables_values.py": "step 3/5: writes values.tex, the two table files and VALUES_DIFF.md",
    "04_tables_values/make_inventory.py": "step 3/5: writes the generated sections of this README",
    "04_tables_values/paper_manifest.py": "step 3/5: what the paper displays, written to 02_output/paper_manifest.json from manuscript/ when present, read from it otherwise",
    "05_record_study/record_pipeline.py": "module: the record study's trend estimators, the AMOC reader and the output paths",
    "05_record_study/identification.py": "record study: joint region of the two trends on the AMOC record -> identification.json",
    "05_record_study/puissance.py": "record study: power of the trend estimators -> puissance.json",
    "05_record_study/mle_ditlevsen.py": "record study: the Ditlevsen-Ditlevsen likelihood test -> mle_ditlevsen.json",
    "05_record_study/calib82.py": "record study: that test's false-rejection rate on the record -> calib82.json",
    "05_record_study/estimateurs.py": "record study: trend, level and ramp statistics -> estimateurs.json",
    "05_record_study/fenetres.py": "record study: the level statistic by window width -> fenetres.json",
    "05_record_study/horizon_np.py": "record study: the Neyman-Pearson bound -> horizon_np.json",
    "05_record_study/horizon_attente.py": "record study: the bound while waiting -> horizon_attente.json",
    "05_record_study/compare_shipped.py": "control: recomputed study outputs against 00_data/record_study/",
}


def data_for(path: str) -> str:
    """What a script needs in order to run."""
    if "ews_calibration" in path or "identification" in path:
        return "AMOC series"
    if "calib82" in path:
        return "AMOC series, via identification.py"
    if any(s in path for s in ("reduced_form_scc", "noise_reversal_frontier", "fig_crossover", "moving_budget")):
        return "AMOC series, via ews_calibration.py"
    if "record_detectability" in path or "fig_record_power" in path:
        return "record-study outputs (02_output/record_study/)"
    if "emission_paths" in path or "fig_forest_tc" in path:
        return "RCMIP extract (00_data/emissions/)"
    if "inputs_literature" in path:
        return "literature constants"
    return "none (self-contained)"


def main() -> None:
    man = paper_manifest.load()
    computed = json.loads(COMPUTED.read_text(encoding="utf-8"))
    values_txt = VALUES.read_text(encoding="utf-8")
    VAL = r"(?:[^{}]|\{[^{}]*\})*"   # one level of braces allowed in a value (a unit)
    defined = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(" + VAL + r")\}", values_txt))

    # ── the network count, read once, from the file the generated sections describe ──
    net_macros = {m.group(1) for line in values_txt.splitlines() if "[requires network]" in line
                  for m in [re.search(r"\\newcommand\{\\(\w+)\}", line)] if m}
    flagged = {m for m in defined if m in computed and computed[m].get("requires_network")}
    if flagged != net_macros:
        raise SystemExit(
            "make_inventory.py: values.tex and computed.json disagree about which macros need the\n"
            "        network.\n"
            + "".join(f"  marked in values.tex, not flagged in computed.json: {m}\n" for m in sorted(net_macros - flagged))
            + "".join(f"  flagged in computed.json, not marked in values.tex: {m}\n" for m in sorted(flagged - net_macros))
            + "        Rerun make_tables_values.py so values.tex describes this run.")

    # ── scripts on disk against the SCRIPTS map ──
    on_disk = {p.relative_to(CODE).as_posix() for p in CODE.rglob("*.py") if "__pycache__" not in p.parts}
    unlisted, stale = sorted(on_disk - set(SCRIPTS)), sorted(set(SCRIPTS) - on_disk)
    if unlisted or stale:
        raise SystemExit("make_inventory.py: the SCRIPTS map is out of step with 01_code/.\n"
                         + "".join(f"  on disk, not listed: {p}\n" for p in unlisted)
                         + "".join(f"  listed, not on disk: {p}\n" for p in stale))
    run_lines = re.findall(r'^[^\S\n]*run[^\S\n]+\S+[^\S\n]+"\$CODE/([^"]+)"', RUNALL.read_text(encoding="utf-8"),
                           re.MULTILINE)
    for fig, (scr, _d) in FIGURE_PRODUCERS.items():
        if scr not in run_lines:
            raise SystemExit(f"make_inventory.py: {fig} is declared as made by {scr}, which run_all.sh does not run.")

    # ── the exhibits, numbered ──
    rows_map = []
    fl = man["floats"]
    for f in fl:
        name = f"{f['kind'].capitalize()} {f['num']}"
        lab = f"`{f['label']}`" if f["label"] else ""
        if f["kind"] == "figure":
            for g in f["graphics"]:
                if g not in FIGURE_PRODUCERS:
                    raise SystemExit(f"make_inventory.py: paper.tex displays {g}, which has no declared producer.")
                scr, data = FIGURE_PRODUCERS[g]
                rows_map.append((name, lab, f"`02_output/figures/{g}`", f"`01_code/{scr}`", data))
        elif f["inputs"]:
            for t in f["inputs"]:
                if t not in TABLE_PRODUCERS:
                    raise SystemExit(f"make_inventory.py: paper.tex inputs tables/{t}, which has no declared producer.")
                body = (TABLES / t).read_text(encoding="utf-8")
                used = set(re.findall(r"\\([A-Za-z]+)", strip_comments(body))) & set(defined)
                net = "some cells need the AMOC series" if used & net_macros else "no data beyond values.tex"
                rows_map.append((name, lab, f"`02_output/tables/{t}`", f"`01_code/{TABLE_PRODUCERS[t]}`",
                                 f"cells are macros of `values.tex`; {net}"))
        else:
            used = set(f["commands"]) & set(defined)
            what = (f"{len(used)} macros of `values.tex`" + (", some needing the AMOC series" if used & net_macros else "")
                    if used else "no computed number")
            rows_map.append((name, lab, "typed in the manuscript", "—", what))
    n_fig = sum(1 for f in fl if f["kind"] == "figure")
    n_tab = sum(1 for f in fl if f["kind"] == "table")

    # ── macros by producing script ──
    by_base = {}
    for p in sorted(on_disk):
        by_base.setdefault(p.rsplit("/", 1)[-1], p)
    COLLECT = by_base["collect_computed.py"]

    def producer(src: str) -> str:
        """A source that opens with a script name is that script's; collect_computed.py's own
        closed forms open with 'closed form'; a named script not on disk is reported."""
        lead = re.match(r"\s*(\w+\.py)", src)
        if lead:
            return by_base.get(lead.group(1), lead.group(1) + "  [NOT IN 01_code/]")
        if "closed form" in src.lower():
            return COLLECT
        m = re.search(r"(\w+\.py)", src)
        return by_base.get(m.group(1), m.group(1) + "  [NOT IN 01_code/]") if m else COLLECT

    by_src = {}
    for mac in sorted(defined):
        full = computed[mac]["source"] if mac in computed else "inputs_literature.py (sourced constant)"
        by_src.setdefault(producer(full), []).append(mac)
    mac_rows = [(f"`01_code/{p}`", str(len(by_src[p])), str(sum(1 for m in by_src[p] if m in net_macros)),
                 "closed forms and sourced constants" if p == COLLECT else data_for(p))
                for p in sorted(by_src)]
    if sum(int(r[2]) for r in mac_rows) != len(net_macros):
        raise SystemExit("make_inventory.py: the macro table's network column does not sum to the "
                         f"{len(net_macros)} macros marked in values.tex.")

    # ── macros the paper does not use: paper.tex, the table files it inputs, FIGURE_MACROS ──
    used = set(man["commands"])
    for t in man["table_inputs"]:
        used |= set(re.findall(r"\\([A-Za-z]+)", strip_comments((TABLES / t).read_text(encoding="utf-8"))))
    used |= set(figure_macros())
    unused = sorted(m for m in defined if m not in used)

    # ── the counts of the data availability statement, from counts.json ──
    if not COUNTS.exists():
        raise SystemExit(f"[ERROR] {COUNTS} is missing. It is written by make_tables_values.py, which "
                         "runs immediately before this script in step 3/5. Run: bash run_all.sh")
    c = json.loads(COUNTS.read_text(encoding="utf-8"))
    if (c["computed_network"], c["total"]) != (len(net_macros), len(defined)):
        raise SystemExit("make_inventory.py: counts.json and values.tex disagree "
                         f"({c['computed_network']} network of {c['total']} against {len(net_macros)} of "
                         f"{len(defined)}); the two files are from different runs. Run: bash run_all.sh")

    def table(rows, cols):
        return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
                + "\n".join("| " + " | ".join(r) + " |" for r in rows))

    blocks = {}
    blocks["TITLE"] = f'**"{man["title"]}"**\nRomain Fillon'
    blocks["MAP"] = (
        f"The paper displays {n_fig} figures and {n_tab} tables. Numbers are as LaTeX prints them; "
        "labels are those of the manuscript, as listed in `02_output/paper_manifest.json`.\n\n"
        + table(rows_map, ["Exhibit", "Label", "File", "Produced by", "Inputs"])
        + f"\n\nEvery number the text quotes is a macro of `values.tex` ({len(defined)} macros), "
        "written as `02_output/values/values_generated.tex` by "
        "`01_code/04_tables_values/make_tables_values.py` from `02_output/computed.json` "
        "(the printed results of the scripts, collected by `collect_computed.py`) and from "
        "`01_code/inputs_literature.py` (the sourced constants). Each macro carries, as a comment "
        "on its own line, the script and the inputs it came from; section 3 groups them by script.")
    blocks["COUNTS"] = (
        f"**{c['computed_network']} of the {c['computed']} computed macros in `values.tex` carry "
        f"`[requires network]`**, alongside {c['literature']} sourced literature constants "
        f"({c['total']} macros in the file).")
    blocks["PROGRAMS"] = table([(f"`01_code/{p}`", SCRIPTS[p], data_for(p)) for p in sorted(SCRIPTS)],
                               ["Script", "Role", "Needs"])
    blocks["MACROS"] = table(mac_rows, ["Produced by", "Macros", "of which [requires network]", "Needs"])

    readme = OUT.read_text(encoding="utf-8")
    for key in BLOCKS:
        b, e = f"<!-- BEGIN GENERATED {key} -->", f"<!-- END GENERATED {key} -->"
        if b not in readme or e not in readme:
            raise SystemExit(f"[ERROR] {OUT} carries no {b} ... {e} markers; these sections are generated, "
                             "restore the markers.")
        readme = readme[:readme.index(b)] + b + "\n" + blocks[key] + "\n" + readme[readme.index(e):]
    OUT.write_text(readme, encoding="utf-8", newline="\n")
    print(f"README.md: {n_fig} figures, {n_tab} tables, {len(defined)} macros ({len(net_macros)} network, "
          f"{len(unused)} not used by the paper), {len(SCRIPTS)} scripts")
    src = "paper.tex" if paper_manifest.manuscript_present() else "02_output/paper_manifest.json"
    print(f"README.md title: \"{man['title']}\"  (from {src})")


if __name__ == "__main__":
    main()
