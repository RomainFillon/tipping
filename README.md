# Replication package

<!-- BEGIN GENERATED TITLE -->
**"Pricing Risk at a Boundary"**
Romain Fillon
<!-- END GENERATED TITLE -->

## Overview

This package recomputes every number, figure and table of the paper from its inputs, with one
command (`bash run_all.sh`). The manuscript itself is not part of this repository: `values.tex`
(written as `02_output/values/values_generated.tex`), the table files (`02_output/tables/`) and
the figures (`02_output/figures/`) are its inputs, and the run writes them all.

The paper prices the risk of crossing a climate tipping threshold. The code solves the
pricing problem numerically (boundary-value problems and closed forms), calibrates it to three
climate elements by Monte Carlo over the published ranges of their inputs, estimates the
recovery rate of the Atlantic overturning circulation (AMOC) from an observed sea-surface
temperature record, and measures by simulation what that record can and cannot establish.

### Exhibits of the paper and the code that produces them

<!-- BEGIN GENERATED MAP -->
The paper displays 5 figures and 4 tables. Numbers are as LaTeX prints them; labels are those of the manuscript, as listed in `02_output/paper_manifest.json`.

| Exhibit | Label | File | Produced by | Inputs |
|---|---|---|---|---|
| Figure 1 | `fig:two_boundaries` | `02_output/figures/fig_intro_two_boundaries.pdf` | `01_code/03_figures/fig_intro_two_boundaries.py` | no data (closed forms) |
| Figure 2 | `fig:record_power` | `02_output/figures/fig_record_power.pdf` | `01_code/03_figures/fig_record_power.py` | record-study outputs (the study reads the AMOC series) |
| Figure 3 | `fig:bifurcation_intro` | `02_output/figures/fig_bifurcation_intro.pdf` | `01_code/03_figures/fig_bifurcation_intro.py` | no data (closed forms) |
| Table 1 | `tab:geometries` | typed in the manuscript | — | no computed number |
| Figure 4 | `fig:crossover` | `02_output/figures/fig_crossover.pdf` | `01_code/03_figures/fig_crossover.py` | AMOC series, via the recovery-rate estimate |
| Table 2 | `tab:disagreements` | `02_output/tables/disagreements.tex` | `01_code/04_tables_values/make_tables_values.py` | cells are macros of `values.tex`; some cells need the AMOC series |
| Figure 5 | `fig:forest_tc` | `02_output/figures/fig_forest_tc.pdf` | `01_code/03_figures/fig_forest_tc.py` | RCMIP emission paths; literature constants |
| Table 3 | `tab:eps_dyn` | `02_output/tables/eps_dyn.tex` | `01_code/04_tables_values/make_tables_values.py` | cells are macros of `values.tex`; some cells need the AMOC series |
| Table 4 | `tab:inputs` | typed in the manuscript | — | 26 macros of `values.tex` |

Every number the text quotes is a macro of `values.tex` (242 macros), written as `02_output/values/values_generated.tex` by `01_code/04_tables_values/make_tables_values.py` from `02_output/computed.json` (the printed results of the scripts, collected by `collect_computed.py`) and from `01_code/inputs_literature.py` (the sourced constants). Each macro carries, as a comment on its own line, the script and the inputs it came from; section 3 groups them by script.
<!-- END GENERATED MAP -->

The generated sections of this README (between `<!-- BEGIN GENERATED ... -->` markers) are
written by `01_code/04_tables_values/make_inventory.py` at every run, from what the paper
displays (`02_output/paper_manifest.json`, see section 3) and the run's outputs; a hand edit
there is overwritten by the next run.

---

## 1. Data Availability Statement

### Statement about rights

The author has legitimate access to and permission to use all the data used in this
manuscript. The one dataset whose host publishes no licence (the AMOC series below) is not
redistributed; the package contains a script that downloads it from its publisher. The one
third-party dataset that is redistributed (the RCMIP extract) is redistributed under its own
licence, CC BY-SA 4.0.

### Summary of the data sources

| Data | Source and citation | Licence | In the package | Read by |
|---|---|---|---|---|
| AMOC SST fingerprint: subpolar-gyre sea-surface temperature index, annual, 1871–2016 | Caesar, L., S. Rahmstorf, A. Robinson, G. Feulner and V. Saba (2018), "Observed fingerprint of a weakening Atlantic Ocean overturning circulation", *Nature* 556, 191–196, doi:10.1038/s41586-018-0006-5; file served by the Potsdam Institute for Climate Impact Research (PIK). Derived by those authors from HadISST: Rayner, N. A., et al. (2003), *J. Geophys. Res.* 108(D14), 4407, doi:10.1029/2002JD002670 | None published by the host | **No.** Downloaded once, before the run, by `00_data/amoc/fetch_pik_sg_index.py` | `01_code/01_calibration/ews_calibration.py`; `01_code/05_record_study/identification.py` |
| CO2 emission paths SSP2-4.5 and SSP3-7.0, 2010–2500 | Nicholls, Z. and J. Lewis (2021), Reduced Complexity Model Intercomparison Project (RCMIP) protocol v5.1.0, Zenodo, doi:10.5281/zenodo.4589756; Nicholls, Z. R. J., et al. (2020), *Geosci. Model Dev.* 13, 5175–5190, doi:10.5194/gmd-13-5175-2020; scenarios and extensions of Meinshausen, M., et al. (2020), *Geosci. Model Dev.* 13, 3571–3605, doi:10.5194/gmd-13-3571-2020 | **CC BY-SA 4.0** | **Yes**, a two-row extract, `00_data/emissions/rcmip_co2_world_ssp245_ssp370.csv`, with its checksums and the script that rebuilds it from the published file (`extract_rcmip.py`) | `01_code/01_calibration/emission_paths.py` |
| Outputs of the record-detectability study (eight JSON files) | This paper: written by the scripts of `01_code/05_record_study/` | Same as the code | **Yes**, `00_data/record_study/`; recomputed by the default run, or reused with `RECORD_STUDY=cached` | `01_code/01_calibration/record_detectability.py` (through `02_output/record_study/`) |
| Values taken from publications | See the table below | — | **Yes**, as constants with their citations in `01_code/inputs_literature.py` | the calibration scripts |

#### The AMOC series (not redistributed)

The series is the file

    https://www.pik-potsdam.de/~caesar/AMOC_slowdown/sg_index_hadisst.txt

of 1 638 bytes: one header line (`# for the years 1871-2016 in K`) and 146 annual values. The
host page publishes no licence, copyright notice or conditions of use, so the package does
not redistribute the file. A replicator downloads it once, before the run:

    python3 00_data/amoc/fetch_pik_sg_index.py

The script writes the bytes unchanged to `00_data/amoc/sg_index_hadisst.txt` and refuses them
unless their md5 is the one the paper was computed on, `c48516162322edebaaf113b2841f93be`
(recorded in `00_data/amoc/PROVENANCE.md`). The scripts that read the file check the md5 again
and stop with an explicit message if the file is missing or different. This download is the
package's only external dependency.

If the URL stops resolving, the index can be rebuilt from HadISST with the subpolar-gyre region
definition of Caesar et al. (2018). That reconstruction is not part of the package; a rebuilt
series will have a different md5, and should match the summary statistics
`ews_calibration.py` prints for the series itself: 146 values, variance of the detrended annual
means 0.0683 K², lag-one autocorrelation 0.6106.

<!-- BEGIN GENERATED COUNTS -->
**36 of the 199 computed macros in `values.tex` carry `[requires network]`**, alongside 43 sourced literature constants (242 macros in the file).
<!-- END GENERATED COUNTS -->

Those macros cannot be recomputed without the series. In the default mode the record study also
needs it (`identification.py`, then `calib82.py`); with `RECORD_STUDY=cached` the study's shipped
outputs are used instead, and the `\rec*` macros are recomputed from them without the series.

#### The RCMIP extract (redistributed under CC BY-SA 4.0)

`00_data/emissions/rcmip_co2_world_ssp245_ssp370.csv` is an adaptation (two rows,
`Emissions|CO2`, `World`, scenarios `ssp245` and `ssp370`, values unchanged) of
`rcmip-emissions-annual-means-v5-1-0.csv` from the RCMIP protocol v5.1.0 by Zebedee Nicholls and
Jared Lewis (doi:10.5281/zenodo.4589756), licensed under the Creative Commons
Attribution-ShareAlike 4.0 International licence
(https://creativecommons.org/licenses/by-sa/4.0/). **The extract remains under CC BY-SA 4.0**,
and is not covered by the licence of the code. The published file (48 MB) is not shipped;
`00_data/emissions/extract_rcmip.py` rebuilds the extract from it and checks its md5. Details
in `00_data/emissions/PROVENANCE.md`.

#### Values taken from publications

Every such value is a constant of `01_code/inputs_literature.py`, with its citation and, where
the source has one, the table or section it is read from. The table below groups them by source;
the citation strings in that file are the reference.

| Source | Values | Where in the source |
|---|---|---|
| Armstrong McKay, D. I., et al. (2022), "Exceeding 1.5°C global warming could trigger multiple climate tipping points", *Science* 377, eabn7950, doi:10.1126/science.abn7950 | Thresholds ΔT* (central, low and high ends) and timescales T (central and range) of the AMOC, the Amazon and the West Antarctic ice sheet | Table 1 |
| IPCC (2021), *Climate Change 2021: The Physical Science Basis*, Working Group I contribution to the Sixth Assessment Report, Cambridge University Press, doi:10.1017/9781009157896 | Transient climate response to cumulative CO2 emissions (TCRE): central 0.45, likely range 0.27–0.63 °C per 1000 GtCO2 | Summary for Policymakers, D.1.1 |
| Forster, P. M., et al. (2024), "Indicators of Global Climate Change 2023", *Earth Syst. Sci. Data* 16, 2625–2658 | Observed warming above pre-industrial, 2014–2023 average, 1.19 [1.06 to 1.30] °C, rounded to 1.2 | as cited in `inputs_literature.py` |
| Friedlingstein, P., et al. (2023), "Global Carbon Budget 2023", *Earth Syst. Sci. Data* 15, 5301–5369 | Present global CO2 emissions, fossil and land use, about 40 GtCO2 per year | as cited in `inputs_literature.py` |
| Dietz, S., J. Rising, T. Stoerk and G. Wagner (2021), "Economic impacts of tipping points in the climate system", *PNAS* 118(34), e2103081118 | Baseline SCC and the increase due to tipping points (Amazon, all elements) | Table 2 |
| Hill, E. A., et al. (2023), *The Cryosphere* 17, 3739–3759, doi:10.5194/tc-17-3739-2023 | Relaxation time of the West Antarctic ice sheet's grounding-line flux, 10 to 20 years | Section 4.1 |
| Boulton, C. A., T. M. Lenton and N. Boers (2022), *Nature Climate Change* 12, 271–278 | Recovery time of the Amazon vegetation index | Methods |
| Fuss, S., et al. (2018), *Environ. Res. Lett.* 13, 063002 | Costs of direct air capture (backstop range) | as cited in `inputs_literature.py` |
| Howard, P. H. and T. Sterner (2017), *Environ. Resource Econ.* 68(1), 197–225 | Ratio of their SCC to DICE-2013R | Abstract |
| Hambel, C., F. van der Ploeg and T. van den Bremer, replication package, Zenodo 20075290 | Calibration constants of their tipping model, read from the printed output of their code (no code of theirs is included here) | `Code_Parameters.m`, `Code_Initiation.m`, as printed by `MakeTable1.m` |

Constants marked "modelling choice" in `inputs_literature.py` (the discount rate, prior widths,
the number of Monte-Carlo draws) are the author's and are stated there.

---

## 2. Computational requirements

### Software

- **Python 3.14.3** (CPython, 64-bit).
- The complete environment, with every package at its exact version, is
  **`requirements-lock.txt`** (14 packages). Use it to reproduce the outputs.
  `01_code/requirements.txt` lists only the four direct dependencies (numpy 2.4.4,
  scipy 1.17.1, matplotlib 3.10.8, pandas 3.0.1), at the same versions.
- **Bash**, to run `run_all.sh`. On Windows, Git Bash.
- **No LaTeX.** Step 5/5 compiles the manuscript only when a `manuscript/` directory sits next
  to the package, which is not the case in this repository; the step is skipped and says so.

### Systems tested

- Windows 11 Pro, Git Bash, Python 3.14.3 (the author's machine, and a fresh clone in a
  fresh virtual environment built from `requirements-lock.txt`).

No other system has been tested. Nothing in the code is specific to Windows; on Linux and macOS
`run_all.sh` calls `python3` by default.

### Hardware

Intel Core i7-1165G7 (4 cores, 8 threads, 2.8 GHz), 32 GB of RAM. The record study uses
6 worker processes. Disk: about 60 MB for the package and its outputs.

### Runtime

Measured on the hardware below, from a fresh clone, wall clock of `run_all.sh`:

| Mode | Command | Runtime |
|---|---|---|
| Full recomputation (default) | `bash run_all.sh` | 1 h 25 min to 2 h 11 min (5,114 s and 7,854 s, two runs) |
| Record study reused | `RECORD_STUDY=cached bash run_all.sh` | 9 min (549 s) |

The record study accounts for almost all of the difference: its eight scripts took 4,622 s and
7,239 s in the two full runs (`puissance.py` 1,292–2,200 s, `mle_ditlevsen.py` 822–1,413 s,
`calib82.py` 791–1,254 s, `identification.py` 572–1,169 s, `horizon_np.py` 649–705 s,
`horizon_attente.py` about 460 s, the other two under 30 s each). The longest script outside it
is `noise_reversal_frontier.py` (7 to 8 min). The faster full run is the clean-room run, in a
fresh virtual environment built from `requirements-lock.txt`, with nothing else running.
The two modes write the same `values.tex`, the same `computed.json` and the same tables, byte
for byte; `run_all.sh` prints its per-step wall clock at the end.

---

## 3. Description of the programs

`run_all.sh` is the single entry point. It runs, in order:

1. **Step 1/5, generators.** Every script that produces a number, a figure or a study output is
   run once, and its standard output is captured in `02_output/values/raw_runs/<key>.out`. In
   dependency order: the closed-form figures; the Monte-Carlo calibration
   (`calibration_ci.py`, `eps_inf_ci.py`, `fig_forest_tc.py`); the ceiling (`bhp_bound.py`); the
   record study (`01_code/05_record_study/`, eight scripts, or a copy of their shipped outputs
   with `RECORD_STUDY=cached`) and its reader (`record_detectability.py`, `fig_record_power.py`);
   `crossover.py`; the AMOC recovery rate (`ews_calibration.py`) and the four scripts that read
   it (`reduced_form_scc.py`, `noise_reversal_frontier.py`, `fig_crossover.py`,
   `moving_budget.py`); the numerical validation (`validation_bvp.py`).
2. **Step 2/5.** `collect_computed.py` parses the captured outputs into `02_output/computed.json`.
   It runs nothing; a missing capture stops it.
3. **Step 3/5.** `make_tables_values.py` writes `values.tex` (as
   `02_output/values/values_generated.tex`), the two table files (`02_output/tables/`) and
   `02_output/values/VALUES_DIFF.md` (a per-macro comparison with the previous `values.tex`);
   `make_inventory.py` writes the generated sections of this README.
4. **Step 4/5.** `check_exhibits.py` checks that every file the paper includes exists and was
   written by this run.
5. **Step 5/5.** Compiles the manuscript, when it is present (see below); skipped otherwise.

The run exits non-zero if any step fails, and its last lines name the failed steps, the
verdict of the record-study comparison and the time each step took. If step 2/5 or 3/5 fails,
the run stops there with a non-zero code and writes nothing more, so that a failed run cannot
leave behind the outputs of an earlier one under the appearance of a success.

**With or without the manuscript.** The manuscript is not part of this repository. What the
code needs to know about it (the title, the figures and tables it displays, with their labels
and sections, and the macros it cites) is in `02_output/paper_manifest.json`, a file generated
from `paper.tex` by `01_code/04_tables_values/paper_manifest.py` and never edited by hand. The
mode is set by the presence of the `manuscript/` directory as a whole, never by a single file:

- *Without `manuscript/`* (this repository): the run prints
  `manuscript/ not present: using 02_output/paper_manifest.json`; step 4/5 checks the exhibits
  under `02_output/`; `VALUES_DIFF.md` compares with the committed
  `02_output/values/values_generated.tex`; step 5/5 is skipped.
- *With `manuscript/`* (the author's working copy): the run also copies `values.tex`, the tables
  and the figures into `manuscript/`, rewrites the manifest from `paper.tex`, checks that the
  two copies of each file are identical byte for byte, and compiles `manuscript/paper.pdf`.

The outputs under `02_output/` are the same in both modes.

### Directory layout

| Path | Contents |
|---|---|
| `run_all.sh` | The single entry point |
| `requirements-lock.txt` | The complete Python environment, exact versions |
| `00_data/amoc/` | The script that downloads the AMOC series, and its provenance; the series itself is not shipped |
| `00_data/emissions/` | The RCMIP extract (CC BY-SA 4.0), its provenance and the script that rebuilds it |
| `00_data/record_study/` | The shipped outputs of the record study, and their provenance |
| `01_code/01_calibration/` | Calibration: Monte Carlo, the AMOC recovery rate, the reduced-form level, the record-study reader |
| `01_code/02_solvers/` | The boundary-value problems and the price |
| `01_code/03_figures/` | The five figures |
| `01_code/04_tables_values/` | `computed.json`, `values.tex`, the table files, the paper manifest, this README's generated sections |
| `01_code/05_record_study/` | The record-detectability study |
| `02_output/` | Everything the run writes, including the inputs of the manuscript (`values/values_generated.tex`, `tables/`, the five figures in `figures/`) and `paper_manifest.json`; see the notes on what is versioned |

### Every script

<!-- BEGIN GENERATED PROGRAMS -->
| Script | Role | Needs |
|---|---|---|
| `01_code/01_calibration/annual_mean_ou.py` | module: annual means -> continuous OU parameters (imported by ews_calibration.py and the record study) | none (self-contained) |
| `01_code/01_calibration/bhp_bound.py` | the ceiling of ssec:ceiling under the floored hazard (eq:ceiling_body) | none (self-contained) |
| `01_code/01_calibration/calibration_ci.py` | Monte Carlo over the calibration inputs: budgets M* and their intervals | none (self-contained) |
| `01_code/01_calibration/crossover.py` | which disagreement moves the price (ssec:which_parameter): the \xover* macros | none (self-contained) |
| `01_code/01_calibration/emission_paths.py` | module: the deadline bound on SSP2-4.5 and SSP3-7.0 (RCMIP extract) | RCMIP extract (00_data/emissions/) |
| `01_code/01_calibration/eps_dyn.py` | module: the lower edge eps_dyn of the fixed-budget regime and the factor F (ssec:window) | none (self-contained) |
| `01_code/01_calibration/eps_inf_ci.py` | Monte Carlo: the deadline bound t_c of each element | none (self-contained) |
| `01_code/01_calibration/ews_calibration.py` | reads the AMOC series: recovery rate, variance and noise of the overturning circulation | AMOC series |
| `01_code/01_calibration/phi_fold.py` | module: the fold amplitude Phi of app:amp | none (self-contained) |
| `01_code/01_calibration/record_detectability.py` | reads the record-study outputs; the \rec* macros | record-study outputs (02_output/record_study/) |
| `01_code/01_calibration/reduced_form_scc.py` | the reduced-form price level (diagnostic macros) | AMOC series, via ews_calibration.py |
| `01_code/02_solvers/moving_budget.py` | the price while the budget is being spent (ssec:window, app:additivity) | AMOC series, via ews_calibration.py |
| `01_code/02_solvers/noise_reversal_frontier.py` | BVP: where the noise reversal of the theorem (ssec:gen_thm) ends | AMOC series, via ews_calibration.py |
| `01_code/02_solvers/paper2_all_bifurcations.py` | module: the BVP solvers of the three geometries (imported by validation_bvp.py) | none (self-contained) |
| `01_code/02_solvers/price.py` | module: the price at fixed state, eq:gen_scc, and its leading law | none (self-contained) |
| `01_code/02_solvers/validation_bvp.py` | BVP: the numerical validation of app:solver | none (self-contained) |
| `01_code/03_figures/fig_bifurcation_intro.py` | Figure fig:bifurcation_intro | none (self-contained) |
| `01_code/03_figures/fig_crossover.py` | Figure fig:crossover (imports crossover.py and eps_dyn.py) | AMOC series, via ews_calibration.py |
| `01_code/03_figures/fig_forest_tc.py` | Figure fig:forest_tc and the deadline macros on the emission paths | RCMIP extract (00_data/emissions/) |
| `01_code/03_figures/fig_intro_two_boundaries.py` | Figure fig:two_boundaries | none (self-contained) |
| `01_code/03_figures/fig_record_power.py` | Figure fig:record_power (imports record_detectability.compute()) | record-study outputs (02_output/record_study/) |
| `01_code/04_tables_values/collect_computed.py` | step 2/5: collects the printed results into computed.json | none (self-contained) |
| `01_code/04_tables_values/make_inventory.py` | step 3/5: writes the generated sections of this README | none (self-contained) |
| `01_code/04_tables_values/make_tables_values.py` | step 3/5: writes values.tex, the two table files and VALUES_DIFF.md | none (self-contained) |
| `01_code/04_tables_values/paper_manifest.py` | step 3/5: what the paper displays, written to 02_output/paper_manifest.json from manuscript/ when present, read from it otherwise | none (self-contained) |
| `01_code/05_record_study/calib82.py` | record study: that test's false-rejection rate on the record -> calib82.json | AMOC series, via identification.py |
| `01_code/05_record_study/compare_shipped.py` | control: recomputed study outputs against 00_data/record_study/ | none (self-contained) |
| `01_code/05_record_study/estimateurs.py` | record study: trend, level and ramp statistics -> estimateurs.json | none (self-contained) |
| `01_code/05_record_study/fenetres.py` | record study: the level statistic by window width -> fenetres.json | none (self-contained) |
| `01_code/05_record_study/horizon_attente.py` | record study: the bound while waiting -> horizon_attente.json | none (self-contained) |
| `01_code/05_record_study/horizon_np.py` | record study: the Neyman-Pearson bound -> horizon_np.json | none (self-contained) |
| `01_code/05_record_study/identification.py` | record study: joint region of the two trends on the AMOC record -> identification.json | AMOC series |
| `01_code/05_record_study/mle_ditlevsen.py` | record study: the Ditlevsen-Ditlevsen likelihood test -> mle_ditlevsen.json | none (self-contained) |
| `01_code/05_record_study/puissance.py` | record study: power of the trend estimators -> puissance.json | none (self-contained) |
| `01_code/05_record_study/record_pipeline.py` | module: the record study's trend estimators, the AMOC reader and the output paths | none (self-contained) |
| `01_code/check_exhibits.py` | control (step 4/5): every exhibit paper.tex displays was produced by this run | none (self-contained) |
| `01_code/inputs_literature.py` | module: the sourced constants, each with its citation | literature constants |
| `01_code/tcre_draw.py` | module: the TCRE draw shared by the three Monte-Carlo scripts | none (self-contained) |
<!-- END GENERATED PROGRAMS -->

### The macros of `values.tex`, by producing script

<!-- BEGIN GENERATED MACROS -->
| Produced by | Macros | of which [requires network] | Needs |
|---|---|---|---|
| `01_code/01_calibration/bhp_bound.py` | 11 | 0 | none (self-contained) |
| `01_code/01_calibration/calibration_ci.py` | 6 | 0 | none (self-contained) |
| `01_code/01_calibration/crossover.py` | 6 | 0 | none (self-contained) |
| `01_code/01_calibration/eps_inf_ci.py` | 3 | 0 | none (self-contained) |
| `01_code/01_calibration/ews_calibration.py` | 5 | 5 | AMOC series |
| `01_code/01_calibration/record_detectability.py` | 53 | 0 | record-study outputs (02_output/record_study/) |
| `01_code/01_calibration/reduced_form_scc.py` | 2 | 2 | AMOC series, via ews_calibration.py |
| `01_code/02_solvers/moving_budget.py` | 8 | 8 | AMOC series, via ews_calibration.py |
| `01_code/02_solvers/noise_reversal_frontier.py` | 2 | 2 | AMOC series, via ews_calibration.py |
| `01_code/02_solvers/validation_bvp.py` | 30 | 7 | none (self-contained) |
| `01_code/03_figures/fig_forest_tc.py` | 25 | 0 | RCMIP extract (00_data/emissions/) |
| `01_code/04_tables_values/collect_computed.py` | 48 | 12 | closed forms and sourced constants |
| `01_code/inputs_literature.py` | 43 | 0 | literature constants |
<!-- END GENERATED MACROS -->

---

## 4. Instructions to replicators

From a fresh clone, at the root of the package:

1. **Get the AMOC series** (once; the only step that uses the network):

       python3 00_data/amoc/fetch_pik_sg_index.py

   It prints the md5 and writes `00_data/amoc/sg_index_hadisst.txt`, or stops if the file PIK
   serves differs from the one the paper used. On Windows, `python3` may be the Microsoft Store
   alias rather than an interpreter; call the Python 3.14 of your installation by its path in
   steps 1 and 2 if so.

2. **Create the Python environment**:

       python3 -m venv .venv
       .venv/bin/python -m pip install -r requirements-lock.txt

   On Windows, the interpreter of the environment is `.venv/Scripts/python.exe`.

3. **Run the package**, with the interpreter of that environment:

       PYTHON=.venv/bin/python bash run_all.sh

   On Windows (Git Bash): `PYTHON=.venv/Scripts/python.exe bash run_all.sh`. To skip the
   recomputation of the record study, add `RECORD_STUDY=cached` in front of the command.

4. **Check the result.** The run ends with `Failed steps: none`, and with the number of the eight
   recomputed record-study outputs identical to the shipped ones. `02_output/values/VALUES_DIFF.md`
   compares every regenerated macro with the committed `values.tex` (a clean reproduction
   reports `0 substantive`), and `git status` shows whether any generated file changed: a
   clean reproduction changes none.

5. **The paper.** `values.tex` (`02_output/values/values_generated.tex`), the table files
   (`02_output/tables/`) and the figures (`02_output/figures/`) are the inputs of the
   manuscript, which is not part of this repository. Step 5/5 is skipped and says so.

Figures are reproduced to the byte. `run_all.sh` sets `SOURCE_DATE_EPOCH=1767225600`
(2026-01-01 00:00:00 UTC, a fixed constant rather than the date of any run), which matplotlib
writes as the creation date of every PDF instead of the time of the run. A figure script run by
hand without it draws the same picture but writes the current date, and so different bytes.

---

## 5. Notes

### Random seeds

Every random draw is seeded, so a rerun returns the same numbers:

- `01_code/01_calibration/calibration_ci.py`, `01_code/01_calibration/eps_inf_ci.py` and
  `01_code/03_figures/fig_forest_tc.py`: `np.random.seed(42)`, set once before the draws;
  `01_code/tcre_draw.py` draws right after it.
- The record study, `01_code/05_record_study/`: one `numpy.random.default_rng` seed per
  simulated cell, stated in each script's header (`identification.py` 42, `puissance.py` 1, 2, …,
  `mle_ditlevsen.py` 6000+, `calib82.py` 8000+, `estimateurs.py` 1300+, `fenetres.py` 2000+,
  `horizon_np.py` 5000+, `horizon_attente.py` 7000+). The cells run on 6 worker processes; each
  cell carries its own seed, so the results do not depend on the number of processes.
- No other script draws random numbers (the solvers, `ews_calibration.py` and the figures are
  deterministic).

### Macros emitted but not used by the paper

<!-- BEGIN GENERATED UNUSED -->
`values.tex` defines 242 macros; 137 are used by `paper.tex`, the table files it inputs or the figures, and **105 are not**. These are diagnostics: intermediate results and checks that the scripts compute on the way to the reported numbers, kept so that each can be inspected next to its provenance. By producing script:

| Produced by | Macros not used by the paper |
|---|---|
| `01_code/01_calibration/bhp_bound.py` | 8 |
| `01_code/01_calibration/eps_inf_ci.py` | 1 |
| `01_code/01_calibration/ews_calibration.py` | 2 |
| `01_code/01_calibration/record_detectability.py` | 41 |
| `01_code/01_calibration/reduced_form_scc.py` | 2 |
| `01_code/02_solvers/moving_budget.py` | 4 |
| `01_code/02_solvers/noise_reversal_frontier.py` | 2 |
| `01_code/02_solvers/validation_bvp.py` | 13 |
| `01_code/03_figures/fig_forest_tc.py` | 8 |
| `01_code/04_tables_values/collect_computed.py` | 12 |
| `01_code/inputs_literature.py` | 12 |
<!-- END GENERATED UNUSED -->

### Generated outputs are versioned

The inputs of the manuscript are committed: `02_output/values/values_generated.tex` (the
manuscript's `values.tex`), `02_output/tables/disagreements.tex`, `02_output/tables/eps_dyn.tex`
and the five figures of the paper in `02_output/figures/`. So are `02_output/computed.json`,
`02_output/values/counts.json`, `02_output/values/VALUES_DIFF.md` and
`02_output/paper_manifest.json`, which record the state the paper was compiled from. A run
overwrites all of them; `git diff` then shows what changed. The other outputs (the other files
in `02_output/figures/`, `02_output/record_study/`, the captured outputs in
`02_output/values/raw_runs/`) are not versioned.

---

## 6. License

- **Code.** The code of this package (`01_code/`, `run_all.sh` and the scripts under
  `00_data/`) is released under the MIT licence; see `LICENSE`.
- **The RCMIP extract**, `00_data/emissions/rcmip_co2_world_ssp245_ssp370.csv`, is
  redistributed under its own licence, CC BY-SA 4.0, and not under the MIT licence. Its
  attribution and licence notice are in `00_data/emissions/PROVENANCE.md` and in section 1
  ("The RCMIP extract").
- **The AMOC series** of Caesar et al. (2018) is not redistributed: its host publishes no
  licence. `00_data/amoc/fetch_pik_sg_index.py` downloads it from its publisher; see section 1
  ("The AMOC series").
