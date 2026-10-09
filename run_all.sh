#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- the replication package's single entry point.
#
#   bash run_all.sh                         # full recomputation (about 3 hours)
#   RECORD_STUDY=cached bash run_all.sh     # reuse the shipped record-study outputs (about 15 min)
#   PYTHON=/path/to/python bash run_all.sh  # choose the interpreter (default: python3)
#
# Recomputes every number, table and figure the manuscript displays, from the raw
# inputs, into 02_output/, and compiles the manuscript when it is present.
#
#   1/5  run every generator once, capturing its stdout into 02_output/values/raw_runs/
#   2/5  collect the printed results into 02_output/computed.json
#   3/5  write values.tex and the table files into 02_output/, and copy them, with the
#        figures, where paper.tex reads them (only when manuscript/ is present)
#   4/5  check that every exhibit paper.tex displays was produced by this run
#   5/5  compile manuscript/paper.pdf (only when manuscript/ is present)
#
# With or without the manuscript. The public package does not include manuscript/. When the
# directory is absent -- the whole directory, never a single missing file -- the run says so,
# reads what the paper displays from the committed 02_output/paper_manifest.json, checks the
# exhibits under 02_output/, and skips the compilation. Every output is the same either way.
#
# Exit code 0 means every step succeeded and every exhibit was produced. Any other value
# means something failed, and the summary names it. If step 2/5 or 3/5 fails, the run stops
# there and writes no output at all, rather than rebuilding them from an earlier run's results.
#
# There is no cache of generator outputs: every generator is executed by this script, every
# time, and step 2/5 reads only what this run captured. The one documented exception is
# RECORD_STUDY=cached, which copies the eight shipped outputs of the record-detectability
# study (00_data/record_study/) instead of recomputing them; every other step still runs.
#
# No network. The run makes no network call. One input, the AMOC SST fingerprint (Caesar et
# al. 2018, PIK), is not shipped because its host publishes no licence: download it ONCE,
# before this script, with
#     python3 00_data/amoc/fetch_pik_sg_index.py
# The scripts that read it stop if it is missing or if its md5 differs from the one recorded
# in 00_data/amoc/PROVENANCE.md. See the data availability statement in README.md.
#
# PYTHONUTF8=1 is set below and is required on Windows: the scripts print Greek letters.
#
# SOURCE_DATE_EPOCH is fixed below, so that the figure PDFs are byte-reproducible: matplotlib
# writes it as the CreationDate of every PDF instead of the time of the run. The value,
# 1767225600, is 2026-01-01 00:00:00 UTC; it is a constant, not the date of any run.
# =============================================================================
set -uo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export SOURCE_DATE_EPOCH=1767225600

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/01_code"
FIGOUT="$ROOT/02_output/figures"
RAW="$ROOT/02_output/values/raw_runs"
STUDY="$ROOT/02_output/record_study"
MANU="$ROOT/manuscript"
MANFIG="$MANU/figures"
MANIFEST="$ROOT/02_output/paper_manifest.json"
COUNTS="$ROOT/02_output/values/counts.json"   # the one source of the macro counts
PY="${PYTHON:-python3}"
# A relative path (PYTHON=.venv/bin/python) is made absolute here, against the directory the
# command was typed in: the generators run from 02_output/figures, where it would not resolve.
case "$PY" in
  /*|[A-Za-z]:*) ;;
  */*|*'\'*) PY="$(pwd)/$PY" ;;
esac
RECORD_STUDY="${RECORD_STUDY:-full}"
AMOC_FILE="$ROOT/00_data/amoc/sg_index_hadisst.txt"   # downloaded once, before the run

case "$RECORD_STUDY" in
  full|cached) ;;
  *) echo "[ERROR] RECORD_STUDY must be 'full' (the default) or 'cached', not '$RECORD_STUDY'."
     exit 1 ;;
esac

# The interpreter is resolved here rather than discovered mid-run. On Windows, PATH often
# puts the Microsoft Store alias ahead of the real install; it is not an interpreter and
# it fails intermittently, so a probe can pass and a later call still die.
PY_RESOLVED="$(command -v "$PY" 2>/dev/null || true)"
case "$PY_RESOLVED" in
  *WindowsApps*)
    echo "[ERROR] '$PY' resolves to the Windows Store alias:"
    echo "          $PY_RESOLVED"
    echo "        That alias is not an interpreter. Rerun as"
    echo "          PYTHON=/c/path/to/python.exe bash run_all.sh"
    echo "        or turn it off in Settings > Apps > App execution aliases."
    exit 1 ;;
esac
if ! "$PY" -c "import sys" >/dev/null 2>&1; then
  echo "[ERROR] '$PY' is not a working Python interpreter."
  echo "        Rerun with PYTHON=/path/to/python set."
  exit 1
fi

# The mode is set by the directory alone. A manuscript/ that lacks a file is an error further
# on, never a reason to switch to the manifest.
if [ -d "$MANU" ]; then
  HAVE_MANU=1
  echo "manuscript/ present: paper.tex is read, and 02_output/paper_manifest.json rewritten from it"
else
  HAVE_MANU=0
  echo "manuscript/ not present: using 02_output/paper_manifest.json"
  if [ ! -f "$MANIFEST" ]; then
    echo "[ERROR] 02_output/paper_manifest.json is missing too: nothing says what the paper displays."
    exit 1
  fi
fi

mkdir -p "$FIGOUT" "$RAW" "$STUDY" "$ROOT/02_output/tables" "$ROOT/02_output/values"
[ "$HAVE_MANU" = 1 ] && mkdir -p "$MANFIG" "$MANU/tables"

# The marker fixes the instant the run began. Step 4/5 compares every exhibit against it,
# so "present" can never be mistaken for "produced": a committed figure whose generator
# has since been removed is older than the marker and is reported as not made.
MARKER="$ROOT/02_output/.run_started"
: > "$MARKER"

# Every generator's stdout is captured into raw_runs/<key>.out, and step 2/5 reads those
# files rather than executing anything. The captures, and the record-study outputs, are
# cleared here so that nothing from an earlier run can be read as belonging to this one.
rm -f "$RAW"/*.out
rm -f "$STUDY"/*.json
rm -rf "$STUDY/draws"

FAILED=""
TIMINGS=""
STUDY_CHECK="not run"

run() {  # run <key> <script>
  local key="$1" script="$2"
  local out="$RAW/$key.out"
  echo "-- [$key] $(basename "$script") --"
  local t_step=$SECONDS
  ( cd "$FIGOUT" && "$PY" "$script" ) | tee "$out"
  local rc=${PIPESTATUS[0]}
  TIMINGS="$TIMINGS
  $(printf '%-24s %5ds' "$key" $((SECONDS - t_step)))"
  if [ $rc -ne 0 ]; then
    rm -f "$out"          # a partial capture must not be read as a result
    echo "   [ERROR] $key failed (rc=$rc)"
    FAILED="$FAILED $key"
    return 1
  fi
  return 0
}

t0=$SECONDS
echo "=== 1/5  Generators (each executed once; stdout captured to raw_runs/) ==="
# Order is a dependency order. Two groups of scripts read the AMOC series (local copy,
# md5-checked): the record study (identification.py, and calib82.py through it) and
# ews_calibration.py, whose estimates four later scripts read back out of its capture.
# Everything else is self-contained.
run bifurcation             "$CODE/03_figures/fig_bifurcation_intro.py"
# The introduction's figure: closed forms only, no input, no seed.
run intro_two_boundaries    "$CODE/03_figures/fig_intro_two_boundaries.py"
run fig_forest_tc           "$CODE/03_figures/fig_forest_tc.py"
run calibration_ci          "$CODE/01_calibration/calibration_ci.py"
run eps_inf_ci              "$CODE/01_calibration/eps_inf_ci.py"
run bhp_bound               "$CODE/01_calibration/bhp_bound.py"

# --- the record-detectability study (ssec:record and app:record in paper.tex) ---
# Eight scripts write eight JSON files into 02_output/record_study/; record_detectability.py
# reads them. About 2.5 hours on 6 worker processes, so RECORD_STUDY=cached copies the
# shipped outputs instead. In the default mode the recomputed files are compared with the
# shipped ones and the result is printed in the summary.
echo
echo "-- record study: RECORD_STUDY=$RECORD_STUDY --"
if [ "$RECORD_STUDY" = "cached" ]; then
  cp "$ROOT"/00_data/record_study/*.json "$STUDY"/
  echo "   copied the eight shipped study outputs from 00_data/record_study/ (not recomputed)"
  STUDY_CHECK="not recomputed (RECORD_STUDY=cached)"
else
  # identification.py reads the AMOC series; calib82.py reads identification.py's outputs
  # and mle_ditlevsen.py's critical values. The other five simulate and read no data.
  run study_identification    "$CODE/05_record_study/identification.py"
  run study_puissance         "$CODE/05_record_study/puissance.py"
  run study_mle_ditlevsen     "$CODE/05_record_study/mle_ditlevsen.py"
  run study_calib82           "$CODE/05_record_study/calib82.py"
  run study_estimateurs       "$CODE/05_record_study/estimateurs.py"
  run study_fenetres          "$CODE/05_record_study/fenetres.py"
  run study_horizon_np        "$CODE/05_record_study/horizon_np.py"
  run study_horizon_attente   "$CODE/05_record_study/horizon_attente.py"
  STUDY_CHECK="$("$PY" "$CODE/05_record_study/compare_shipped.py" | tee "$RAW/study_compare.out" | tail -n 1)"
  cat "$RAW/study_compare.out"
fi

# Reads the study outputs in 02_output/record_study; it simulates nothing.
run record_detectability    "$CODE/01_calibration/record_detectability.py"
# fig_record_power.py imports record_detectability.compute() rather than re-reading the
# conversion, so the figure and the rec* macros are two views of one evaluation.
run fig_record_power        "$CODE/03_figures/fig_record_power.py"
# crossover.py must precede fig_crossover.py: the figure imports it rather than
# recomputing anything, so that the exhibit and the five macros are two views of one
# evaluation. The import makes the order a real dependency, not a convention.
run crossover               "$CODE/01_calibration/crossover.py"

# --- the AMOC recovery rate, and the four scripts that depend on it ---
if run ews_calibration "$CODE/01_calibration/ews_calibration.py"; then
  run reduced_form_scc        "$CODE/01_calibration/reduced_form_scc.py"
  run noise_reversal_frontier "$CODE/02_solvers/noise_reversal_frontier.py"
  # fig_crossover.py shades the proximities below eps_dyn, which needs the recovery rate.
  run fig_crossover           "$CODE/03_figures/fig_crossover.py"
  # The price while the budget is being spent. Reads theta and M* through eps_dyn.py;
  # deterministic; about 30 s with its own convergence check.
  run moving_budget           "$CODE/02_solvers/moving_budget.py"
else
  # The consequence is stated here rather than left to be discovered at compile time.
  echo
  echo "   [ERROR] the AMOC SST record could not be read from"
  echo "             $AMOC_FILE"
  echo "           (missing, or md5 different from 00_data/amoc/PROVENANCE.md; the message"
  echo "           of ews_calibration.py above says which). Download it once with"
  echo "             python3 00_data/amoc/fetch_pik_sg_index.py"
  echo
  echo "   CONSEQUENCE, stated now so it is not discovered later:"
  echo "     * the recovery rate and the detrended variance were NOT estimated;"
  echo "     * the four scripts that read them were NOT run -- reduced_form_scc,"
  echo "       noise_reversal_frontier, fig_crossover, moving_budget;"
  if [ "$RECORD_STUDY" = "full" ]; then
    echo "     * the record study could not read the series either (identification.py,"
    echo "       then calib82.py); RECORD_STUDY=cached reuses its shipped outputs;"
  fi
  # The count is read from counts.json, written by make_tables_values.py with values.tex.
  echo "     * this run therefore STOPS BEFORE WRITING values.tex: collect_computed.py"
  echo "       exits on the first missing capture, so values.tex is still the one the"
  echo "       PREVIOUS run wrote. Do not read it as the product of this run."
  if [ -f "$COUNTS" ]; then
    echo "     * the macros marked [requires network] -- $("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));print(f\"{c['computed_network']} of {c['computed']}\")" "$COUNTS" 2>/dev/null || echo '?? of ??') at the last full run,"
    echo "       including ones the body of the paper quotes -- are the ones this run could"
    echo "       not recompute;"
  else
    echo "     * the macros marked [requires network] are the ones this run could not"
    echo "       recompute -- this package has not completed a full run here, so the count"
    echo "       is not known -- including ones the body of the paper quotes;"
  fi
  echo "     * the manuscript cannot be compiled to its published values from this run."
  echo
  echo "   What this run still reproduced: four of the five figures, and the macros that"
  echo "   take no calibrated input."
  echo
  echo "   The data availability statement in README.md explains why this one series is"
  echo "   downloaded rather than shipped, and how the index can be rebuilt from HadISST"
  echo "   if the URL has stopped resolving for good."
  echo
  FAILED="$FAILED ews-dependents"
fi

# The numerical-methods appendix: collapse of the amplitude, slopes of the fold and
# pitchfork prices, error of the leading law, the transcritical constant, and the
# sensitivity of F for the AMOC. It runs AFTER ews_calibration because the last part reads
# theta from its capture (through eps_dyn.py); without it, that part says so and emits
# nothing, and the others still run. Deterministic; about 40 s.
run validation_bvp          "$CODE/02_solvers/validation_bvp.py"

# Steps 2/5 and 3/5 have no partial result worth keeping. If either fails, the run stops
# there, with a non-zero code, and writes nothing more: values.tex, the tables, the figures
# and README.md all stay exactly as the previous run left them. Going on would rebuild them
# from the PREVIOUS computed.json or values_generated.tex, and a failed run would then
# produce the committed outputs and pass for a successful one.
stop_run() {  # stop_run <step> <what failed>
  rm -f "$MARKER"
  echo
  echo "============================================================"
  echo "STOPPED at step $1: $2"
  echo "Nothing was written after this point: values.tex, the tables, the figures and"
  echo "README.md are those of the PREVIOUS run, not of this one."
  echo "Failed steps:$FAILED"
  echo "============================================================"
  exit 1
}

echo
echo "=== 2/5  Collect the printed results into computed.json ==="
if ! "$PY" "$CODE/04_tables_values/collect_computed.py"; then
  FAILED="$FAILED collect"
  stop_run "2/5" "collect_computed.py failed (see its message above); 02_output/computed.json was not rewritten"
fi

echo
echo "=== 3/5  Write values.tex and the tables, and feed the manuscript ==="
if ! "$PY" "$CODE/04_tables_values/make_tables_values.py"; then
  FAILED="$FAILED values"
  stop_run "3/5" "make_tables_values.py failed (see its message above); nothing was copied into manuscript/"
fi

# The files the manuscript reads. The run writes them under 02_output/, where they are
# versioned; with manuscript/ present they are copied there too, under these names.
FIGS="fig_intro_two_boundaries fig_bifurcation_intro fig_record_power fig_crossover fig_forest_tc"
COPIES="02_output/values/values_generated.tex:manuscript/values.tex
02_output/tables/disagreements.tex:manuscript/tables/disagreements.tex
02_output/tables/eps_dyn.tex:manuscript/tables/eps_dyn.tex"
for f in $FIGS; do
  COPIES="$COPIES
02_output/figures/$f.pdf:manuscript/figures/$f.pdf"
done

# A figure is copied only if this run wrote it: 02_output/figures keeps the files of
# earlier runs (and the committed ones), and a copy would make an old figure look new to
# step 4/5. Without the manuscript nothing is copied, and step 4/5 checks 02_output/.
for f in $FIGS; do
  if [ ! "$FIGOUT/$f.pdf" -nt "$MARKER" ]; then
    echo "   [WARN] $f.pdf was not regenerated by this run; not copied"
  fi
done
if [ "$HAVE_MANU" = 1 ]; then
  # The manuscript is fed BEFORE the manifest and the inventory are written: both read what
  # this run has just written, not what it inherited.
  for pair in $COPIES; do
    src="${pair%%:*}"; dst="${pair#*:}"
    case "$src" in
      02_output/figures/*) [ "$ROOT/$src" -nt "$MARKER" ] || continue ;;
    esac
    cp "$ROOT/$src" "$ROOT/$dst"
  done
  "$PY" "$CODE/04_tables_values/paper_manifest.py" --write \
    || { echo "   [ERROR] paper_manifest.py failed"; FAILED="$FAILED manifest"; }
  # The two copies of every file must be the same bytes: the public package ships the
  # 02_output/ copies, and the paper is compiled from the manuscript/ ones.
  ndiff=0
  for pair in $COPIES; do
    src="${pair%%:*}"; dst="${pair#*:}"
    if ! cmp -s "$ROOT/$src" "$ROOT/$dst"; then
      echo "   [ERROR] $dst differs from $src"
      ndiff=$((ndiff + 1))
    fi
  done
  if [ "$ndiff" -eq 0 ]; then
    echo "   manuscript/ and 02_output/ copies identical, byte for byte ($(echo "$COPIES" | wc -l | tr -d ' ') files)"
  else
    FAILED="$FAILED copies"
  fi
else
  echo "   manuscript/ not present: nothing copied; the outputs stay under 02_output/"
fi

# The generated sections of README.md are written from this run, never typed by hand.
"$PY" "$CODE/04_tables_values/make_inventory.py" \
  || { echo "   [ERROR] README generation failed"; FAILED="$FAILED readme"; }

echo
echo "=== 4/5  Exhibit coverage: paper.tex is the authority ==="
"$PY" "$CODE/check_exhibits.py" "$MARKER" \
  || { echo "   [ERROR] an exhibit paper.tex displays was not produced by this run"; \
       FAILED="$FAILED exhibits"; }

echo
echo "=== 5/5  Compile the manuscript ==="
if [ "$HAVE_MANU" = 0 ]; then
  echo "   SKIPPED: manuscript/ is not part of this package. values.tex, the tables and the"
  echo "   figures it reads are under 02_output/ (see README.md)."
# An incomplete run must not leave a pdf that looks like the paper. If anything above
# failed, the document is still built, under a name that says what it is, and
# manuscript/paper.pdf is left alone.
elif [ -n "$FAILED" ]; then
  echo "   Earlier steps failed, so manuscript/paper.pdf is NOT written by this run."
  ( cd "$ROOT/manuscript" && latexmk -pdf -interaction=nonstopmode \
      -jobname=paper_INCOMPLETE paper.tex >/dev/null 2>&1 )
  if [ -f "$ROOT/manuscript/paper_INCOMPLETE.pdf" ]; then
    echo "   Built manuscript/paper_INCOMPLETE.pdf instead. It is NOT the paper: it is"
    echo "   what this run could assemble, and undefined macros show as '??' in it."
  else
    echo "   latexmk could not build even the incomplete document (see manuscript/paper_INCOMPLETE.log)."
  fi
  if [ -f "$ROOT/manuscript/paper.pdf" ]; then
    echo "   NOTE: manuscript/paper.pdf exists and predates this run."
    echo "         It was NOT produced by this run. Do not report it as reproduced."
  fi
else
  # A clean run owns paper.pdf, and clears any incomplete document left by a previous one.
  rm -f "$ROOT/manuscript/paper_INCOMPLETE.pdf" "$ROOT/manuscript/paper_INCOMPLETE.log"
  if ( cd "$ROOT/manuscript" && latexmk -pdf -interaction=nonstopmode paper.tex >/dev/null 2>&1 ); then
    echo "   manuscript/paper.pdf OK"
  else
    echo "   [ERROR] latexmk failed (see manuscript/paper.log)"
    FAILED="$FAILED compile"
  fi
fi

echo
echo "============================================================"
echo "Done in $((SECONDS - t0))s.  Failed steps:${FAILED:- none}"
echo "Record study: $STUDY_CHECK"
echo "Per-step wall clock:${TIMINGS}"
echo "Per-macro diff: 02_output/values/VALUES_DIFF.md"
echo "============================================================"
rm -f "$MARKER"
[ -z "$FAILED" ] || exit 1
