# record_study — shipped outputs of the record-detectability study

These eight JSON files are the outputs of the record-detectability study behind the
subsection "What the record can establish" (`ssec:record`), its appendix (`app:record`) and
the figure `fig_record_power.pdf`. They are written by the eight scripts of
`01_code/05_record_study/`, which `run_all.sh` runs by default; this copy is what those scripts
returned when the paper was computed.

They serve two purposes:

1. **A check.** In the default mode (`RECORD_STUDY=full`), `run_all.sh` recomputes the study
   into `02_output/record_study/` and `01_code/05_record_study/compare_shipped.py` compares each
   recomputed file with the copy here, byte for byte after normalising line endings. The verdict
   is printed in the run's summary.
2. **A shortcut.** The study takes about 2.5 hours on 6 worker processes. With
   `RECORD_STUDY=cached bash run_all.sh`, the run copies these files into
   `02_output/record_study/` instead of recomputing them; every other step still runs.

`01_code/01_calibration/record_detectability.py` reads seven of them from
`02_output/record_study/` and emits the `\rec*` macros of `values.tex`; `identification.json`
is read only by `calib82.py`.

| file | produced by | reads the AMOC series? | seeds | md5 (LF line endings) |
|---|---|---|---|---|
| identification.json | `05_record_study/identification.py` | yes, via `record_pipeline.load()` | 42 | `db94e35693ef381c03e9f620e83bd45b` |
| puissance.json | `05_record_study/puissance.py` | no | 1, 2, … | `44fc8e154b9ae1e5aa0360fcd98be818` |
| mle_ditlevsen.json | `05_record_study/mle_ditlevsen.py` | no | 6000, 6001, … | `6f8f924adbc48cc59e63a36a964f56bf` |
| calib82.json | `05_record_study/calib82.py` | through identification.json | 8000, 8001, … | `651883335ec4c32e7f242e28407ac731` |
| estimateurs.json | `05_record_study/estimateurs.py` | no | 1300, 1301, … | `2b7e72590aa9c778971fd67869451e89` |
| fenetres.json | `05_record_study/fenetres.py` | no | 2000, 2001, … | `f52458f2a72ed5c452fb51f4b48724fa` |
| horizon_np.json | `05_record_study/horizon_np.py` | no | 5000, 5001, … | `81579443225ccb90be7ce3bcf6f0ec06` |
| horizon_attente.json | `05_record_study/horizon_attente.py` | no | 7000, 7001, … | `4fe8f6c9cdf4deb4b4ee7ce15526a75c` |

The file names, and some keys inside the files (`lineaire`, `nul`, `libre`, …), are in French,
the language the study was first written in; they are kept so that the shipped copies stay
byte-identical to what the scripts write.

A change to any of these files is a change to the study: it has to be made in
`01_code/05_record_study/`, rerun, and copied here, never edited by hand.
