r"""
REPLICATION HEADER
  PRODUCES   stdout only: one line per study output, then a one-line verdict
  FEEDS      nothing; a check, printed in the summary of run_all.sh
  INPUTS     02_output/record_study/*.json (this run) and 00_data/record_study/*.json (shipped)
  SEED       none
  RUNTIME    under 1 s
  IMPLEMENTS no equation

compare_shipped.py -- did the recomputed record study return the shipped outputs?

Compares, byte for byte after normalising line endings to LF, each of the eight JSON files the
record study has just written with the copy shipped in 00_data/record_study/. The verdict is a
report, not a gate: a different BLAS or LAPACK can move the last digit of a float, and the
per-macro comparison in 02_output/values/VALUES_DIFF.md then says whether any printed value moved.

Usage: python 01_code/05_record_study/compare_shipped.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
NEW = os.path.join(PKG, "02_output", "record_study")
SHIPPED = os.path.join(PKG, "00_data", "record_study")
NAMES = ["identification", "puissance", "mle_ditlevsen", "calib82", "estimateurs",
         "fenetres", "horizon_np", "horizon_attente"]


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read().replace(b"\r\n", b"\n")


n_same = 0
for name in NAMES:
    a, b = read(os.path.join(NEW, name + ".json")), read(os.path.join(SHIPPED, name + ".json"))
    if a is None:
        verdict = "MISSING from this run"
    elif b is None:
        verdict = "no shipped copy"
    elif a == b:
        verdict = "identical"
        n_same += 1
    else:
        verdict = "DIFFERENT"
    print(f"  {name + '.json':22s} {verdict}")
print(f"{n_same} of {len(NAMES)} recomputed study outputs identical to the shipped ones")
