#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
check_exhibits.py -- every exhibit paper.tex displays was produced by THIS run.

  COMPUTES   nothing; it is the coverage control of run_all.sh
  READS      02_output/paper_manifest.json's list of exhibits, through paper_manifest.load():
             built from manuscript/paper.tex when manuscript/ is present, read from the
             committed manifest when it is absent
  WRITES     nothing; one line per exhibit on stdout
  FEEDS      no exhibit; it is what certifies the others

The authority is paper.tex, not a list kept by hand. Every \includegraphics and every \input
the document actually uses is harvested (by paper_manifest.exhibits()) and resolved to a
file. With manuscript/ present, the file is the manuscript's copy, under manuscript/; without
it, the file the run writes under 02_output/. The file must exist AND be newer than the
marker written at the start of the run. Presence alone is not enough: a committed copy of a
figure whose generator has been removed would still be present, and would still be wrong.

Commented-out lines are stripped before harvesting, for the same reason the manuscript's
own audits strip them: a \includegraphics inside a % comment is not an exhibit, and a
control that counts it will demand a file nothing produces.

Usage:  python 01_code/check_exhibits.py <marker-file>
Exit:   0 if every exhibit was produced by this run, 1 otherwise.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_code" / "04_tables_values"))
import paper_manifest  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def main():
    if len(sys.argv) < 2:
        print("usage: check_exhibits.py <marker-file>", file=sys.stderr)
        return 2
    marker = Path(sys.argv[1])
    if not marker.exists():
        print(f"  [FAIL] run marker {marker} is missing; cannot tell what this run made")
        return 1
    t0 = marker.stat().st_mtime

    if paper_manifest.manuscript_present():
        def resolve(rel):
            return paper_manifest.MANU / rel
    else:
        print("manuscript/ not present: using 02_output/paper_manifest.json")
        resolve = paper_manifest.output_path
    ordered = [(e["kind"], e["name"], resolve(e["path"])) for e in paper_manifest.load()["exhibits"]]

    print("=" * 78)
    print("EXHIBIT COVERAGE -- every object paper.tex displays, produced by this run")
    print("=" * 78)

    missing, stale = [], []
    for kind, name, p in ordered:
        rel = p.relative_to(ROOT).as_posix()
        if not p.exists():
            missing.append((name, rel))
            print(f"  [MISSING ] {kind:<6} {name:<28} {rel}")
        elif p.stat().st_mtime < t0:
            stale.append((name, rel))
            print(f"  [NOT MADE] {kind:<6} {name:<28} {rel}  (older than this run)")
        else:
            print(f"  [produced] {kind:<6} {name:<28} {rel}")

    n = len(ordered)
    ok = not missing and not stale
    print("-" * 78)
    print(f"  {n - len(missing) - len(stale)}/{n} exhibits produced by this run "
          f"({sum(1 for k, _, _ in ordered if k == 'figure')} figures, "
          f"{sum(1 for k, _, _ in ordered if k == 'input')} inputs)")
    if missing:
        print("  MISSING: paper.tex displays these and nothing wrote them:")
        for name, rel in missing:
            print(f"     {name}  ->  {rel}")
    if stale:
        print("  NOT PRODUCED BY THIS RUN: the file is present but predates the run, so")
        print("  it is a committed leftover rather than something the code just made:")
        for name, rel in stale:
            print(f"     {name}  ->  {rel}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
