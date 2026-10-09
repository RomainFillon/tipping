#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
paper_manifest.py — what the manuscript displays, read from paper.tex, so that the package
runs with or without the manuscript next to it.

  COMPUTES   nothing; it reads the manuscript
  READS      manuscript/paper.tex and manuscript/values.tex, when manuscript/ is present;
             02_output/paper_manifest.json otherwise
  WRITES     02_output/paper_manifest.json (with --write, only when manuscript/ is present)
  FEEDS      check_exhibits.py (step 4/5) and make_inventory.py (step 3/5)

Two consumers need to know what the paper shows: check_exhibits.py, which requires every
exhibit to have been produced by the run, and make_inventory.py, which writes the map of
exhibits in README.md. Both used to read paper.tex directly. The public package does not
ship the manuscript, so the facts they need are written once, by this script, to
02_output/paper_manifest.json: the title, the exhibits in the order the document uses them,
every figure and table with its label and section, and the macros the text cites.

There are two modes, and the choice is made by the DIRECTORY alone:
  * manuscript/ present  -> load() reads paper.tex, exactly as the consumers did before, and
                            run_all.sh rewrites the manifest from it at every run;
  * manuscript/ absent   -> load() reads the committed manifest.
A manuscript/ that exists but lacks paper.tex is an error, never a reason to fall back on
the manifest: a missing file must not silently switch the package to another input.

The manifest is a GENERATED file: versioned, never edited by hand.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANU = ROOT / "manuscript"
PAPER = MANU / "paper.tex"
MANU_VALUES = MANU / "values.tex"
MANIFEST = ROOT / "02_output" / "paper_manifest.json"

# Where each file the manuscript reads is written by the run, under 02_output/. The
# manuscript's copies are byte-identical (run_all.sh checks it when manuscript/ is present).
OUTPUT_OF = {"values.tex": "02_output/values/values_generated.tex"}
OUTPUT_DIRS = {"figures/": "02_output/figures/", "tables/": "02_output/tables/"}


def manuscript_present() -> bool:
    return MANU.is_dir()


def output_path(rel: str) -> Path:
    """The file under 02_output/ that the run writes for the manuscript's file `rel`."""
    if rel in OUTPUT_OF:
        return ROOT / OUTPUT_OF[rel]
    for pre, out in OUTPUT_DIRS.items():
        if rel.startswith(pre):
            return ROOT / (out + rel[len(pre):])
    raise SystemExit(f"[ERROR] paper_manifest.py: no 02_output/ location is declared for "
                     f"manuscript/{rel}.")


def strip_comments(txt: str) -> str:
    """Drop % comments AND \\begin{comment} blocks, so that what is read is the compiled
    document only: a paragraph inside a comment block typesets as nothing while still reading,
    to a naive scan, as a live caller of the macros it names."""
    txt = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", txt, flags=re.DOTALL)
    out = []
    for line in txt.split("\n"):
        buf = []
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                break
            buf.append(ch)
        out.append("".join(buf))
    return "\n".join(out)


def strip_tex_comments(txt: str) -> str:
    r"""Drop \begin{comment} blocks and % comments, honouring the \% escape. This is the
    stripper check_exhibits.py has always used; it is kept as it was, so that the exhibits
    harvested are the ones the coverage control harvested before the manifest existed."""
    txt = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", txt, flags=re.DOTALL)
    out = []
    for line in txt.split("\n"):
        cut, i = None, 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def paper_title(txt: str) -> str:
    r"""The manuscript's own \title{...}, with its commented-out subtitle stripped."""
    m = re.search(r"\\title\{", txt)
    if not m:
        raise SystemExit(f"[ERROR] no \\title{{...}} found in {PAPER}.")
    i, depth = m.end(), 1
    while i < len(txt) and depth:
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
        i += 1
    return " ".join(strip_comments(txt[m.end():i - 1]).split())


def exhibits(txt: str):
    r"""Every \includegraphics and every \input the compiled document uses, in order and
    de-duplicated, as paths relative to manuscript/."""
    live = strip_tex_comments(txt)
    found = []
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", live):
        name = m.group(1).strip()
        rel = "figures/" + name + ("" if Path(name).suffix else ".pdf")
        found.append(("figure", name, rel))
    for m in re.finditer(r"\\input\{([^}]+)\}", live):
        name = m.group(1).strip()
        rel = name + ("" if Path(name).suffix else ".tex")
        found.append(("input", name, rel))
    seen, ordered = set(), []
    for kind, name, rel in found:
        if rel not in seen:
            seen.add(rel)
            ordered.append({"kind": kind, "name": name, "path": rel})
    return ordered


def sections(live: str):
    """(position, level, title, label) of every section heading of the compiled document."""
    out = []
    for m in re.finditer(r"\\(section|subsection|subsubsection)\*?\{([^{}]*)\}(\s*\\label\{([^}]+)\})?",
                         live):
        out.append((m.start(), m.group(1), " ".join(m.group(2).split()), m.group(4) or ""))
    app = re.search(r"\\appendix\b", live)
    return out, (app.start() if app else None)


def floats(live: str):
    """The figures and tables of the compiled document, in order, numbered as LaTeX numbers
    them, each with the section and subsection it sits in."""
    heads, app = sections(live)
    out, n = [], {"figure": 0, "table": 0}
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", live, re.DOTALL):
        kind, body = m.group(1), m.group(2)
        n[kind] += 1
        labels = [l for l in re.findall(r"\\label\{([^}]+)\}", body) if l.startswith(("fig:", "tab:"))]
        sec = sub = None
        for pos, level, title, label in heads:
            if pos > m.start():
                break
            if level == "section":
                sec, sub = (title, label), None
            elif level == "subsection":
                sub = (title, label)
        out.append(dict(
            kind=kind, num=n[kind], label=labels[0] if labels else "",
            appendix=bool(app is not None and m.start() > app),
            section=sec[0] if sec else "", section_label=sec[1] if sec else "",
            subsection=sub[0] if sub else "", subsection_label=sub[1] if sub else "",
            graphics=re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body),
            inputs=re.findall(r"\\input\{tables/([^}]+)\}", body),
            commands=sorted(set(re.findall(r"\\([A-Za-z]+)", body)))))
    return out


def build() -> dict:
    """The manifest, read from manuscript/paper.tex (and manuscript/values.tex for the
    list of cited macros). Requires manuscript/."""
    if not PAPER.exists():
        raise SystemExit(f"[ERROR] manuscript/ is present but {PAPER} is not. The package reads "
                         "the manuscript whenever the manuscript/ directory exists; it falls back "
                         "on 02_output/paper_manifest.json only when the whole directory is absent.")
    txt = PAPER.read_text(encoding="utf-8")
    live = strip_comments(txt)
    commands = set(re.findall(r"\\([A-Za-z]+)", live))
    table_inputs = sorted(set(re.findall(r"\\input\{tables/([^}]+)\}", live)))
    cited = set(commands)
    for t in table_inputs:
        cited |= set(re.findall(r"\\([A-Za-z]+)",
                                strip_comments((MANU / "tables" / t).read_text(encoding="utf-8"))))
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", MANU_VALUES.read_text(encoding="utf-8"))) \
        if MANU_VALUES.exists() else set()
    return {
        "_generated": "GENERATED by 01_code/04_tables_values/paper_manifest.py from manuscript/paper.tex. "
                      "Do not edit: run_all.sh rewrites it whenever manuscript/ is present, and reads it "
                      "when manuscript/ is absent.",
        "title": paper_title(txt),
        "exhibits": exhibits(txt),
        "floats": floats(live),
        "commands": sorted(commands),
        "table_inputs": table_inputs,
        "macros_cited": sorted(cited & defined),
    }


def load() -> dict:
    """The manifest: built from paper.tex when manuscript/ is present, read from the committed
    file when manuscript/ is absent."""
    if manuscript_present():
        return build()
    if not MANIFEST.exists():
        raise SystemExit(f"[ERROR] manuscript/ is absent and {MANIFEST} is missing: nothing says "
                         "what the paper displays.")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def main() -> int:
    if "--write" not in sys.argv[1:]:
        print("usage: paper_manifest.py --write   (requires manuscript/)", file=sys.stderr)
        return 2
    if not manuscript_present():
        print("[ERROR] paper_manifest.py --write reads manuscript/, which is not present.")
        return 1
    m = build()
    MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{MANIFEST.relative_to(ROOT).as_posix()}: {len(m['exhibits'])} exhibits, "
          f"{len(m['floats'])} floats, {len(m['macros_cited'])} cited macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
