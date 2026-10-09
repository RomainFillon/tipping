# -*- coding: utf-8 -*-
"""
Provenance tool, NOT part of run_all.sh (like 00_data/emissions/extract_rcmip.py).

The AMOC series is not redistributed with this package: the PIK host page publishes no
licence, copyright notice or conditions of use. A replicator runs this script ONCE, before
run_all.sh. It downloads the HadISST subpolar-gyre SST index of Caesar et al. (2018) from the
PIK server, with the URL and the urllib call ews_calibration.py once used itself, writes
the bytes unchanged to sg_index_hadisst.txt next to this file, and checks their md5 against
the one recorded in PROVENANCE.md (DERIVATION_INPUTS["amoc_series_md5"]). run_all.sh then
reads the local copy and makes no network call; ews_calibration.py stops with an explicit
message if the file is missing or its md5 differs.

Usage:  python 00_data/amoc/fetch_pik_sg_index.py
"""
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "01_code"))
from inputs_literature import DERIVATION_INPUTS  # noqa: E402

URL = "https://www.pik-potsdam.de/~caesar/AMOC_slowdown/sg_index_hadisst.txt"
OUT = os.path.join(HERE, "sg_index_hadisst.txt")
EXPECTED = DERIVATION_INPUTS["amoc_series_md5"][0]

with urllib.request.urlopen(URL, timeout=30) as resp:
    data = resp.read()
md5 = hashlib.md5(data).hexdigest()
print(f"downloaded {len(data)} bytes from {URL}")
print(f"md5    {md5}   (expected {EXPECTED})")
print(f"sha256 {hashlib.sha256(data).hexdigest()}")
if md5 != EXPECTED:
    sys.exit("[ERROR] the file PIK serves today differs from the one the paper was computed on; "
             "nothing written. See PROVENANCE.md.")
with open(OUT, "wb") as f:
    f.write(data)
print(f"wrote {OUT}")
