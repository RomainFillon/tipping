# -*- coding: utf-8 -*-
"""
REPLICATION HEADER
  PRODUCES   00_data/emissions/rcmip_co2_world_ssp245_ssp370.csv (the shipped extract)
  FEEDS      01_code/01_calibration/emission_paths.py, which reads the extract only
  INPUTS     rcmip-emissions-annual-means-v5-1-0.csv, RCMIP protocol v5.1.0
             (Nicholls and Lewis, Zenodo, doi:10.5281/zenodo.4589756, 2021-03-09),
             md5 4044106f55ca65b094670e7577eaf9b3; NOT shipped (48 MB), see PROVENANCE.md
  SEED       none
  RUNTIME    about 5 s
  IMPLEMENTS no equation; a row selection

Not run by run_all.sh. The package ships the extract and this script, so the extract can
be rebuilt from the published file and checked against it; nothing in the default run
needs the network for these two series.

The rows kept are Variable "Emissions|CO2" (fossil and industrial plus AFOLU, the MAGICC
total), Region "World", Scenario "ssp245" and "ssp370", unit Mt CO2/yr, at the years the
file reports (annual to 2015, then 2020 and every ten years to 2500). No value is changed.
"""
import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "rcmip-emissions-annual-means-v5-1-0.csv")
OUT = os.path.join(HERE, "rcmip_co2_world_ssp245_ssp370.csv")
MD5 = "4044106f55ca65b094670e7577eaf9b3"

if not os.path.exists(FULL):
    sys.exit("[ERROR] %s not found. Download it from "
             "https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/"
             "rcmip-emissions-annual-means-v5-1-0.csv (or Zenodo 4589756)." % FULL)
with open(FULL, "rb") as fh:
    md5 = hashlib.md5(fh.read()).hexdigest()
if md5 != MD5:
    sys.exit("[ERROR] md5 of the RCMIP file is %s, expected %s" % (md5, MD5))

d = pd.read_csv(FULL)
rows = d[(d.Region == "World") & (d.Variable == "Emissions|CO2")
         & d.Scenario.isin(["ssp245", "ssp370"])]
if len(rows) != 2 or set(rows.Unit) != {"Mt CO2/yr"}:
    sys.exit("[ERROR] expected two rows in Mt CO2/yr, found:\n%s"
             % rows[["Scenario", "Unit"]])
years = [c for c in d.columns if c.isdigit()]
out = pd.DataFrame({"year": [int(y) for y in years]})
for _, r in rows.iterrows():
    out[r.Scenario] = r[years].astype(float).values
out = out.dropna(subset=["ssp245", "ssp370"], how="all")
out = out[out.year >= 2010]
out.to_csv(OUT, index=False, float_format="%.6f")
print("wrote %s, %d years, %d-%d, models: %s"
      % (OUT, len(out), out.year.min(), out.year.max(),
         dict(zip(rows.Scenario, rows.Model))))
