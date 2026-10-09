# amoc — the AMOC SST fingerprint (subpolar-gyre index), not redistributed

`sg_index_hadisst.txt` is read by two parts of the package:

- `01_code/01_calibration/ews_calibration.py`, the first link of the chain that estimates the
  recovery rate θ̂ and the variance of the overturning circulation (`sec:ews_identification`
  in `paper.tex`, the calibration appendix);
- `01_code/05_record_study/identification.py`, through `record_pipeline.load()`, the joint
  region of the two trends in the record-detectability study (`app:record`).

The file is **not shipped**: the PIK host page publishes no
licence, copyright notice or conditions of use. A replicator downloads it once, before
`run_all.sh`, with

    python 00_data/amoc/fetch_pik_sg_index.py

which writes the bytes unchanged next to this file and refuses them if their md5 differs from
the one below. `run_all.sh` makes no network call; both readers stop with an explicit message
if the file is missing or its md5 differs.

| | |
|---|---|
| File | `sg_index_hadisst.txt` |
| Series | Subpolar-gyre sea surface temperature index derived from HadISST, the AMOC fingerprint of Caesar, Rahmstorf, Robinson, Feulner and Saba (2018), *Nature* **556**, 191–196, doi:10.1038/s41586-018-0006-5 (`caesar2018observed` in the manuscript) |
| URL used (from the code) | https://www.pik-potsdam.de/~caesar/AMOC_slowdown/sg_index_hadisst.txt |
| Downloaded | 2026-10-01 (16:50 UTC), by `fetch_pik_sg_index.py`, with the URL and the `urllib` call `ews_calibration.py` used when it still downloaded the series itself |
| Size | 1 638 bytes, CRLF line endings, one header line (`# for the years 1871-2016 in K`) and 146 values |
| md5 | `c48516162322edebaaf113b2841f93be` (also `DERIVATION_INPUTS["amoc_series_md5"]` in `01_code/inputs_literature.py`, which the fetch script and both readers check) |
| sha256 | `ab1a7dc74f7b7289baf5f509262965d0281240bdbc2d625b28e99515d514890e` |
| Licence | none published by the host |

The first and last years (`\AMOCrecordFirstYear`, `\AMOCrecordLastYear`) are read from the
file's header by `ews_calibration.py`, which checks them against the number of values.

If PIK updates the file, the md5 check fails and the run stops: the numbers in the paper were
computed on the version above. The index is a documented transformation of HadISST and can
be rebuilt from it with the subpolar-gyre region of Caesar et al. (2018); the data availability
statement of README.md gives the summary statistics a rebuilt series should match.
