# emissions — CO2 emission paths SSP2-4.5 and SSP3-7.0

`rcmip_co2_world_ssp245_ssp370.csv` is read by `01_code/01_calibration/emission_paths.py`,
which `01_code/03_figures/fig_forest_tc.py` imports to compute the deadline bound on two
emission paths (`ssec:comparable` and `fig:forest_tc` in `paper.tex`, and the calibration
appendix, `ssec:window` and `sec:calibration`). It is an extract: two rows of
the published file, values unchanged.

| | |
|---|---|
| Published file | `rcmip-emissions-annual-means-v5-1-0.csv` |
| Dataset | Reduced Complexity Model Intercomparison Project (RCMIP) protocol, v5.1.0, Z. Nicholls and J. Lewis, Zenodo, 2021-03-09, doi:10.5281/zenodo.4589756 |
| Paper to cite with it | Nicholls et al. (2020), *Geosci. Model Dev.* **13**, 5175–5190, doi:10.5194/gmd-13-5175-2020 |
| Scenario source | Meinshausen et al. (2020), *Geosci. Model Dev.* **13**, 3571–3605, doi:10.5194/gmd-13-3571-2020 (extensions after 2100: fossil CO2 ramped linearly to zero by 2250, land-use CO2 phased out linearly 2100–2150) |
| URL used | https://rcmip-protocols-au.s3-ap-southeast-2.amazonaws.com/v5.1.0/rcmip-emissions-annual-means-v5-1-0.csv (the URL rcmip.org gives for phases 1–2; the same file is on Zenodo) |
| Downloaded | 2026-10-01 |
| md5 (published file) | `4044106f55ca65b094670e7577eaf9b3`, identical to the md5 Zenodo lists for the file |
| sha256 (published file) | `2af9f90c42f9baa813199a902cdd83513fff157a0f96e1d1e6c48b58ffb8b0c1` |
| Licence | CC BY-SA 4.0 (Zenodo record) |
| Rows kept | Variable `Emissions|CO2`, Region `World`, Scenario `ssp245` (model MESSAGE-GLOBIOM) and `ssp370` (model AIM/CGE), unit `Mt CO2/yr`, years 2010–2500 as reported (annual to 2015, then 2020 and every ten years) |
| Coverage | `Emissions|CO2` is the MAGICC total, fossil and industrial plus AFOLU, the same coverage as E_0 = 40 GtCO2/yr (fossil plus land use, Friedlingstein et al. 2023) |
| sha256 (extract) | `d25361166f82c8185356415a7712f6dc9fcdb9002f3aed03d3e163f9e876ae52` |
| md5 (extract) | `a38eebf195b6df37bbf66d36d472188a` |

The published file (48 MB) is not shipped. `extract_rcmip.py` rebuilds the extract from it
and checks its md5 first; it is not part of `run_all.sh`.

A newer RCMIP record exists (phase 3, protocol 1.0.0, doi:10.5281/zenodo.17660566). It is
not used: the paper's paths are the CMIP6 SSP2-4.5 and SSP3-7.0 with the extensions of
Meinshausen et al. (2020), which is what v5.1.0 carries.

## Licence of this extract

`rcmip_co2_world_ssp245_ssp370.csv` is an adaptation (two rows, values unchanged) of
`rcmip-emissions-annual-means-v5-1-0.csv`, from the Reduced Complexity Model Intercomparison
Project (RCMIP) protocol v5.1.0 by Zebedee Nicholls and Jared Lewis
(doi:10.5281/zenodo.4589756), licensed under the Creative Commons Attribution-ShareAlike 4.0
International licence (https://creativecommons.org/licenses/by-sa/4.0/). The extract is
redistributed under the same licence, CC BY-SA 4.0, and not under the licence of the code.
