"""
tcre_draw.py -- the TCRE draw of the three Monte-Carlo scripts.

calibration_ci.py, eps_inf_ci.py and fig_forest_tc.py each drew the TCRE on their own line,
as np.clip(normal(mean, sd, N), lo, hi). That clip was a censoring, not a truncation: the
draws outside [lo, hi] were moved onto the bounds and kept. The three lines are replaced by
the one function below, which the three scripts import.

The law: normal, mean DERIVATION_INPUTS["TCRE"] (AR6 best estimate) and standard deviation
DERIVATION_INPUTS["TCRE_sd"], computed in inputs_literature.py from the AR6 likely range
read as the 17th-83rd centiles. The only constraint is TCRE > 0, imposed by REJECTION:
draws <= 0 are discarded and replaced, so the function returns exactly n positive draws.
"""
import numpy as np

from inputs_literature import DERIVATION_INPUTS


def draw_tcre(rng, n, return_discarded=False):
    """n draws of the TCRE, degC per 1000 GtCO2, from N(TCRE, TCRE_sd) restricted to > 0.

    rng   the random source. The three scripts pass the module np.random itself, i.e. the
          global legacy RandomState they seed with np.random.seed(42) just before; nothing
          is drawn between the seed and this call, so the TCRE draws are the first n normals
          of seed 42 in all three scripts.
    n     the number of draws returned (exactly n).

    Order of the draws: n normals are drawn at once, in one call rng.normal(mean, sd, n).
    Every draw <= 0 is then replaced, in index order, by a fresh call
    rng.normal(mean, sd, k) for the k draws still <= 0, and the replacement is repeated
    until none is left. The draws that pass the first round therefore keep their place and
    their value, and the stream after this call is shifted by the number of replacements.

    With return_discarded=True, also returns (n_discarded, n_total): the number of draws
    rejected and the number drawn in all, n_total = n + n_discarded.
    """
    mean = float(DERIVATION_INPUTS["TCRE"][0])
    sd = float(DERIVATION_INPUTS["TCRE_sd"][0])
    x = rng.normal(mean, sd, n)
    n_discarded = 0
    bad = x <= 0
    while bad.any():
        k = int(bad.sum())
        n_discarded += k
        x[bad] = rng.normal(mean, sd, k)
        bad = x <= 0
    if len(x) != n or not (x > 0).all():
        raise SystemExit("[ERROR] tcre_draw.py: draw_tcre did not return %d positive draws" % n)
    if return_discarded:
        return x, (n_discarded, n + n_discarded)
    return x
