__all__ = ["background"]

import astropy.units as u
import numpy as np


def f_omni_background(E, species="ion", factor=1):
    f_unit = u.Unit("cm-2 s-1 sr-1")
    gaps = dict(
        ion=np.array([28.3, 76.8]) * u.keV,
        elc=np.array([27.5, 65.9]) * u.keV,
    )
    bg_levels = dict(
        ion=np.array([3e4, 4e3]) * f_unit,
        elc=np.array([6e5, 8e2]) * f_unit,
    )

    f = 1e-4 * np.ones(E.shape) * f_unit
    f[E <= gaps[species][0]] = factor * bg_levels[species][0]
    f[E >= gaps[species][1]] = factor * bg_levels[species][1]
    if species == "elc":
        f[(7e-1 * u.keV <= E) & (E <= gaps[species][0])] = (
            factor * 1e5 * f_unit
        )

    return f
