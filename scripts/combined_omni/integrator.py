__all__ = ["omni_integrate"]

import astropy.constants as c
import astropy.units as u
import numpy as np


def omni_integrate(E, f, species="ion"):
    charge = c.si.e if species == "ion" else -c.si.e
    mass = c.si.m_p if species == "ion" else c.si.m_e
    assert f.unit == "cm-2 s-1 sr-1"

    f[np.isnan(f)] = 0
    N = (
        4
        * np.pi
        * np.sqrt(mass / 2)
        * np.trapz(f * u.sr * E ** (-3 / 2), x=E, axis=1)
    ).to(u.Unit("cm-3"))
    P = (
        4
        * np.pi
        * np.sqrt(mass / 2)
        * np.trapz(f * u.sr * E ** (-1 / 2), x=E, axis=1)
    ).to(u.Unit("keV cm-3"))

    return N, P
