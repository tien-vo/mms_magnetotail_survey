__all__ = ["omni_integrate"]

import astropy.constants as c
import astropy.units as u
import numpy as np


def omni_integrate(E, f, species="ion"):
    m = c.si.m_p if species == "ion" else c.si.m_e
    assert f.unit == "cm-2 s-1 sr-1"

    N_integrand = 4 * np.pi * np.sqrt(m / 2 / E) * (f / E) * u.sr
    P_integrand = 4 * np.pi * np.sqrt(m / 2 / E) * f * u.sr
    N_integrand[np.isnan(N_integrand)] = 0
    P_integrand[np.isnan(P_integrand)] = 0

    N = np.trapz(N_integrand, x=E, axis=1).to(u.Unit("cm-3"))
    P = np.trapz(P_integrand, x=E, axis=1).to(u.Unit("keV cm-3"))

    return N, P
