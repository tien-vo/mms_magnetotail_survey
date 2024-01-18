import astropy.constants as c
import astropy.units as u
import numpy as np
import xarray as xr

from mms_survey.calc.integrator import integrate_omni
from mms_survey.utils.xarray import to_np


def background_threshold(species: str, W: xr.DataArray):
    f_unit = u.Unit("cm-2 s-1 sr-1")
    gaps = dict(
        ion=np.array([28.3, 76.8]) * u.keV,
        elc=np.array([27.5, 65.9]) * u.keV,
    )
    bg_levels = dict(
        ion=np.array([5e4, 4e3]) * f_unit,
        elc=np.array([3e5, 8e2]) * f_unit,
    )

    W = to_np(W)
    f_bg = 1e-4 * np.ones_like(W.value) * f_unit
    f_bg[W <= gaps[species][0]] = bg_levels[species][0]
    f_bg[W >= gaps[species][1]] = bg_levels[species][1]
    if species == "elc":
        low_W_range = (7e-1 * u.keV <= W) & (W <= gaps[species][0])
        f_bg[low_W_range] = 1e5 * f_unit

    return f_bg


def apply_background_filter(
    species: str,
    f3d: xr.DataArray,
    W: xr.DataArray,
    theta: xr.DataArray,
    phi: xr.DataArray,
    V_sc: xr.DataArray = None,
    cutoff_energy: u.Quantity = 0 * u.eV,
):
    # Calculate filter based on omni-distribution
    f1d = to_np(integrate_omni(species, f3d, W, theta, phi, V_sc=V_sc))
    if species == "ion":
        f1d_sorted = np.take_along_axis(f1d, np.argsort(f1d, axis=1), axis=1)
        f1d = f1d - np.nanmean(f1d_sorted[:, :5], axis=1)[:, np.newaxis]
    filter_condition = f1d <= background_threshold(species, W)
    if V_sc is not None:
        q = c.si.e if species == "ion" else -c.si.e
        W, V_sc = xr.broadcast(W, V_sc)
        _W = to_np(W) + q * to_np(V_sc)
        filter_condition |= _W < 0
    else:
        _W = to_np(W)
    filter_condition |= _W < cutoff_energy

    # Transpose energy up to second axis for this computation
    original_dims = f3d.dims
    dims = ("time", "energy", "zenith", "azimuth")
    f3d = f3d.transpose(*dims)
    f3d_values = to_np(f3d)
    f3d_values[filter_condition, ...] = 0.0
    f3d[...] = f3d_values
    f3d = f3d.transpose(*original_dims)

    return f3d
