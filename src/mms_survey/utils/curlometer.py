__all__ = ["curlometer"]

import astropy.units as u
import numpy as np
import xarray as xr
from tvolib.numeric import curlometer as np_curlometer


def curlometer(
    Q_name: str,
    Q1: xr.DataArray,
    Q2: xr.DataArray,
    Q3: xr.DataArray,
    Q4: xr.DataArray,
    R1: xr.DataArray,
    R2: xr.DataArray,
    R3: xr.DataArray,
    R4: xr.DataArray,
):
    # Sanity checks
    Q_unit = Q1.attrs["units"]
    R_unit = R1.attrs["units"]
    same_Q_unit = (
        Q_unit == Q2.attrs["units"] == Q3.attrs["units"] == Q4.attrs["units"]
    )
    same_R_unit = (
        R_unit == R2.attrs["units"] == R3.attrs["units"] == R4.attrs["units"]
    )
    assert (
        same_Q_unit and same_R_unit
    ), "Input quantities must have compatible units"

    # Interpolate everything to Q1 time
    components = ["x", "y", "z"]
    kw = dict(time=Q1.time, kwargs=dict(fill_value=np.nan))
    Q1 = Q1.sel(space=components)
    Q2 = Q2.interp(**kw).sel(space=components)
    Q3 = Q3.interp(**kw).sel(space=components)
    Q4 = Q4.interp(**kw).sel(space=components)
    kw = dict(ephemeris_time=Q1.time, kwargs=dict(fill_value=np.nan))
    R1 = R1.interp(**kw).sel(space=components)
    R2 = R2.interp(**kw).sel(space=components)
    R3 = R3.interp(**kw).sel(space=components)
    R4 = R4.interp(**kw).sel(space=components)

    # Calculate
    Q_bc = 0.25 * (Q1 + Q2 + Q3 + Q4).values
    R_bc = 0.25 * (R1 + R2 + R3 + R4).values
    clm = np_curlometer(
        Q1.values,
        Q2.values,
        Q3.values,
        Q4.values,
        R1.values,
        R2.values,
        R3.values,
        R4.values,
    )
    attrs = dict(units=str(u.Unit(Q_unit) / u.Unit(R_unit)))
    return xr.Dataset(
        data_vars={
            "R_bc": (["time", "space"], R_bc, {"units": R_unit}),
            f"{Q_name}_bc": (["time", "space"], Q_bc, {"units": Q_unit}),
            f"curl_{Q_name}": (["time", "space"], clm["curl_Q"], attrs),
            f"div_{Q_name}": (["time"], clm["div_Q"], attrs),
        },
        coords={"time": Q1.time.values, "space": ["x", "y", "z"]},
    )
