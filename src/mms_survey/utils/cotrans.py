import quaternion as np_q
import xarray as xr


def quaternion_rotate(V: xr.DataArray, Q: xr.DataArray, inverse: bool = False):
    r"""
    Rotate vector V given quaternion Q

    Parameters
    ----------
    V: xarray.DataArray
        Vector to rotate, must have 3 spatial dimensions and be in the
        same coordinates as Q depending on whether this is an inverse
        operation.
    Q: xarray.DataArray
        Quaternion data
    inverse: bool
        If true, V_rotated = Q.V.Q^\dagger, else V_rotated = Q^\dagger.V.Q

    Return
    ------
    V_rotated: xarray.DataArray
        Rotated vector with the same attributes as input V
    """
    if inverse:
        in_coord = "TO_COORDINATE_SYSTEM"
        out_coord = "COORDINATE_SYSTEM"
    else:
        in_coord = "COORDINATE_SYSTEM"
        out_coord = "TO_COORDINATE_SYSTEM"

    # ---- Sanity checks
    assert (
        "space" in V.coords
    ), "Input vector must have spatial coordinates"
    assert (
        "quaternion" in Q.coords
    ), "Input quaternion must have quaternion coordinates"
    assert (
        V.attrs["COORDINATE_SYSTEM"] == Q.attrs[in_coord]
    ), "Inputs must be in the same coordinate system"

    # numpy-quaternion uses (w, i, j, k) representation
    Q = Q.reindex({"quaternion": ["w", "x", "y", "z"]})
    V = V.reindex({"space": ["x", "y", "z"]})

    # Extract one spatial coordinate, assign it `w`, and concatenate into V
    Vw = xr.zeros_like(V.sel(space="x")).assign_coords(space="w")
    V = xr.concat([Vw, V], dim="space")

    # This rotation step forces data into memory
    _Q = np_q.as_quat_array(Q)
    if inverse:
        _Q = _Q.conjugate()
    _V = np_q.as_quat_array(V)
    _V_rotated = _Q * _V * _Q.conjugate()
    V_rotated = xr.zeros_like(V)
    V_rotated[...] = np_q.as_float_array(_V_rotated)

    V_rotated.attrs["COORDINATE_SYSTEM"] = Q.attrs[out_coord]
    return V_rotated.sel(space=["x", "y", "z"])
