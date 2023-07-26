__all__ = ["read_trange", "read_num_intervals", "read_data"]

import astropy.units as u
import numpy as np
import h5py as h5
import lib


def read_trange(interval, dtype=str):
    h5f = h5.File(lib.data_file, "r")
    return h5f["/trange"][interval, :].astype("datetime64[ns]").astype(dtype)


def read_num_intervals():
    h5f = h5.File(lib.data_file, "r")
    return h5f["/trange"].shape[0]


def read_data(where):
    h5f = h5.File(lib.data_file, "r")
    data = h5f[where][:]
    if "unit" in h5f[where].attrs:
        data *= u.Unit(h5f[where].attrs["unit"])

    return data
