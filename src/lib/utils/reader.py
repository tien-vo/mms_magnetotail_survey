__all__ = ["read_trange", "read_num_intervals", "read_data"]

import astropy.units as u
import numpy as np
import h5py as h5
import lib


def read_trange(interval, dtype=str):
    trange = np.loadtxt(
        lib.resource_dir / "intervals.csv",
        delimiter=",").astype("datetime64[s]").astype("datetime64[ns]")
    return trange[interval, :].astype(dtype)


def read_num_intervals():
    trange = np.loadtxt(
        lib.resource_dir / "intervals.csv",
        delimiter=",").astype("datetime64[s]").astype("datetime64[ns]")
    return trange.shape[0]


def read_data(where):
    h5f = h5.File(lib.data_file, "r")
    data = h5f[where][:]
    if "unit" in h5f[where].attrs:
        data *= u.Unit(h5f[where].attrs["unit"])

    return data
