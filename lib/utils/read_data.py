import astropy.units as u
import numpy as np
import h5py as h5
import lib

__all__ = ["read_trange", "read_num_intervals", "read_dataset"]


def read_trange(interval, dtype=str):
    with h5.File(lib.data_file, "r") as h5f:
        trange = h5f["/trange"][interval, :].astype("datetime64[ns]").astype(dtype)

    return trange


def read_num_intervals():
    with h5.File(lib.data_file, "r") as h5f:
        N_intervals = h5f["/trange"].shape[0]

    return N_intervals


def read_dataset(where):
    with h5.File(lib.data_file, "r") as h5f:
        data = h5f[where][:]
        if "unit" in h5f[where].attrs:
            data *= u.Unit(h5f[where].attrs["unit"])
        if data.dtype == "S29":
            data = np.array(data, dtype="datetime64[ns]")

    return data
