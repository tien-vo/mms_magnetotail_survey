__all__ = [
    "read_trange",
    "read_num_intervals",
    "read_data",
    "read_event_interval",
]

import astropy.units as u
import h5py as h5
import numpy as np

import lib


def read_trange(interval, dtype=str):
    trange = (
        np.loadtxt(lib.resource_dir / "intervals.csv", delimiter=",")
        .astype("datetime64[s]")
        .astype("datetime64[ns]")
    )
    return trange[interval, :].astype(dtype)


def read_num_intervals():
    trange = (
        np.loadtxt(lib.resource_dir / "intervals.csv", delimiter=",")
        .astype("datetime64[s]")
        .astype("datetime64[ns]")
    )
    return trange.shape[0]


def read_event_interval(event):
    trange = (
        np.loadtxt(lib.resource_dir / "intervals.csv", delimiter=",")
        .astype("datetime64[s]")
        .astype("datetime64[ns]")
    )
    events = np.loadtxt(
        lib.resource_dir / "turbulent_events.csv",
        delimiter=",",
        dtype="datetime64[ns]",
    )
    event_start = events[event, 0]
    event_stop = events[event, 1]

    idx = np.where(
        (trange[:, 0] <= event_start) & (event_stop <= trange[:, 1])
    )[0]
    if len(idx) == 0:
        return None
    else:
        return idx[0]


def read_data(where):
    h5f = h5.File(lib.data_file, "r")
    data = h5f[where][:]
    if "unit" in h5f[where].attrs:
        data *= u.Unit(h5f[where].attrs["unit"])

    return data
