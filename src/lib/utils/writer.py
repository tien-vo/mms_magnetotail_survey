__all__ = ["write_data"]

import astropy.units as u
import numpy as np
import h5py as h5
import lib


def write_data(probe, interval, instrument, data, where=None):

    fname = f"{lib.h5_dir}/mms{probe}/{instrument}/interval_{interval}.h5"
    h5f = h5.File(fname, "a")

    if isinstance(data, dict):
        for key, value in data.items():
            write_data(probe, interval, instrument, value, where=key)
    elif isinstance(data, np.ndarray):
        if where in h5f:
            del h5f[where]
        if isinstance(data, u.Quantity):
            h5d = h5f.create_dataset(where, data=data.value)
            h5d.attrs["unit"] = str(data.unit)
        else:
            h5f.create_dataset(where, data=data)
