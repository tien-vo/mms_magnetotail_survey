import os

import h5py as h5
import astropy.units as u
from pathos.pools import ProcessPool as Pool

import lib
from lib.load import omni
from lib.utils import read_num_intervals, write_data


def helper(interval):
    data = omni(interval)

    fname = data_dir / f"interval_{interval}.h5"
    h5f = h5.File(fname, "a")
    for key, value in data.items():
        if (where := f"/{key}") in h5f:
            del h5f[where]

        if isinstance(value, u.Quantity):
            h5d = h5f.create_dataset(where, data=value.value)
            h5d.attrs["unit"] = str(value.unit)
        else:
            h5f.create_dataset(where, data=value)

    print(f"Saved OMNI data for interval {interval}", flush=True)


intervals = range(read_num_intervals())
instrument = "omni"
data_dir = lib.h5_dir / instrument
data_dir.mkdir(parents=True, exist_ok=True)
for intv in intervals:
    h5.File(data_dir / f"interval_{intv}.h5", "w")

with Pool(8) as pool:
    for _ in pool.uimap(helper, intervals):
        pass
