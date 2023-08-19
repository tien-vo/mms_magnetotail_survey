import os
from itertools import product

import h5py as h5
from pathos.pools import ProcessPool as Pool

import lib
from lib.load import mec
from lib.utils import read_num_intervals, write_data


def helper(probe, interval):
    data = mec(probe, interval, drate="srvy")
    write_data(probe, interval, instrument, data)
    print(f"MMS{probe}: Saved MEC data for interval {interval}", flush=True)


probes = range(1, 2)
intervals = range(read_num_intervals())
instrument = "mec"
for probe in probes:
    where = lib.h5_dir / f"mms{probe}" / instrument
    where.mkdir(parents=True, exist_ok=True)
    for intv in intervals:
        h5.File(where / f"interval_{intv}.h5", "w")

with Pool(8) as pool:
    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
        pass
