import os
from itertools import product

import h5py as h5
from pathos.pools import ProcessPool as Pool

import lib
from lib.load import feeps
from lib.utils import read_num_intervals, write_data


def helper(probe, interval):
    for species in ["ion", "elc"]:
        data = feeps(probe, interval, drate="srvy", species=species)
        write_data(probe, interval, f"{species}-feeps", data)

    print(
        f"MMS{probe}: Saved {species} FEEPS data for interval {interval}",
        flush=True,
    )


probes = range(1, 5)
intervals = range(read_num_intervals())
for probe in probes:
    for species in ["ion", "elc"]:
        where = lib.h5_dir / f"mms{probe}" / f"{species}-feeps"
        where.mkdir(parents=True, exist_ok=True)
        for intv in intervals:
            h5.File(where / f"interval_{intv}.h5", "w")

# with Pool(8) as pool:
#    for _ in pool.uimap(lambda args: helper(*args), product(probes, intervals)):
#        pass
